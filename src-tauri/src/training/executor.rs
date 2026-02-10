use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::model::error::ModelError;
use super::config::{TrainingConfig, TrainingMethod, TrainingProgress, TrainingResult};
use super::scripts;
use super::venv;

/// Run a training job as a Python subprocess.
pub async fn run_training(
    app: AppHandle,
    config: TrainingConfig,
    cancel: Arc<AtomicBool>,
    pid_store: Arc<Mutex<Option<u32>>>,
) -> Result<TrainingResult, ModelError> {
    let training_dir = venv::get_training_dir(&app)?;
    let venv_python = venv::get_venv_python(&training_dir);

    if !venv_python.exists() {
        return Err(ModelError::TrainingError(
            "Training environment not set up. Please install dependencies first.".into(),
        ));
    }

    // Generate the script
    let script = match config.method {
        TrainingMethod::Dpo => scripts::generate_dpo_script(&config),
        _ => scripts::generate_sft_script(&config),
    };

    let script_path = training_dir.join("train_script.py");
    std::fs::write(&script_path, &script)
        .map_err(|e| ModelError::TrainingError(format!("Failed to write training script: {}", e)))?;

    // Create output directory
    std::fs::create_dir_all(&config.output_path)
        .map_err(|e| ModelError::TrainingError(format!("Failed to create output directory: {}", e)))?;

    // Spawn subprocess
    let mut child = tokio::process::Command::new(&venv_python)
        .arg(&script_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| ModelError::TrainingError(format!("Failed to start training: {}", e)))?;

    // Store PID for cancellation
    if let Some(pid) = child.id() {
        *pid_store.lock().unwrap() = Some(pid);
    }

    let stdout = child.stdout.take()
        .ok_or_else(|| ModelError::TrainingError("Failed to capture stdout".into()))?;
    let stderr = child.stderr.take()
        .ok_or_else(|| ModelError::TrainingError("Failed to capture stderr".into()))?;

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();

    let mut final_loss: Option<f64> = None;
    let mut final_result: Option<serde_json::Value> = None;
    let mut stderr_lines: Vec<String> = Vec::new();

    loop {
        if cancel.load(Ordering::Relaxed) {
            // Kill the process
            let _ = child.kill().await;
            *pid_store.lock().unwrap() = None;
            return Err(ModelError::TrainingCancelled);
        }

        tokio::select! {
            line = stdout_reader.next_line() => {
                match line {
                    Ok(Some(text)) => {
                        if let Some(progress) = parse_progress(&text) {
                            if let Some(loss) = progress.loss {
                                final_loss = Some(loss);
                            }
                            let _ = app.emit("training:progress", &progress);
                        } else if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                            if json.get("type").and_then(|v| v.as_str()) == Some("result") {
                                final_result = Some(json);
                            } else if json.get("type").and_then(|v| v.as_str()) == Some("status") {
                                let stage = json.get("stage").and_then(|v| v.as_str()).unwrap_or("unknown");
                                let message = json.get("message").and_then(|v| v.as_str()).unwrap_or("");
                                let _ = app.emit("training:progress", TrainingProgress {
                                    stage: stage.into(),
                                    message: message.into(),
                                    percent: if stage == "complete" { 100.0 } else { -1.0 },
                                    epoch: None, step: None, total_steps: None,
                                    loss: None, learning_rate: None,
                                    eta_seconds: None, gpu_memory_used_mb: None,
                                });
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(_) => break,
                }
            }
            line = stderr_reader.next_line() => {
                match line {
                    Ok(Some(text)) => {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            stderr_lines.push(trimmed.to_string());
                        }
                    }
                    Ok(None) => {}
                    Err(_) => {}
                }
            }
        }
    }

    let status = child.wait().await
        .map_err(|e| ModelError::TrainingError(format!("Failed to wait for process: {}", e)))?;

    *pid_store.lock().unwrap() = None;

    if !status.success() {
        // Extract meaningful error from stderr, filtering noise
        let error_msg = extract_error_message(&stderr_lines);
        return Err(ModelError::TrainingError(format!(
            "Training failed (exit code {}): {}",
            status.code().unwrap_or(-1),
            if error_msg.is_empty() { "Unknown error".to_string() } else { error_msg }
        )));
    }

    // Build result from script output or defaults
    let output_size = dir_size(&config.output_path);
    let output_size_display = format_size(output_size);

    if let Some(ref res) = final_result {
        Ok(TrainingResult {
            output_path: config.output_path.clone(),
            output_size: res.get("output_size").and_then(|v| v.as_u64()).unwrap_or(output_size),
            output_size_display: format_size(res.get("output_size").and_then(|v| v.as_u64()).unwrap_or(output_size)),
            method: config.method.to_string(),
            epochs_completed: res.get("epochs_completed").and_then(|v| v.as_u64()).unwrap_or(config.epochs as u64) as u32,
            final_loss: res.get("final_loss").and_then(|v| v.as_f64()).or(final_loss),
            adapter_merged: config.merge_adapter,
        })
    } else {
        Ok(TrainingResult {
            output_path: config.output_path.clone(),
            output_size,
            output_size_display,
            method: config.method.to_string(),
            epochs_completed: config.epochs,
            final_loss,
            adapter_merged: config.merge_adapter,
        })
    }
}

fn parse_progress(line: &str) -> Option<TrainingProgress> {
    let json: serde_json::Value = serde_json::from_str(line).ok()?;
    if json.get("type")?.as_str()? != "progress" {
        return None;
    }

    Some(TrainingProgress {
        stage: "training".into(),
        message: format!(
            "Step {}/{}",
            json.get("step").and_then(|v| v.as_u64()).unwrap_or(0),
            json.get("total_steps").and_then(|v| v.as_u64()).unwrap_or(0),
        ),
        percent: json.get("percent").and_then(|v| v.as_f64()).unwrap_or(0.0),
        epoch: json.get("epoch").and_then(|v| v.as_f64()).map(|e| e as u32),
        step: json.get("step").and_then(|v| v.as_u64()),
        total_steps: json.get("total_steps").and_then(|v| v.as_u64()),
        loss: json.get("loss").and_then(|v| v.as_f64()),
        learning_rate: json.get("lr").and_then(|v| v.as_f64()),
        eta_seconds: json.get("eta").and_then(|v| v.as_u64()),
        gpu_memory_used_mb: json.get("gpu_mem").and_then(|v| v.as_u64()),
    })
}

fn dir_size(path: &str) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += dir_size(&entry.path().to_string_lossy());
                }
            }
        }
    }
    total
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Extract a useful error message from stderr lines.
/// Filters out warnings, tqdm bars, and noise; returns the traceback + final error.
fn extract_error_message(lines: &[String]) -> String {
    // Look for the last traceback block
    let mut traceback_start = None;
    for (i, line) in lines.iter().enumerate() {
        if line.starts_with("Traceback (most recent call last)") {
            traceback_start = Some(i);
        }
    }

    // If we found a traceback, return from there to the end (up to 30 lines)
    if let Some(start) = traceback_start {
        let tb_lines: Vec<&str> = lines[start..]
            .iter()
            .map(|s| s.as_str())
            .take(30)
            .collect();
        return tb_lines.join("\n");
    }

    // Otherwise, look for common error patterns in the last lines
    let meaningful: Vec<&str> = lines
        .iter()
        .rev()
        .filter(|l| {
            !l.starts_with("WARNING")
                && !l.contains("FutureWarning")
                && !l.contains("UserWarning")
                && !l.contains("| 0/")   // tqdm progress bars
                && !l.starts_with("  0%")
                && !l.contains("it/s]")
                && !l.contains("it/s,")
                && !l.is_empty()
        })
        .take(10)
        .map(|s| s.as_str())
        .collect();

    if meaningful.is_empty() {
        // Fall back to last non-empty stderr line
        lines.iter().rev()
            .find(|l| !l.is_empty())
            .cloned()
            .unwrap_or_default()
    } else {
        let mut result: Vec<&str> = meaningful;
        result.reverse();
        result.join("\n")
    }
}
