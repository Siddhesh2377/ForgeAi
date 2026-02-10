use std::path::{Path, PathBuf};

use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::commands::{detect_gpu, find_python};
use crate::model::error::ModelError;
use super::config::{TrainingDepsStatus, TrainingProgress};

/// Get the training tools directory.
pub fn get_training_dir(app: &AppHandle) -> Result<PathBuf, ModelError> {
    let data_dir = app.path().app_data_dir()
        .map_err(|e| ModelError::TrainingError(format!("Cannot resolve app data dir: {}", e)))?;
    Ok(data_dir.join("tools").join("training"))
}

/// Get the venv python binary path.
pub fn get_venv_python(training_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        training_dir.join("venv").join("Scripts").join("python.exe")
    } else {
        training_dir.join("venv").join("bin").join("python3")
    }
}

/// Check if all training dependencies are installed.
pub fn check_training_deps(training_dir: &Path) -> TrainingDepsStatus {
    let python = find_python();
    let python_found = python.is_some();
    let python_version = python.map(|(_, v)| v);

    let venv_python = get_venv_python(training_dir);
    let venv_ready = venv_python.exists();

    let mut missing_packages = vec![];
    let mut packages_ready = false;
    let mut torch_version = None;
    let mut cuda_available = false;
    let mut cuda_version = None;

    if venv_ready {
        let required = [
            "torch", "transformers", "peft", "datasets", "trl", "accelerate",
        ];

        for pkg in &required {
            let output = std::process::Command::new(&venv_python)
                .args(["-c", &format!("import {}", pkg)])
                .output();
            match output {
                Ok(o) if o.status.success() => {}
                _ => missing_packages.push(pkg.to_string()),
            }
        }

        // Optional: bitsandbytes (may not be available on all platforms)
        let _ = std::process::Command::new(&venv_python)
            .args(["-c", "import bitsandbytes"])
            .output();

        packages_ready = missing_packages.is_empty();

        // Check torch version and CUDA
        if packages_ready {
            if let Ok(output) = std::process::Command::new(&venv_python)
                .args(["-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda or 'none')"])
                .output()
            {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let lines: Vec<&str> = stdout.trim().lines().collect();
                    if lines.len() >= 3 {
                        torch_version = Some(lines[0].to_string());
                        cuda_available = lines[1].trim() == "True";
                        let cv = lines[2].trim();
                        if cv != "none" && cv != "None" {
                            cuda_version = Some(cv.to_string());
                        }
                    }
                }
            }
        }
    }

    let ready = python_found && venv_ready && packages_ready;

    TrainingDepsStatus {
        python_found,
        python_version,
        venv_ready,
        packages_ready,
        missing_packages,
        cuda_available,
        cuda_version,
        torch_version,
        ready,
    }
}

/// Map a CUDA version string (e.g. "13.1", "12.4") to PyTorch wheel index URLs,
/// ordered from best match to fallback.
fn cuda_wheel_urls(cuda_version: &str) -> Vec<String> {
    let parts: Vec<u32> = cuda_version
        .split('.')
        .filter_map(|s| s.parse().ok())
        .collect();
    let (major, minor) = (
        parts.first().copied().unwrap_or(0),
        parts.get(1).copied().unwrap_or(0),
    );

    // Known PyTorch CUDA wheel tags — newest first.
    // CUDA is backwards-compatible within a major version, so we pick the
    // highest tag that does not exceed the driver's CUDA version.
    let known: &[(u32, u32, &str)] = &[
        (13, 0, "cu130"),
        (12, 8, "cu128"),
        (12, 6, "cu126"),
        (12, 4, "cu124"),
        (12, 1, "cu121"),
        (11, 8, "cu118"),
    ];

    let mut urls: Vec<String> = Vec::new();
    for &(maj, min, tag) in known {
        if major > maj || (major == maj && minor >= min) {
            urls.push(format!("https://download.pytorch.org/whl/{}", tag));
        }
    }
    urls
}

/// Run a command, streaming stdout/stderr as `training:setup-log` events.
async fn run_cmd_logged(
    app: &AppHandle,
    cmd: &Path,
    args: &[&str],
) -> Result<bool, ModelError> {
    let mut child = tokio::process::Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| ModelError::TrainingError(format!("Failed to spawn: {}", e)))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let app1 = app.clone();
    let h1 = tokio::spawn(async move {
        if let Some(out) = stdout {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = app1.emit("training:setup-log", &line);
            }
        }
    });

    let app2 = app.clone();
    let h2 = tokio::spawn(async move {
        if let Some(err) = stderr {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = app2.emit("training:setup-log", &line);
            }
        }
    });

    let status = child.wait().await
        .map_err(|e| ModelError::TrainingError(format!("Process wait failed: {}", e)))?;

    let _ = h1.await;
    let _ = h2.await;

    Ok(status.success())
}

/// Set up the training Python environment.
pub async fn setup_training_env(app: &AppHandle) -> Result<(), ModelError> {
    let training_dir = get_training_dir(app)?;
    std::fs::create_dir_all(&training_dir).map_err(ModelError::IoError)?;

    let (python_cmd, _) = find_python().ok_or_else(|| ModelError::TrainingError(
        "Python 3 not found. Please install Python 3.10+.".into(),
    ))?;

    // 1. Create venv
    emit_progress(app, "Creating Python environment...", 5.0);

    let venv_dir = training_dir.join("venv");
    if !venv_dir.exists() {
        let status = tokio::process::Command::new(&python_cmd)
            .args(["-m", "venv", &venv_dir.to_string_lossy()])
            .status()
            .await
            .map_err(|e| ModelError::TrainingError(format!("Failed to create venv: {}", e)))?;

        if !status.success() {
            return Err(ModelError::TrainingError(
                "Failed to create Python venv. Ensure python3-venv is installed.".into(),
            ));
        }
    }

    let venv_python = get_venv_python(&training_dir);

    // 2. Upgrade pip
    emit_progress(app, "Upgrading pip...", 10.0);
    let _ = run_cmd_logged(app, &venv_python, &[
        "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check",
    ]).await;

    // 3. Install PyTorch (GPU or CPU)
    // On Linux, the default PyPI `torch` is CPU-only. CUDA wheels require
    // the official PyTorch wheel index (download.pytorch.org/whl/cuXXX).
    // We use --force-reinstall so that re-running setup replaces a CPU
    // torch with CUDA torch if a GPU is now available.
    let gpu = detect_gpu();
    let mut torch_installed = false;

    if gpu.has_nvidia {
        if let Some(ref cuda_ver) = gpu.cuda_version {
            let urls = cuda_wheel_urls(cuda_ver);
            for (i, url) in urls.iter().enumerate() {
                let tag = url.rsplit('/').next().unwrap_or("cuda");
                emit_progress(
                    app,
                    &format!("Installing PyTorch with CUDA {} ({}/{})...", tag, i + 1, urls.len()),
                    15.0 + (i as f64) * 3.0,
                );
                let url_str = url.as_str();
                let success = run_cmd_logged(app, &venv_python, &[
                    "-m", "pip", "install", "torch",
                    "--index-url", url_str,
                    "--force-reinstall",
                    "--disable-pip-version-check",
                ]).await.ok().unwrap_or(false);
                if success {
                    torch_installed = true;
                    break;
                }
            }
        }

        if !torch_installed {
            emit_progress(app, "CUDA wheels not available, falling back to CPU PyTorch...", 22.0);
            install_torch_cpu(app, &venv_python).await?;
        }
    } else {
        emit_progress(app, "Installing PyTorch CPU...", 15.0);
        install_torch_cpu(app, &venv_python).await?;
    }

    // 4. Install training packages
    let packages = [
        ("transformers", 40.0),
        ("peft", 50.0),
        ("trl", 60.0),
        ("datasets", 70.0),
        ("accelerate", 78.0),
        ("sentencepiece", 82.0),
        ("safetensors", 85.0),
    ];

    for (pkg, pct) in &packages {
        emit_progress(app, &format!("Installing {}...", pkg), *pct);
        let success = run_cmd_logged(app, &venv_python, &[
            "-m", "pip", "install", pkg, "--disable-pip-version-check",
        ]).await.map_err(|e| ModelError::TrainingError(format!("Failed to install {}: {}", pkg, e)))?;

        if !success {
            return Err(ModelError::TrainingError(format!(
                "Failed to install {}. Check your internet connection.", pkg
            )));
        }
    }

    // 5. Try bitsandbytes (optional, may fail on some platforms)
    emit_progress(app, "Installing bitsandbytes (optional, for QLoRA)...", 90.0);
    let _ = run_cmd_logged(app, &venv_python, &[
        "-m", "pip", "install", "bitsandbytes", "--disable-pip-version-check",
    ]).await;

    emit_progress(app, "Training environment ready.", 100.0);

    Ok(())
}

async fn install_torch_cpu(app: &AppHandle, venv_python: &Path) -> Result<(), ModelError> {
    let success = run_cmd_logged(app, venv_python, &[
        "-m", "pip", "install", "torch",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--disable-pip-version-check",
    ]).await.map_err(|e| ModelError::TrainingError(format!("Failed to install PyTorch: {}", e)))?;

    if !success {
        return Err(ModelError::TrainingError(
            "Failed to install PyTorch CPU. Check your internet connection.".into(),
        ));
    }
    Ok(())
}

fn emit_progress(app: &AppHandle, message: &str, percent: f64) {
    let _ = app.emit(
        "training:setup-progress",
        TrainingProgress {
            stage: "setup".into(),
            message: message.into(),
            percent,
            epoch: None,
            step: None,
            total_steps: None,
            loss: None,
            learning_rate: None,
            eta_seconds: None,
            gpu_memory_used_mb: None,
        },
    );
}
