use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::model::error::ModelError;

use super::config::{MergeConfig, OutputFormat};
use super::methods;
use super::output;
use super::planner::{TensorMergePlan, TensorOperation};
use super::precompute;
use super::projections;
use super::registry::ParentRegistry;
use super::tensor_io;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeProgress {
    pub stage: String,
    pub percent: f64,
    pub message: String,
    pub current_tensor: Option<String>,
    pub tensors_done: usize,
    pub tensors_total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergePhase {
    pub phase: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    pub output_path: String,
    pub output_size: u64,
    pub output_size_display: String,
    pub tensors_written: usize,
    pub method: String,
    pub copied_files: Vec<String>,
}

fn emit_progress(app: &AppHandle, progress: &MergeProgress) {
    let _ = app.emit("merge:progress", progress);
}

fn emit_phase(app: &AppHandle, phase: &str, detail: &str) {
    let _ = app.emit("merge:phase", MergePhase {
        phase: phase.to_string(),
        detail: detail.to_string(),
    });
}

pub fn execute_merge(
    app: &AppHandle,
    config: &MergeConfig,
    plan: &TensorMergePlan,
    registry: &ParentRegistry,
    cancel: Arc<AtomicBool>,
) -> Result<MergeResult, ModelError> {
    let strategy = methods::get_strategy(config.method);
    let total_ops = plan.operations.iter().filter(|op| !matches!(op, TensorOperation::CopyMetadata { .. })).count();
    let mut tensors_done = 0;

    // Phase 1: Validating
    emit_phase(app, "validating", "Checking compatibility");
    emit_progress(app, &MergeProgress {
        stage: "validating".into(),
        percent: 2.0,
        message: "Validating merge configuration...".into(),
        current_tensor: None,
        tensors_done: 0,
        tensors_total: total_ops,
    });

    if cancel.load(Ordering::Relaxed) {
        return Err(ModelError::MergeCancelled);
    }

    // Phase 2: Pre-compute output manifest (tensor shapes + offsets, no data loading)
    emit_phase(app, "planning", "Pre-computing tensor offsets");
    let manifest = precompute::build_output_manifest(&plan.operations, registry)?;

    emit_progress(app, &MergeProgress {
        stage: "planning".into(),
        percent: 7.0,
        message: format!("Plan: {} tensors, {} estimated output",
            manifest.tensors.len(),
            crate::model::format_file_size(manifest.total_data_bytes)),
        current_tensor: None,
        tensors_done: 0,
        tensors_total: total_ops,
    });

    if cancel.load(Ordering::Relaxed) {
        return Err(ModelError::MergeCancelled);
    }

    // Determine base parent
    let base_parent = config.base_parent_id.as_ref().and_then(|id| registry.get(id));
    let metadata_parent = base_parent.or_else(|| registry.all().first());
    let output_path = &config.output.path;

    // Resolve parent config dir for GGUF metadata
    let parent_config_dir = metadata_parent.map(|mp| {
        if mp.is_dir {
            mp.file_path.clone()
        } else {
            std::path::Path::new(&mp.file_path)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default()
        }
    });

    // Phase 3: Open streaming writer + merge loop
    emit_phase(app, "merging", "Processing tensors");

    let (actual_file_path, aux_target_dir, mut writer) = match config.output.format {
        OutputFormat::SafeTensors => {
            let dir = std::path::Path::new(output_path);
            std::fs::create_dir_all(dir).map_err(ModelError::IoError)?;
            let model_file = dir.join("model.safetensors");
            let file_str = model_file.to_string_lossy().to_string();
            let st_writer = output::StreamingSafeTensorsWriter::new(&file_str, &manifest)?;
            (file_str, output_path.clone(), output::StreamWriter::SafeTensors(st_writer))
        }
        OutputFormat::Gguf => {
            let mp = metadata_parent
                .ok_or_else(|| ModelError::MergeError("No parent for metadata".into()))?;

            let source_gguf = if matches!(mp.format, crate::model::ModelFormat::Gguf) {
                Some(mp.file_path.as_str())
            } else {
                None
            };

            let cfg_dir = if source_gguf.is_none() {
                parent_config_dir.as_deref()
            } else {
                None
            };

            let gguf_writer = output::StreamingGgufWriter::new(
                output_path,
                &manifest,
                &config.output.model_name,
                source_gguf,
                Some(&mp.compat),
                cfg_dir,
            )?;

            let aux_dir = std::path::Path::new(output_path)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            (output_path.clone(), aux_dir, output::StreamWriter::Gguf(gguf_writer))
        }
    };

    // Streaming merge loop — each tensor is written immediately and dropped
    for op in &plan.operations {
        if cancel.load(Ordering::Relaxed) {
            drop(writer);
            let _ = std::fs::remove_file(&actual_file_path);
            return Err(ModelError::MergeCancelled);
        }

        match op {
            TensorOperation::Copy { tensor_name, parent_id } => {
                let parent = registry.get(parent_id).ok_or_else(|| {
                    ModelError::ParentNotFound(parent_id.clone())
                })?;

                emit_progress(app, &MergeProgress {
                    stage: "merging".into(),
                    percent: 10.0 + (tensors_done as f64 / total_ops as f64) * 80.0,
                    message: format!("Copying {}", tensor_name),
                    current_tensor: Some(tensor_name.clone()),
                    tensors_done,
                    tensors_total: total_ops,
                });

                let tensor = tensor_io::load_tensor(parent, tensor_name)?;
                writer.write_tensor(&tensor)?;
                tensors_done += 1;
            }

            TensorOperation::Merge { tensor_name, parent_ids, weights } => {
                emit_progress(app, &MergeProgress {
                    stage: "merging".into(),
                    percent: 10.0 + (tensors_done as f64 / total_ops as f64) * 80.0,
                    message: format!("Merging {} ({} parents)", tensor_name, parent_ids.len()),
                    current_tensor: Some(tensor_name.clone()),
                    tensors_done,
                    tensors_total: total_ops,
                });

                let mut parent_tensors: Vec<(Tensor, f64)> = Vec::new();
                for (pid, weight) in parent_ids.iter().zip(weights.iter()) {
                    let parent = registry.get(pid).ok_or_else(|| {
                        ModelError::ParentNotFound(pid.clone())
                    })?;
                    let tensor = tensor_io::load_tensor(parent, tensor_name)?;
                    parent_tensors.push((tensor, *weight));
                }

                // Apply projection if strategy is set and shapes mismatch
                if let Some(ref proj_strategy) = config.projection_strategy {
                    if parent_tensors.len() >= 2 {
                        // Use first parent's shape as target
                        let target_shape: Vec<usize> = parent_tensors[0].0.dims().to_vec();
                        for i in 1..parent_tensors.len() {
                            if parent_tensors[i].0.dims() != target_shape.as_slice() {
                                let adapted = projections::adapt_tensor(
                                    &parent_tensors[i].0,
                                    &target_shape,
                                    proj_strategy,
                                )?;
                                parent_tensors[i].0 = adapted;
                            }
                        }
                    }
                }

                let base_tensor = if strategy.requires_base() {
                    if let Some(bp) = base_parent {
                        Some(tensor_io::load_tensor(bp, tensor_name)?)
                    } else if !parent_tensors.is_empty() {
                        Some(parent_tensors[0].0.clone())
                    } else {
                        None
                    }
                } else {
                    None
                };

                let merged = strategy.merge(
                    &parent_tensors,
                    &config.params,
                    base_tensor.as_ref(),
                )?;

                writer.write_tensor(&merged)?;
                tensors_done += 1;
            }

            TensorOperation::Synthesize { tensor_name, shape, strategy: synth_strategy } => {
                emit_progress(app, &MergeProgress {
                    stage: "merging".into(),
                    percent: 10.0 + (tensors_done as f64 / total_ops as f64) * 80.0,
                    message: format!("Synthesizing {}", tensor_name),
                    current_tensor: Some(tensor_name.clone()),
                    tensors_done,
                    tensors_total: total_ops,
                });

                let tensor = match synth_strategy.as_str() {
                    "random_init" => {
                        let num_elements: usize = shape.iter().product();
                        let data: Vec<f32> = (0..num_elements)
                            .map(|_| rand::random::<f32>() * 0.02 - 0.01)
                            .collect();
                        Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
                            .map_err(|e| ModelError::CandleError(e.to_string()))?
                    }
                    _ => {
                        Tensor::zeros(shape.as_slice(), DType::F32, &Device::Cpu)
                            .map_err(|e| ModelError::CandleError(e.to_string()))?
                    }
                };

                writer.write_tensor(&tensor)?;
                tensors_done += 1;
            }

            TensorOperation::CopyMetadata { .. } => {
                // Handled by auxiliary file copy below
            }
        }
    }

    // Phase 4: Finalize output
    emit_phase(app, "writing", "Finalizing output file");
    emit_progress(app, &MergeProgress {
        stage: "writing".into(),
        percent: 91.0,
        message: format!("Finalizing {} tensors...", tensors_done),
        current_tensor: None,
        tensors_done,
        tensors_total: total_ops,
    });

    writer.finish()?;

    // Phase 4b: Copy auxiliary files
    emit_phase(app, "copying", "Copying tokenizer and config files");
    let copied_files = copy_auxiliary_files(&aux_target_dir, registry, config.base_parent_id.as_deref());

    // Phase 5: Verifying
    emit_phase(app, "verifying", "Checking output integrity");
    emit_progress(app, &MergeProgress {
        stage: "verifying".into(),
        percent: 97.0,
        message: "Verifying output file...".into(),
        current_tensor: None,
        tensors_done: total_ops,
        tensors_total: total_ops,
    });

    let output_file = std::fs::metadata(&actual_file_path).map_err(ModelError::IoError)?;
    let output_size = output_file.len();

    emit_progress(app, &MergeProgress {
        stage: "complete".into(),
        percent: 100.0,
        message: format!("Merge complete: {}", crate::model::format_file_size(output_size)),
        current_tensor: None,
        tensors_done: total_ops,
        tensors_total: total_ops,
    });

    Ok(MergeResult {
        output_path: actual_file_path,
        output_size,
        output_size_display: crate::model::format_file_size(output_size),
        tensors_written: tensors_done,
        method: config.method.display_name().to_string(),
        copied_files,
    })
}

/// Files to copy from parent model directory to output directory.
const AUXILIARY_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "preprocessor_config.json",
];

/// Copy tokenizer/config files from the best source parent directory to the output directory.
fn copy_auxiliary_files(
    output_dir_path: &str,
    registry: &ParentRegistry,
    base_parent_id: Option<&str>,
) -> Vec<String> {
    use std::path::Path;

    let output_dir = Path::new(output_dir_path);
    if !output_dir.exists() {
        let _ = std::fs::create_dir_all(output_dir);
    }

    let source_dir = base_parent_id
        .and_then(|id| registry.get(id))
        .or_else(|| registry.all().iter().find(|p| p.is_dir))
        .or_else(|| registry.all().first());

    let source_parent = match source_dir {
        Some(p) => p,
        None => return vec![],
    };

    let source_path = Path::new(&source_parent.file_path);
    let source_dir = if source_parent.is_dir {
        source_path.to_path_buf()
    } else {
        match source_path.parent() {
            Some(dir) => dir.to_path_buf(),
            None => return vec![],
        }
    };

    let mut copied = Vec::new();

    for &filename in AUXILIARY_FILES {
        let src = source_dir.join(filename);
        let dst = output_dir.join(filename);
        if src.exists() && !dst.exists() {
            if let Ok(_) = std::fs::copy(&src, &dst) {
                copied.push(filename.to_string());
            }
        }
    }

    copied
}
