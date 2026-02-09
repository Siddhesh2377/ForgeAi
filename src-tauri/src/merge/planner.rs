use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::error::ModelError;
use crate::model::inspect;

use super::config::{ComponentType, MergeConfig, MergeMethod};
use super::registry::ParentRegistry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorOperation {
    /// Direct copy from one parent
    Copy {
        tensor_name: String,
        parent_id: String,
    },
    /// Merge multiple parent tensors using configured method
    Merge {
        tensor_name: String,
        parent_ids: Vec<String>,
        weights: Vec<f64>,
    },
    /// Create new tensors (e.g., MoE router)
    Synthesize {
        tensor_name: String,
        shape: Vec<usize>,
        strategy: String,
    },
    /// Copy metadata/config files
    CopyMetadata {
        parent_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMergePlan {
    pub operations: Vec<TensorOperation>,
    pub total_tensors: usize,
    pub method: MergeMethod,
    pub estimated_output_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergePreview {
    pub total_operations: usize,
    pub copy_operations: usize,
    pub merge_operations: usize,
    pub synthesize_operations: usize,
    pub estimated_output_bytes: u64,
    pub estimated_output_display: String,
    pub tensor_sources: Vec<TensorSourceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSourceInfo {
    pub tensor_name: String,
    pub operation: String,
    pub source_parents: Vec<String>,
}

pub fn build_plan(
    config: &MergeConfig,
    registry: &ParentRegistry,
) -> Result<TensorMergePlan, ModelError> {
    let mut operations = Vec::new();

    // Build lookup maps
    let _parent_weight_map: HashMap<&str, f64> = config
        .parents
        .iter()
        .map(|pw| (pw.parent_id.as_str(), pw.weight))
        .collect();

    let tensor_override_map: HashMap<&str, &str> = config
        .tensor_overrides
        .iter()
        .map(|to| (to.tensor_name.as_str(), to.parent_id.as_str()))
        .collect();

    let layer_assignment_map: HashMap<u64, &str> = config
        .layer_assignments
        .iter()
        .map(|la| (la.layer_index, la.source_parent_id.as_str()))
        .collect();

    // Get all unique tensor names across parents
    let shared = registry.shared_tensor_names();
    let all_parents = registry.all();

    if all_parents.is_empty() {
        return Err(ModelError::MergeError("No parents loaded".to_string()));
    }

    // For methods that just copy layers (Frankenmerge, Passthrough)
    let is_layer_copy = matches!(config.method, MergeMethod::Frankenmerge | MergeMethod::Passthrough);

    // Collect all tensor names from first parent as baseline
    let primary_parent = &all_parents[0];
    let tensor_names = &primary_parent.compat.tensor_names;

    for tensor_name in tensor_names {
        // Priority 1: Tensor-level override
        if let Some(&source_id) = tensor_override_map.get(tensor_name.as_str()) {
            operations.push(TensorOperation::Copy {
                tensor_name: tensor_name.clone(),
                parent_id: source_id.to_string(),
            });
            continue;
        }

        // Priority 2: Component-level override
        let component = inspect::classify_tensor(tensor_name);
        let layer_idx = inspect::extract_layer_index(tensor_name);

        if let Some(idx) = layer_idx {
            let component_override = config.component_overrides.iter().find(|co| {
                idx >= co.layer_start
                    && idx <= co.layer_end
                    && match (&co.component, component) {
                        (ComponentType::Attention, "attention") => true,
                        (ComponentType::Mlp, "mlp") => true,
                        (ComponentType::Norm, "norm") => true,
                        _ => false,
                    }
            });

            if let Some(co) = component_override {
                operations.push(TensorOperation::Copy {
                    tensor_name: tensor_name.clone(),
                    parent_id: co.parent_id.clone(),
                });
                continue;
            }

            // Priority 3: Layer assignment
            if let Some(&source_id) = layer_assignment_map.get(&idx) {
                operations.push(TensorOperation::Copy {
                    tensor_name: tensor_name.clone(),
                    parent_id: source_id.to_string(),
                });
                continue;
            }
        }

        // Priority 4: Global method
        if is_layer_copy {
            // For layer-copy methods without assignment, use first parent
            operations.push(TensorOperation::Copy {
                tensor_name: tensor_name.clone(),
                parent_id: primary_parent.id.clone(),
            });
        } else if config.method == MergeMethod::MoeConversion {
            // MoE: MLP tensors become expert copies, attention stays merged
            if component == "mlp" {
                // Each parent's MLP becomes an expert
                for (i, parent) in all_parents.iter().enumerate() {
                    let expert_name = tensor_name.replace("mlp", &format!("experts.{}.mlp", i));
                    operations.push(TensorOperation::Copy {
                        tensor_name: expert_name,
                        parent_id: parent.id.clone(),
                    });
                }
                // Synthesize router
                if tensor_name.contains("gate_proj") || tensor_name.contains("ffn_gate") {
                    operations.push(TensorOperation::Synthesize {
                        tensor_name: tensor_name.replace("gate_proj", "router.weight")
                            .replace("ffn_gate", "router.weight"),
                        shape: vec![all_parents.len(), 1], // placeholder shape
                        strategy: "random_init".to_string(),
                    });
                }
            } else if shared.contains(tensor_name) {
                // Non-MLP tensors: standard merge
                let parent_ids: Vec<String> = config.parents.iter().map(|p| p.parent_id.clone()).collect();
                let weights: Vec<f64> = config.parents.iter().map(|p| p.weight).collect();
                operations.push(TensorOperation::Merge {
                    tensor_name: tensor_name.clone(),
                    parent_ids,
                    weights,
                });
            } else {
                operations.push(TensorOperation::Copy {
                    tensor_name: tensor_name.clone(),
                    parent_id: primary_parent.id.clone(),
                });
            }
        } else if shared.contains(tensor_name) {
            // Standard merge
            let parent_ids: Vec<String> = config.parents.iter().map(|p| p.parent_id.clone()).collect();
            let weights: Vec<f64> = config.parents.iter().map(|p| p.weight).collect();
            operations.push(TensorOperation::Merge {
                tensor_name: tensor_name.clone(),
                parent_ids,
                weights,
            });
        } else {
            // Tensor only exists in first parent, copy it
            operations.push(TensorOperation::Copy {
                tensor_name: tensor_name.clone(),
                parent_id: primary_parent.id.clone(),
            });
        }
    }

    // Add metadata copy from base or first parent
    let metadata_source = config
        .base_parent_id
        .as_deref()
        .unwrap_or(&primary_parent.id);
    operations.push(TensorOperation::CopyMetadata {
        parent_id: metadata_source.to_string(),
    });

    let total_tensors = operations.len();

    // Rough size estimate: sum of first parent's file size
    let estimated_output_bytes = primary_parent.file_size;

    Ok(TensorMergePlan {
        operations,
        total_tensors,
        method: config.method,
        estimated_output_bytes,
    })
}

pub fn preview_plan(
    config: &MergeConfig,
    registry: &ParentRegistry,
) -> Result<MergePreview, ModelError> {
    let plan = build_plan(config, registry)?;

    let mut copy_ops = 0;
    let mut merge_ops = 0;
    let mut synth_ops = 0;
    let mut tensor_sources = Vec::new();

    for op in &plan.operations {
        match op {
            TensorOperation::Copy { tensor_name, parent_id } => {
                copy_ops += 1;
                tensor_sources.push(TensorSourceInfo {
                    tensor_name: tensor_name.clone(),
                    operation: "copy".to_string(),
                    source_parents: vec![parent_id.clone()],
                });
            }
            TensorOperation::Merge { tensor_name, parent_ids, .. } => {
                merge_ops += 1;
                tensor_sources.push(TensorSourceInfo {
                    tensor_name: tensor_name.clone(),
                    operation: "merge".to_string(),
                    source_parents: parent_ids.clone(),
                });
            }
            TensorOperation::Synthesize { tensor_name, .. } => {
                synth_ops += 1;
                tensor_sources.push(TensorSourceInfo {
                    tensor_name: tensor_name.clone(),
                    operation: "synthesize".to_string(),
                    source_parents: vec![],
                });
            }
            TensorOperation::CopyMetadata { .. } => {}
        }
    }

    let estimated = plan.estimated_output_bytes;

    Ok(MergePreview {
        total_operations: plan.total_tensors,
        copy_operations: copy_ops,
        merge_operations: merge_ops,
        synthesize_operations: synth_ops,
        estimated_output_bytes: estimated,
        estimated_output_display: crate::model::format_file_size(estimated),
        tensor_sources,
    })
}
