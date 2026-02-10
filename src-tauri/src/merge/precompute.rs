use crate::model::error::ModelError;

use super::planner::TensorOperation;
use super::registry::ParentRegistry;

#[derive(Debug, Clone)]
pub struct OutputTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub f32_byte_size: u64,
    pub data_offset: u64,
}

pub struct OutputManifest {
    pub tensors: Vec<OutputTensorInfo>,
    pub total_data_bytes: u64,
}

fn compute_f32_byte_size(shape: &[usize]) -> u64 {
    let elem_count: usize = shape.iter().product();
    (elem_count * 4) as u64
}

pub fn build_output_manifest(
    operations: &[TensorOperation],
    registry: &ParentRegistry,
) -> Result<OutputManifest, ModelError> {
    let mut tensors = Vec::new();
    let mut current_offset: u64 = 0;

    for op in operations {
        match op {
            TensorOperation::Copy { tensor_name, parent_id } => {
                let parent = registry.get(parent_id).ok_or_else(|| {
                    ModelError::ParentNotFound(parent_id.clone())
                })?;
                let shape = parent.compat.tensor_shape(tensor_name)
                    .ok_or_else(|| ModelError::MergeError(
                        format!("Tensor '{}' not found in parent '{}'", tensor_name, parent.name)
                    ))?
                    .to_vec();
                let byte_size = compute_f32_byte_size(&shape);
                tensors.push(OutputTensorInfo {
                    name: tensor_name.clone(),
                    shape,
                    f32_byte_size: byte_size,
                    data_offset: current_offset,
                });
                current_offset += byte_size;
            }
            TensorOperation::Merge { tensor_name, parent_ids, .. } => {
                let first_pid = &parent_ids[0];
                let parent = registry.get(first_pid).ok_or_else(|| {
                    ModelError::ParentNotFound(first_pid.clone())
                })?;
                let shape = parent.compat.tensor_shape(tensor_name)
                    .ok_or_else(|| ModelError::MergeError(
                        format!("Tensor '{}' not found in parent '{}'", tensor_name, parent.name)
                    ))?
                    .to_vec();
                let byte_size = compute_f32_byte_size(&shape);
                tensors.push(OutputTensorInfo {
                    name: tensor_name.clone(),
                    shape,
                    f32_byte_size: byte_size,
                    data_offset: current_offset,
                });
                current_offset += byte_size;
            }
            TensorOperation::Synthesize { tensor_name, shape, .. } => {
                let byte_size = compute_f32_byte_size(shape);
                tensors.push(OutputTensorInfo {
                    name: tensor_name.clone(),
                    shape: shape.clone(),
                    f32_byte_size: byte_size,
                    data_offset: current_offset,
                });
                current_offset += byte_size;
            }
            TensorOperation::CopyMetadata { .. } => {}
        }
    }

    Ok(OutputManifest {
        total_data_bytes: current_offset,
        tensors,
    })
}
