use candle_core::Tensor;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

/// Passthrough: Direct copy from a single parent (no merging).
pub struct PassthroughMerge;

impl MergeStrategy for PassthroughMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        _params: &MethodParams,
        _base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        if tensors.is_empty() {
            return Err(ModelError::MergeError("No tensor to pass through".into()));
        }
        Ok(tensors[0].0.clone())
    }

    fn name(&self) -> &'static str { "Passthrough" }
    fn requires_base(&self) -> bool { false }
    fn min_parents(&self) -> usize { 1 }
}
