use candle_core::Tensor;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

/// Tensor Surgery: per-tensor source mapping from parents.
/// The planner handles individual tensor assignment as Copy operations.
/// This merge function is the fallback.
pub struct TensorSurgeryMerge;

impl MergeStrategy for TensorSurgeryMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        _params: &MethodParams,
        _base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        if tensors.is_empty() {
            return Err(ModelError::MergeError("No tensors for tensor surgery".into()));
        }

        // Pick the tensor with highest weight
        let best = tensors
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        Ok(best.0.clone())
    }

    fn name(&self) -> &'static str { "Tensor Surgery" }
    fn requires_base(&self) -> bool { false }
    fn min_parents(&self) -> usize { 2 }
}
