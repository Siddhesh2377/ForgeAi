use candle_core::Tensor;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

/// Frankenmerge: layer stacking from different models.
/// In practice, the planner handles layer assignment as Copy operations.
/// This merge function is a fallback for tensors that don't have explicit assignments.
pub struct FrankenmergeMerge;

impl MergeStrategy for FrankenmergeMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        _params: &MethodParams,
        _base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        // Frankenmerge operates at the layer assignment level.
        // If we reach this merge function, just take the first tensor (highest weight).
        if tensors.is_empty() {
            return Err(ModelError::MergeError("No tensors provided".into()));
        }

        // Pick the tensor with highest weight
        let best = tensors
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        Ok(best.0.clone())
    }

    fn name(&self) -> &'static str { "Frankenmerge" }
    fn requires_base(&self) -> bool { false }
    fn min_parents(&self) -> usize { 2 }
}
