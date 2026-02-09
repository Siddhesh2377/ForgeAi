use candle_core::Tensor;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

/// Component Merge: route attention/MLP/norm components to different parents.
/// In practice, the planner handles component routing as Copy operations.
/// This merge function is the fallback for tensors without component overrides.
pub struct ComponentMergeMerge;

impl MergeStrategy for ComponentMergeMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        _params: &MethodParams,
        _base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        if tensors.is_empty() {
            return Err(ModelError::MergeError("No tensors for component merge".into()));
        }

        // Fallback: weighted average for unrouted tensors
        if tensors.len() == 1 {
            return Ok(tensors[0].0.clone());
        }

        let total_weight: f64 = tensors.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return Ok(tensors[0].0.clone());
        }

        let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());

        let mut result = (&tensors[0].0 * (tensors[0].1 / total_weight)).map_err(map_err)?;
        for (tensor, weight) in &tensors[1..] {
            let scaled = (tensor * (*weight / total_weight)).map_err(map_err)?;
            result = (&result + &scaled).map_err(map_err)?;
        }

        Ok(result)
    }

    fn name(&self) -> &'static str { "Component Merge" }
    fn requires_base(&self) -> bool { false }
    fn min_parents(&self) -> usize { 2 }
}
