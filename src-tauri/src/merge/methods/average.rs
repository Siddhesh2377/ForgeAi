use candle_core::Tensor;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

pub struct AverageMerge;

impl MergeStrategy for AverageMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        _params: &MethodParams,
        _base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        if tensors.is_empty() {
            return Err(ModelError::MergeError("No tensors to average".into()));
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].0.clone());
        }

        // Normalize weights
        let total_weight: f64 = tensors.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return Err(ModelError::MergeError("Total weight must be positive".into()));
        }

        let mut result = (&tensors[0].0 * (tensors[0].1 / total_weight))
            .map_err(|e| ModelError::CandleError(e.to_string()))?;

        for (tensor, weight) in &tensors[1..] {
            let scaled = (tensor * (*weight / total_weight))
                .map_err(|e| ModelError::CandleError(e.to_string()))?;
            result = (&result + &scaled)
                .map_err(|e| ModelError::CandleError(e.to_string()))?;
        }

        Ok(result)
    }

    fn name(&self) -> &'static str { "Average" }
    fn requires_base(&self) -> bool { false }
    fn min_parents(&self) -> usize { 2 }
}
