use candle_core::Tensor;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

pub struct TaskArithmeticMerge;

impl MergeStrategy for TaskArithmeticMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        params: &MethodParams,
        base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        let base = base_tensor.ok_or_else(|| {
            ModelError::MergeError("Task Arithmetic requires a base model".into())
        })?;

        let scaling = params.scaling.unwrap_or(1.0);
        let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());

        // Compute task vectors: T_i - T_base
        // Then sum: T_base + scaling * sum(w_i * (T_i - T_base))
        let mut task_vector_sum = Tensor::zeros_like(base).map_err(map_err)?;

        let total_weight: f64 = tensors.iter().map(|(_, w)| w).sum();
        let norm = if total_weight > 0.0 { total_weight } else { 1.0 };

        for (tensor, weight) in tensors {
            let task_vec = (tensor - base).map_err(map_err)?;
            let weighted = (&task_vec * (*weight / norm)).map_err(map_err)?;
            task_vector_sum = (&task_vector_sum + &weighted).map_err(map_err)?;
        }

        let scaled = (&task_vector_sum * scaling).map_err(map_err)?;
        let result = (base + &scaled).map_err(map_err)?;

        Ok(result)
    }

    fn name(&self) -> &'static str { "Task Arithmetic" }
    fn requires_base(&self) -> bool { true }
    fn min_parents(&self) -> usize { 2 }
}
