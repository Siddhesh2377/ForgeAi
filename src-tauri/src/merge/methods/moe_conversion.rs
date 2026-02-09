use candle_core::Tensor;
use rand::Rng;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

/// MoE Conversion: Convert dense models to Mixture-of-Experts.
/// MLP layers from each parent become separate experts.
/// Non-MLP tensors are averaged. Router weights are randomly initialized.
/// The planner handles the structural transformation; this is the fallback merge.
pub struct MoeConversionMerge;

impl MergeStrategy for MoeConversionMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        _params: &MethodParams,
        _base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        if tensors.is_empty() {
            return Err(ModelError::MergeError("No tensors for MoE conversion".into()));
        }

        // For non-MLP tensors that reach this merge function, average them
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

    fn name(&self) -> &'static str { "MoE Conversion" }
    fn requires_base(&self) -> bool { false }
    fn min_parents(&self) -> usize { 2 }
}

/// Create a randomly initialized router weight tensor.
pub fn create_router_weights(num_experts: usize, hidden_size: usize) -> Result<Tensor, ModelError> {
    let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());
    let mut rng = rand::thread_rng();

    // Xavier initialization
    let scale = (2.0 / (hidden_size + num_experts) as f64).sqrt();
    let data: Vec<f32> = (0..num_experts * hidden_size)
        .map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * scale)
        .map(|v| v as f32)
        .collect();

    Tensor::from_vec(data, &[num_experts, hidden_size], &candle_core::Device::Cpu)
        .map_err(map_err)
}
