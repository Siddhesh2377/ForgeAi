use candle_core::Tensor;
use rand::Rng;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

/// DARE: Drop And REscale merging.
/// Creates task vectors, applies random dropout mask, then rescales.
pub struct DareMerge;

impl MergeStrategy for DareMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        params: &MethodParams,
        base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        let base = base_tensor.ok_or_else(|| {
            ModelError::MergeError("DARE requires a base model".into())
        })?;

        let density = params.density.unwrap_or(0.5);
        let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());

        let mut rng = rand::thread_rng();
        let total_weight: f64 = tensors.iter().map(|(_, w)| w).sum();
        let norm = if total_weight > 0.0 { total_weight } else { 1.0 };

        let mut merged_delta = Tensor::zeros_like(base).map_err(map_err)?;

        for (tensor, weight) in tensors {
            // Task vector
            let delta = (tensor - base).map_err(map_err)?;
            let numel = delta.elem_count();

            // Create random dropout mask
            let mask_data: Vec<f32> = (0..numel)
                .map(|_| if rng.gen::<f64>() < density { 1.0 / density as f32 } else { 0.0 })
                .collect();
            let mask = Tensor::from_vec(mask_data, delta.shape(), delta.device())
                .map_err(map_err)?;

            // Apply mask and rescale
            let masked_delta = (&delta * &mask).map_err(map_err)?;
            let weighted = (&masked_delta * (*weight / norm)).map_err(map_err)?;

            merged_delta = (&merged_delta + &weighted).map_err(map_err)?;
        }

        let result = (base + &merged_delta).map_err(map_err)?;
        Ok(result)
    }

    fn name(&self) -> &'static str { "DARE" }
    fn requires_base(&self) -> bool { true }
    fn min_parents(&self) -> usize { 2 }
}
