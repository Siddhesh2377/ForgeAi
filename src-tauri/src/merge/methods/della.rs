use candle_core::Tensor;
use rand::Rng;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

/// DELLA: Density-based DARE with lambda interpolation.
/// Combines DARE-style dropout with density-aware rescaling.
pub struct DellaMerge;

impl MergeStrategy for DellaMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        params: &MethodParams,
        base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        let base = base_tensor.ok_or_else(|| {
            ModelError::MergeError("DELLA requires a base model".into())
        })?;

        let density = params.della_density.unwrap_or(0.7);
        let lambda = params.lambda.unwrap_or(1.0);
        let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());

        let mut rng = rand::thread_rng();
        let numel = base.elem_count();
        let total_weight: f64 = tensors.iter().map(|(_, w)| w).sum();
        let norm = if total_weight > 0.0 { total_weight } else { 1.0 };

        let mut merged_delta = Tensor::zeros_like(base).map_err(map_err)?;

        for (tensor, weight) in tensors {
            let delta = (tensor - base).map_err(map_err)?;
            let abs_delta = delta.abs().map_err(map_err)?;
            let flat_abs: Vec<f32> = abs_delta.flatten_all().map_err(map_err)?
                .to_vec1::<f32>().map_err(map_err)?;

            // Compute magnitude-based density scores
            let max_mag = flat_abs.iter().cloned().fold(0.0f32, f32::max);
            let density_scores: Vec<f32> = if max_mag > 1e-10 {
                flat_abs.iter().map(|&v| v / max_mag).collect()
            } else {
                vec![0.5; numel]
            };

            // Create density-aware dropout mask
            let mask_data: Vec<f32> = density_scores
                .iter()
                .map(|&score| {
                    let keep_prob = density * (1.0 + score as f64 * lambda) / (1.0 + lambda);
                    let keep_prob = keep_prob.clamp(0.01, 1.0);
                    if rng.gen::<f64>() < keep_prob {
                        (1.0 / keep_prob) as f32
                    } else {
                        0.0
                    }
                })
                .collect();

            let mask = Tensor::from_vec(mask_data, delta.shape(), delta.device())
                .map_err(map_err)?;

            let masked = (&delta * &mask).map_err(map_err)?;
            let weighted = (&masked * (*weight / norm)).map_err(map_err)?;

            merged_delta = (&merged_delta + &weighted).map_err(map_err)?;
        }

        let result = (base + &merged_delta).map_err(map_err)?;
        Ok(result)
    }

    fn name(&self) -> &'static str { "DELLA" }
    fn requires_base(&self) -> bool { true }
    fn min_parents(&self) -> usize { 2 }
}
