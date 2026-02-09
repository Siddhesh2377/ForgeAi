use candle_core::Tensor;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

/// TIES-Merging: Trim, Elect Sign, Merge.
pub struct TiesMerge;

impl MergeStrategy for TiesMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        params: &MethodParams,
        base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        let base = base_tensor.ok_or_else(|| {
            ModelError::MergeError("TIES requires a base model".into())
        })?;

        let trim_threshold = params.trim_threshold.unwrap_or(0.2);
        let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());

        // Step 1: Compute task vectors and trim small values
        let mut trimmed_deltas: Vec<(Tensor, f64)> = Vec::new();

        for (tensor, weight) in tensors {
            let delta = (tensor - base).map_err(map_err)?;

            // Trim: set parameters with magnitude below threshold to zero
            let abs_delta = delta.abs().map_err(map_err)?;
            let flat = abs_delta.flatten_all().map_err(map_err)?;

            // Compute threshold as a percentile of magnitudes
            let numel = flat.elem_count();
            let threshold_idx = (numel as f64 * trim_threshold) as usize;

            let mut magnitudes: Vec<f32> = flat
                .to_vec1::<f32>()
                .map_err(map_err)?;
            magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let threshold_val = if threshold_idx < magnitudes.len() {
                magnitudes[threshold_idx]
            } else {
                0.0
            };

            // Create trim mask
            let mask_data: Vec<f32> = abs_delta
                .flatten_all().map_err(map_err)?
                .to_vec1::<f32>().map_err(map_err)?
                .iter()
                .map(|&v| if v >= threshold_val { 1.0 } else { 0.0 })
                .collect();
            let mask = Tensor::from_vec(mask_data, delta.shape(), delta.device())
                .map_err(map_err)?;

            let trimmed = (&delta * &mask).map_err(map_err)?;
            trimmed_deltas.push((trimmed, *weight));
        }

        // Step 2: Elect majority sign
        // For each parameter position, determine the sign that has the most weight
        let numel = base.elem_count();
        let mut sign_votes = vec![0.0f64; numel];

        for (delta, weight) in &trimmed_deltas {
            let flat: Vec<f32> = delta.flatten_all().map_err(map_err)?
                .to_vec1::<f32>().map_err(map_err)?;
            for (i, &val) in flat.iter().enumerate() {
                if val > 0.0 {
                    sign_votes[i] += weight;
                } else if val < 0.0 {
                    sign_votes[i] -= weight;
                }
            }
        }

        // Step 3: Merge — only keep values that agree with majority sign
        let mut merged_data = vec![0.0f32; numel];

        for (delta, weight) in &trimmed_deltas {
            let flat: Vec<f32> = delta.flatten_all().map_err(map_err)?
                .to_vec1::<f32>().map_err(map_err)?;
            for (i, &val) in flat.iter().enumerate() {
                let agrees = (val > 0.0 && sign_votes[i] > 0.0)
                    || (val < 0.0 && sign_votes[i] < 0.0);
                if agrees {
                    merged_data[i] += val * (*weight as f32);
                }
            }
        }

        // Normalize by total weight of agreeing models per position
        let total_weight: f64 = tensors.iter().map(|(_, w)| w).sum();
        if total_weight > 0.0 {
            for val in &mut merged_data {
                *val /= total_weight as f32;
            }
        }

        let merged_delta = Tensor::from_vec(merged_data, base.shape(), base.device())
            .map_err(map_err)?;

        let result = (base + &merged_delta).map_err(map_err)?;
        Ok(result)
    }

    fn name(&self) -> &'static str { "TIES" }
    fn requires_base(&self) -> bool { true }
    fn min_parents(&self) -> usize { 2 }
}
