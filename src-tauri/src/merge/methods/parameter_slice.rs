use candle_core::Tensor;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

/// Parameter Slice: dimensional slicing across parents.
/// Splits a tensor dimension across parents, taking a slice from each.
pub struct ParameterSliceMerge;

impl MergeStrategy for ParameterSliceMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        params: &MethodParams,
        _base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        if tensors.len() < 2 {
            return Err(ModelError::MergeError("Parameter Slice requires at least 2 tensors".into()));
        }

        let slice_dim = params.slice_dim.unwrap_or(0);
        let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());

        let shape = tensors[0].0.shape();
        let dims = shape.dims();

        if slice_dim >= dims.len() {
            return Err(ModelError::MergeError(format!(
                "Slice dimension {} exceeds tensor rank {}",
                slice_dim,
                dims.len()
            )));
        }

        let dim_size = dims[slice_dim];
        let num_parents = tensors.len();

        // If explicit ranges provided, use them
        if let Some(ranges) = &params.slice_ranges {
            if ranges.len() == num_parents {
                let slices: Result<Vec<Tensor>, _> = ranges
                    .iter()
                    .zip(tensors.iter())
                    .map(|((start, end), (tensor, _))| {
                        tensor.narrow(slice_dim, *start, end - start).map_err(map_err)
                    })
                    .collect();
                return Tensor::cat(&slices?, slice_dim).map_err(map_err);
            }
        }

        // Default: split evenly weighted by parent weights
        let total_weight: f64 = tensors.iter().map(|(_, w)| w).sum();
        let mut slices = Vec::new();
        let mut offset = 0;

        for (i, (tensor, weight)) in tensors.iter().enumerate() {
            let proportion = if total_weight > 0.0 {
                weight / total_weight
            } else {
                1.0 / num_parents as f64
            };

            let slice_len = if i == num_parents - 1 {
                dim_size - offset
            } else {
                ((dim_size as f64 * proportion).round() as usize).max(1)
            };

            if offset + slice_len > dim_size {
                break;
            }

            let slice = tensor.narrow(slice_dim, offset, slice_len).map_err(map_err)?;
            slices.push(slice);
            offset += slice_len;
        }

        if slices.is_empty() {
            return Ok(tensors[0].0.clone());
        }

        Tensor::cat(&slices, slice_dim).map_err(map_err)
    }

    fn name(&self) -> &'static str { "Parameter Slice" }
    fn requires_base(&self) -> bool { false }
    fn min_parents(&self) -> usize { 2 }
}
