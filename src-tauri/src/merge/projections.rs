use candle_core::{DType, Device, Tensor};

use crate::model::error::ModelError;

/// Zero-pad a tensor to the target shape by creating a zeros tensor
/// and copying the original data into it.
pub fn zero_pad(tensor: &Tensor, target_shape: &[usize]) -> Result<Tensor, ModelError> {
    let src_shape = tensor.dims();

    if src_shape.len() != target_shape.len() {
        return Err(ModelError::CandleError(format!(
            "Shape rank mismatch: source has {} dims, target has {}",
            src_shape.len(),
            target_shape.len()
        )));
    }

    // If shapes already match, return clone
    if src_shape == target_shape {
        return Ok(tensor.clone());
    }

    // Verify source fits within target
    for (i, (&s, &t)) in src_shape.iter().zip(target_shape.iter()).enumerate() {
        if s > t {
            return Err(ModelError::CandleError(format!(
                "Cannot zero-pad: source dim {} ({}) > target ({})",
                i, s, t
            )));
        }
    }

    // Create a zero tensor of target shape
    let dtype = tensor.dtype();
    let mut result = Tensor::zeros(target_shape, dtype, tensor.device())
        .map_err(|e| ModelError::CandleError(e.to_string()))?;

    // For each dimension, narrow the result to the source size and add the source tensor
    // We build a slice expression to cover the source region within the target
    let mut sliced = result.clone();
    for (dim, &size) in src_shape.iter().enumerate() {
        sliced = sliced
            .narrow(dim, 0, size)
            .map_err(|e| ModelError::CandleError(e.to_string()))?;
    }

    // We can't directly assign into a slice with candle, so we use a different approach:
    // Create the padded tensor by concatenating along each dimension
    result = pad_recursive(tensor, target_shape, 0)?;

    Ok(result)
}

/// Recursively pad a tensor along each dimension.
fn pad_recursive(tensor: &Tensor, target_shape: &[usize], dim: usize) -> Result<Tensor, ModelError> {
    if dim >= tensor.dims().len() {
        return Ok(tensor.clone());
    }

    let src_size = tensor.dims()[dim];
    let tgt_size = target_shape[dim];

    let current = if src_size < tgt_size {
        // Create padding tensor for this dimension
        let mut pad_shape: Vec<usize> = tensor.dims().to_vec();
        pad_shape[dim] = tgt_size - src_size;

        let padding = Tensor::zeros(pad_shape.as_slice(), tensor.dtype(), tensor.device())
            .map_err(|e| ModelError::CandleError(e.to_string()))?;

        Tensor::cat(&[tensor, &padding], dim)
            .map_err(|e| ModelError::CandleError(e.to_string()))?
    } else {
        tensor.clone()
    };

    // Recurse for next dimension
    pad_recursive(&current, target_shape, dim + 1)
}

/// Truncate a tensor to the target shape by narrowing each dimension.
pub fn truncate(tensor: &Tensor, target_shape: &[usize]) -> Result<Tensor, ModelError> {
    let src_shape = tensor.dims();

    if src_shape.len() != target_shape.len() {
        return Err(ModelError::CandleError(format!(
            "Shape rank mismatch: source has {} dims, target has {}",
            src_shape.len(),
            target_shape.len()
        )));
    }

    if src_shape == target_shape {
        return Ok(tensor.clone());
    }

    let mut result = tensor.clone();
    for (dim, (&s, &t)) in src_shape.iter().zip(target_shape.iter()).enumerate() {
        if t < s {
            result = result
                .narrow(dim, 0, t)
                .map_err(|e| ModelError::CandleError(e.to_string()))?;
        }
    }

    Ok(result)
}

/// Interpolate (nearest-neighbor) a tensor to the target shape.
/// For 2D matrices this does simple index mapping; for 1D vectors it does linear interpolation.
pub fn interpolate(tensor: &Tensor, target_shape: &[usize]) -> Result<Tensor, ModelError> {
    let src_shape = tensor.dims();

    if src_shape.len() != target_shape.len() {
        return Err(ModelError::CandleError(format!(
            "Shape rank mismatch: source has {} dims, target has {}",
            src_shape.len(),
            target_shape.len()
        )));
    }

    if src_shape == target_shape {
        return Ok(tensor.clone());
    }

    // Convert to f32 for interpolation
    let tensor_f32 = tensor
        .to_dtype(DType::F32)
        .map_err(|e| ModelError::CandleError(e.to_string()))?;

    match src_shape.len() {
        1 => interpolate_1d(&tensor_f32, target_shape[0]),
        2 => interpolate_2d(&tensor_f32, target_shape),
        _ => {
            // For higher-dimensional tensors, fall back to zero-padding or truncation
            if target_shape.iter().all(|&t| t > 0) {
                let needs_pad = src_shape.iter().zip(target_shape.iter()).any(|(&s, &t)| t > s);
                let needs_trunc = src_shape.iter().zip(target_shape.iter()).any(|(&s, &t)| t < s);
                if needs_pad && !needs_trunc {
                    zero_pad(&tensor_f32, target_shape)
                } else if needs_trunc && !needs_pad {
                    truncate(&tensor_f32, target_shape)
                } else {
                    // Mixed: truncate first, then pad
                    let trunc_shape: Vec<usize> = src_shape
                        .iter()
                        .zip(target_shape.iter())
                        .map(|(&s, &t)| s.min(t))
                        .collect();
                    let truncated = truncate(&tensor_f32, &trunc_shape)?;
                    zero_pad(&truncated, target_shape)
                }
            } else {
                Err(ModelError::CandleError("Invalid target shape".into()))
            }
        }
    }
}

/// 1D linear interpolation.
fn interpolate_1d(tensor: &Tensor, target_len: usize) -> Result<Tensor, ModelError> {
    let src_data = tensor
        .flatten_all()
        .map_err(|e| ModelError::CandleError(e.to_string()))?
        .to_vec1::<f32>()
        .map_err(|e| ModelError::CandleError(e.to_string()))?;

    let src_len = src_data.len();
    if src_len == 0 || target_len == 0 {
        return Tensor::zeros(&[target_len], DType::F32, &Device::Cpu)
            .map_err(|e| ModelError::CandleError(e.to_string()));
    }

    let mut result = vec![0.0f32; target_len];
    let scale = if target_len > 1 {
        (src_len - 1) as f64 / (target_len - 1) as f64
    } else {
        0.0
    };

    for i in 0..target_len {
        let src_pos = i as f64 * scale;
        let lo = src_pos.floor() as usize;
        let hi = (lo + 1).min(src_len - 1);
        let frac = (src_pos - lo as f64) as f32;
        result[i] = src_data[lo] * (1.0 - frac) + src_data[hi] * frac;
    }

    Tensor::from_vec(result, &[target_len], &Device::Cpu)
        .map_err(|e| ModelError::CandleError(e.to_string()))
}

/// 2D nearest-neighbor interpolation.
fn interpolate_2d(tensor: &Tensor, target_shape: &[usize]) -> Result<Tensor, ModelError> {
    let src_shape = tensor.dims();
    let (src_rows, src_cols) = (src_shape[0], src_shape[1]);
    let (tgt_rows, tgt_cols) = (target_shape[0], target_shape[1]);

    let data = tensor
        .flatten_all()
        .map_err(|e| ModelError::CandleError(e.to_string()))?
        .to_vec1::<f32>()
        .map_err(|e| ModelError::CandleError(e.to_string()))?;

    let mut result = vec![0.0f32; tgt_rows * tgt_cols];

    for r in 0..tgt_rows {
        let src_r = ((r as f64 / tgt_rows.max(1) as f64) * src_rows as f64).floor() as usize;
        let src_r = src_r.min(src_rows - 1);

        for c in 0..tgt_cols {
            let src_c = ((c as f64 / tgt_cols.max(1) as f64) * src_cols as f64).floor() as usize;
            let src_c = src_c.min(src_cols - 1);

            result[r * tgt_cols + c] = data[src_r * src_cols + src_c];
        }
    }

    Tensor::from_vec(result, target_shape, &Device::Cpu)
        .map_err(|e| ModelError::CandleError(e.to_string()))
}

/// Adapt a tensor to a target shape using the specified strategy.
pub fn adapt_tensor(
    tensor: &Tensor,
    target_shape: &[usize],
    strategy: &str,
) -> Result<Tensor, ModelError> {
    match strategy {
        "zero_padding" => zero_pad(tensor, target_shape),
        "truncation" => truncate(tensor, target_shape),
        "interpolation" => interpolate(tensor, target_shape),
        _ => Err(ModelError::CandleError(format!(
            "Unknown projection strategy: {}",
            strategy
        ))),
    }
}
