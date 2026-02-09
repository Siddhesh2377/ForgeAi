use candle_core::Tensor;

use crate::merge::config::MethodParams;
use crate::model::error::ModelError;

use super::MergeStrategy;

pub struct SlerpMerge;

impl MergeStrategy for SlerpMerge {
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        params: &MethodParams,
        _base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        if tensors.len() < 2 {
            return Err(ModelError::MergeError("SLERP requires exactly 2 tensors".into()));
        }

        let t = params.t.unwrap_or(0.5);
        let a = &tensors[0].0;
        let b = &tensors[1].0;

        slerp_tensors(a, b, t)
    }

    fn name(&self) -> &'static str { "SLERP" }
    fn requires_base(&self) -> bool { false }
    fn min_parents(&self) -> usize { 2 }
}

fn slerp_tensors(a: &Tensor, b: &Tensor, t: f64) -> Result<Tensor, ModelError> {
    let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());

    // Flatten for dot product
    let a_flat = a.flatten_all().map_err(map_err)?;
    let b_flat = b.flatten_all().map_err(map_err)?;

    // Compute norms
    let a_norm = a_flat.sqr().map_err(map_err)?
        .sum_all().map_err(map_err)?
        .sqrt().map_err(map_err)?
        .to_scalar::<f32>().map_err(map_err)? as f64;
    let b_norm = b_flat.sqr().map_err(map_err)?
        .sum_all().map_err(map_err)?
        .sqrt().map_err(map_err)?
        .to_scalar::<f32>().map_err(map_err)? as f64;

    if a_norm < 1e-10 || b_norm < 1e-10 {
        // Degenerate case: fall back to linear interpolation
        let result = (&(a * (1.0 - t)).map_err(map_err)? + &(b * t).map_err(map_err)?)
            .map_err(map_err)?;
        return Ok(result);
    }

    // Normalize
    let a_unit = (&a_flat / a_norm).map_err(map_err)?;
    let b_unit = (&b_flat / b_norm).map_err(map_err)?;

    // Cosine of angle
    let dot = (&a_unit * &b_unit).map_err(map_err)?
        .sum_all().map_err(map_err)?
        .to_scalar::<f32>().map_err(map_err)? as f64;

    // Clamp to avoid numerical issues
    let dot = dot.clamp(-1.0, 1.0);

    if dot.abs() > 0.9995 {
        // Very close: use linear interpolation
        let result = (&(a * (1.0 - t)).map_err(map_err)? + &(b * t).map_err(map_err)?)
            .map_err(map_err)?;
        return Ok(result);
    }

    let omega = dot.acos();
    let sin_omega = omega.sin();

    let scale_a = ((1.0 - t) * omega).sin() / sin_omega;
    let scale_b = (t * omega).sin() / sin_omega;

    // Interpolate in original scale
    let result = (&(a * scale_a).map_err(map_err)? + &(b * scale_b).map_err(map_err)?)
        .map_err(map_err)?;

    Ok(result)
}
