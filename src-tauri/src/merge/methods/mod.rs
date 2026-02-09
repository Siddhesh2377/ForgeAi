pub mod average;
pub mod component_merge;
pub mod dare;
pub mod della;
pub mod frankenmerge;
pub mod moe_conversion;
pub mod parameter_slice;
pub mod passthrough;
pub mod slerp;
pub mod task_arithmetic;
pub mod tensor_surgery;
pub mod ties;

use candle_core::Tensor;

use crate::model::error::ModelError;

use super::config::{MergeMethod, MethodParams};

/// Trait for merge strategy implementations.
pub trait MergeStrategy: Send + Sync {
    /// Merge multiple tensors with weights using this strategy.
    /// `tensors` is a list of (tensor, weight) pairs.
    fn merge(
        &self,
        tensors: &[(Tensor, f64)],
        params: &MethodParams,
        base_tensor: Option<&Tensor>,
    ) -> Result<Tensor, ModelError>;

    fn name(&self) -> &'static str;
    fn requires_base(&self) -> bool;
    fn min_parents(&self) -> usize;
}

/// Dispatch to the correct merge strategy based on method.
pub fn get_strategy(method: MergeMethod) -> Box<dyn MergeStrategy> {
    match method {
        MergeMethod::Average => Box::new(average::AverageMerge),
        MergeMethod::Slerp => Box::new(slerp::SlerpMerge),
        MergeMethod::TaskArithmetic => Box::new(task_arithmetic::TaskArithmeticMerge),
        MergeMethod::Frankenmerge => Box::new(frankenmerge::FrankenmergeMerge),
        MergeMethod::Dare => Box::new(dare::DareMerge),
        MergeMethod::Ties => Box::new(ties::TiesMerge),
        MergeMethod::Della => Box::new(della::DellaMerge),
        MergeMethod::Passthrough => Box::new(passthrough::PassthroughMerge),
        MergeMethod::ComponentMerge => Box::new(component_merge::ComponentMergeMerge),
        MergeMethod::TensorSurgery => Box::new(tensor_surgery::TensorSurgeryMerge),
        MergeMethod::ParameterSlice => Box::new(parameter_slice::ParameterSliceMerge),
        MergeMethod::MoeConversion => Box::new(moe_conversion::MoeConversionMerge),
    }
}
