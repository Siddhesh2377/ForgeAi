use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("Cannot read file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),
    #[error("Invalid {format} file: {reason}")]
    ParseError { format: String, reason: String },
    #[error("File too small to be a valid model: {0} bytes")]
    FileTooSmall(u64),
    #[error("Merge error: {0}")]
    MergeError(String),
    #[error("Merge cancelled")]
    MergeCancelled,
    #[error("Incompatible models: {0}")]
    IncompatibleModels(String),
    #[error("Parent not found: {0}")]
    ParentNotFound(String),
    #[error("Registry full: max {0} parents")]
    RegistryFull(usize),
    #[error("Profiler error: {0}")]
    ProfilerError(String),
    #[error("Candle error: {0}")]
    CandleError(String),
    #[error("Tensor not found: {tensor_name} in parent {parent_id}")]
    TensorNotFound {
        tensor_name: String,
        parent_id: String,
    },
}

impl Serialize for ModelError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
