pub mod error;
pub mod gguf;
pub mod inspect;
pub mod safetensors;
pub mod state;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    SafeTensors,
    Gguf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub file_name: String,
    pub file_path: String,
    pub file_size: u64,
    pub file_size_display: String,
    pub format: ModelFormat,
    pub tensor_count: u64,
    pub parameter_count: u64,
    pub parameter_count_display: String,
    pub layer_count: Option<u64>,
    pub quantization: Option<String>,
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    pub embedding_size: Option<u64>,
    pub metadata: HashMap<String, String>,
    pub tensor_preview: Vec<TensorInfo>,
    #[serde(skip)]
    pub all_tensors: Vec<TensorInfo>,
    // Folder-loaded SafeTensors fields
    pub shard_count: Option<u32>,
    pub has_tokenizer: Option<bool>,
    pub has_config: Option<bool>,
    pub model_type: Option<String>,
    pub vocab_size: Option<u64>,
}

pub fn format_file_size(bytes: u64) -> String {
    const GB: f64 = 1_073_741_824.0;
    const MB: f64 = 1_048_576.0;
    const KB: f64 = 1_024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.2} KB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

pub fn format_param_count(count: u64) -> String {
    const B: f64 = 1_000_000_000.0;
    const M: f64 = 1_000_000.0;
    const K: f64 = 1_000.0;
    let c = count as f64;
    if c >= B {
        format!("{:.2}B", c / B)
    } else if c >= M {
        format!("{:.2}M", c / M)
    } else if c >= K {
        format!("{:.1}K", c / K)
    } else {
        format!("{}", count)
    }
}

pub fn derive_layer_count(tensors: &[TensorInfo]) -> Option<u64> {
    let patterns = ["layers.", "blocks.", "blk.", "h."];
    let mut layer_indices = HashSet::new();

    for t in tensors {
        for pat in &patterns {
            if let Some(pos) = t.name.find(pat) {
                let after = &t.name[pos + pat.len()..];
                if let Some(end) = after.find('.') {
                    if let Ok(idx) = after[..end].parse::<u64>() {
                        layer_indices.insert(idx);
                    }
                }
            }
        }
    }

    if layer_indices.is_empty() {
        None
    } else {
        Some(layer_indices.len() as u64)
    }
}
