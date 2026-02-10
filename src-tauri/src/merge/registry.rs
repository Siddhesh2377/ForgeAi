use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::model::{ModelFormat, ModelInfo};

const MAX_PARENTS: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatInfo {
    pub architecture: Option<String>,
    pub hidden_size: Option<u64>,
    pub num_layers: Option<u64>,
    pub num_attention_heads: Option<u64>,
    pub num_kv_heads: Option<u64>,
    pub vocab_size: Option<u64>,
    pub context_length: Option<u64>,
    pub tensor_metas: Vec<TensorMeta>,
}

impl CompatInfo {
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensor_metas.iter().map(|t| t.name.clone()).collect()
    }

    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensor_metas
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.shape.as_slice())
    }

    pub fn from_model_info(info: &ModelInfo) -> Self {
        let tensor_metas: Vec<TensorMeta> = info.all_tensors.iter().map(|t| TensorMeta {
            name: t.name.clone(),
            shape: t.shape.iter().map(|&d| d as usize).collect(),
            dtype: t.dtype.clone(),
        }).collect();

        let arch = info.architecture.clone();
        let hidden_size = info.embedding_size;
        let num_layers = info.layer_count;

        let meta = &info.metadata;
        let arch_prefix = meta
            .get("general.architecture")
            .cloned()
            .unwrap_or_default();

        let num_attention_heads = meta
            .get(&format!("{}.attention.head_count", arch_prefix))
            .and_then(|v| v.parse::<u64>().ok());
        let num_kv_heads = meta
            .get(&format!("{}.attention.head_count_kv", arch_prefix))
            .and_then(|v| v.parse::<u64>().ok());
        let vocab_size = info.vocab_size.or_else(|| {
            meta.get("tokenizer.ggml.tokens_count")
                .and_then(|v| v.parse::<u64>().ok())
        });

        let context_length = info.context_length.or_else(|| {
            meta.get(&format!("{}.context_length", arch_prefix))
                .and_then(|v| v.parse::<u64>().ok())
        });

        Self {
            architecture: arch,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            vocab_size,
            context_length,
            tensor_metas,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentModel {
    pub id: String,
    pub slot: usize,
    pub name: String,
    pub file_path: String,
    pub format: ModelFormat,
    pub file_size: u64,
    pub file_size_display: String,
    pub parameter_count: u64,
    pub parameter_count_display: String,
    pub layer_count: Option<u64>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub compat: CompatInfo,
    pub color: String,
    pub is_dir: bool,
}

const GENE_COLORS: &[&str] = &[
    "#FF6B35", "#4ECDC4", "#95E1D3", "#A78BFA", "#F472B6", "#FACC15",
];

#[derive(Debug, Default, Clone)]
pub struct ParentRegistry {
    parents: Vec<ParentModel>,
}

impl ParentRegistry {
    /// Build a registry from a pre-existing vec of parents (for background tasks
    /// that need a snapshot without holding the mutex).
    pub fn from_snapshot(parents: Vec<ParentModel>) -> Self {
        Self { parents }
    }
}

impl ParentRegistry {
    pub fn add(&mut self, info: ModelInfo, slot: usize, is_dir: bool) -> Result<ParentModel, String> {
        if self.parents.len() >= MAX_PARENTS {
            return Err(format!("Maximum {} parents allowed", MAX_PARENTS));
        }
        if self.parents.iter().any(|p| p.slot == slot) {
            return Err(format!("Slot {} already occupied", slot));
        }

        let id = uuid::Uuid::new_v4().to_string();
        let color_idx = self.parents.len() % GENE_COLORS.len();
        let compat = CompatInfo::from_model_info(&info);

        let parent = ParentModel {
            id: id.clone(),
            slot,
            name: info.file_name.clone(),
            file_path: info.file_path.clone(),
            format: info.format.clone(),
            file_size: info.file_size,
            file_size_display: info.file_size_display.clone(),
            parameter_count: info.parameter_count,
            parameter_count_display: info.parameter_count_display.clone(),
            layer_count: info.layer_count,
            architecture: info.architecture.clone(),
            quantization: info.quantization.clone(),
            compat,
            color: GENE_COLORS[color_idx].to_string(),
            is_dir,
        };

        self.parents.push(parent.clone());
        Ok(parent)
    }

    pub fn remove(&mut self, parent_id: &str) -> bool {
        let len_before = self.parents.len();
        self.parents.retain(|p| p.id != parent_id);
        self.parents.len() < len_before
    }

    pub fn get(&self, parent_id: &str) -> Option<&ParentModel> {
        self.parents.iter().find(|p| p.id == parent_id)
    }

    pub fn all(&self) -> &[ParentModel] {
        &self.parents
    }

    pub fn clear(&mut self) {
        self.parents.clear();
    }

    pub fn len(&self) -> usize {
        self.parents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    pub fn shared_tensor_names(&self) -> HashSet<String> {
        if self.parents.len() < 2 {
            return HashSet::new();
        }

        let first: HashSet<String> = self.parents[0].compat.tensor_metas.iter().map(|t| t.name.clone()).collect();
        let mut shared: HashSet<String> = first;

        for parent in &self.parents[1..] {
            let names: HashSet<String> = parent.compat.tensor_metas.iter().map(|t| t.name.clone()).collect();
            shared.retain(|n| names.contains(n));
        }

        shared
    }
}
