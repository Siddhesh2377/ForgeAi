use serde::{Deserialize, Serialize};

// ── Training Method ─────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMethod {
    Sft,
    Lora,
    Qlora,
    Dpo,
    FullFinetune,
}

impl std::fmt::Display for TrainingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sft => write!(f, "sft"),
            Self::Lora => write!(f, "lora"),
            Self::Qlora => write!(f, "qlora"),
            Self::Dpo => write!(f, "dpo"),
            Self::FullFinetune => write!(f, "full_finetune"),
        }
    }
}

// ── Dataset Format ──────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetFormat {
    Json,
    Jsonl,
    Csv,
    Parquet,
}

impl std::fmt::Display for DatasetFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::Jsonl => write!(f, "jsonl"),
            Self::Csv => write!(f, "csv"),
            Self::Parquet => write!(f, "parquet"),
        }
    }
}

// ── Training Config ─────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model_path: String,
    pub dataset_path: String,
    pub dataset_format: DatasetFormat,
    pub method: TrainingMethod,
    pub output_path: String,
    #[serde(default)]
    pub merge_adapter: bool,

    // Hyperparameters
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default = "default_batch")]
    pub batch_size: u32,
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: u32,
    #[serde(default = "default_seq_len")]
    pub max_seq_length: u32,
    #[serde(default = "default_warmup")]
    pub warmup_steps: u32,
    #[serde(default = "default_wd")]
    pub weight_decay: f64,
    #[serde(default = "default_save_steps")]
    pub save_steps: u32,

    // LoRA
    #[serde(default)]
    pub lora_rank: Option<u32>,
    #[serde(default)]
    pub lora_alpha: Option<u32>,
    #[serde(default)]
    pub lora_dropout: Option<f64>,
    #[serde(default)]
    pub target_modules: Option<Vec<String>>,
    #[serde(default)]
    pub layers_to_transform: Option<Vec<u32>>,

    // QLoRA
    #[serde(default)]
    pub quantization_bits: Option<u32>,

    // DPO
    #[serde(default)]
    pub dpo_beta: Option<f64>,

    // GPU
    #[serde(default)]
    pub gpu_memory_limit_gb: Option<f64>,
}

fn default_lr() -> f64 { 2e-4 }
fn default_epochs() -> u32 { 3 }
fn default_batch() -> u32 { 1 }
fn default_grad_accum() -> u32 { 8 }
fn default_seq_len() -> u32 { 512 }
fn default_warmup() -> u32 { 100 }
fn default_wd() -> f64 { 0.01 }
fn default_save_steps() -> u32 { 500 }

// ── Progress / Result ───────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub stage: String,
    pub message: String,
    pub percent: f64,
    #[serde(default)]
    pub epoch: Option<u32>,
    #[serde(default)]
    pub step: Option<u64>,
    #[serde(default)]
    pub total_steps: Option<u64>,
    #[serde(default)]
    pub loss: Option<f64>,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub eta_seconds: Option<u64>,
    #[serde(default)]
    pub gpu_memory_used_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub output_path: String,
    pub output_size: u64,
    pub output_size_display: String,
    pub method: String,
    pub epochs_completed: u32,
    pub final_loss: Option<f64>,
    pub adapter_merged: bool,
}

// ── Dataset Info ────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub path: String,
    pub format: DatasetFormat,
    pub rows: u64,
    pub columns: Vec<String>,
    pub preview: Vec<serde_json::Value>,
    pub size: u64,
    pub size_display: String,
    pub detected_template: Option<String>,
}

// ── Deps Status ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDepsStatus {
    pub python_found: bool,
    pub python_version: Option<String>,
    pub venv_ready: bool,
    pub packages_ready: bool,
    pub missing_packages: Vec<String>,
    pub cuda_available: bool,
    pub cuda_version: Option<String>,
    pub torch_version: Option<String>,
    pub ready: bool,
}

// ── Surgery Config ──────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgeryConfig {
    pub model_path: String,
    pub output_path: String,
    pub operations: Vec<SurgeryOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurgeryOperation {
    RemoveLayer { index: u64 },
    DuplicateLayer { source_index: u64, insert_at: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgeryResult {
    pub output_path: String,
    pub output_size: u64,
    pub output_size_display: String,
    pub original_layers: u64,
    pub final_layers: u64,
    pub tensors_written: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgeryProgress {
    pub stage: String,
    pub message: String,
    pub percent: f64,
}

// ── Target Module Group ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetModuleGroup {
    pub name: String,
    pub modules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCapabilityMapping {
    pub capability: String,
    pub name: String,
    pub layers: Vec<u64>,
}

// ── Layer Detail (Surgery View) ─────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLayerDetail {
    pub index: u64,
    pub total_bytes: u64,
    pub display: String,
    pub attention_count: u32,
    pub mlp_count: u32,
    pub norm_count: u32,
    pub other_count: u32,
    pub attention_bytes: u64,
    pub mlp_bytes: u64,
    pub norm_bytes: u64,
    pub other_bytes: u64,
    pub capabilities: Vec<String>,
    pub tensors: Vec<LayerTensorInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub memory_display: String,
    pub component: String,
}

// ── Full Dataset Info (DataStudio) ──────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetFullInfo {
    pub path: String,
    pub format: DatasetFormat,
    pub rows: u64,
    pub columns: Vec<String>,
    pub column_analysis: Vec<ColumnAnalysis>,
    pub preview: Vec<serde_json::Value>,
    pub size: u64,
    pub size_display: String,
    pub detected_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnAnalysis {
    pub name: String,
    pub dtype: String,
    pub non_null_count: u64,
    pub null_count: u64,
    pub sample_values: Vec<String>,
    pub avg_length: Option<f64>,
}
