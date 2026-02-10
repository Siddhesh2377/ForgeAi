use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeMethod {
    Average,
    Slerp,
    TaskArithmetic,
    Frankenmerge,
    Dare,
    Ties,
    Della,
    Passthrough,
    ComponentMerge,
    TensorSurgery,
    ParameterSlice,
    MoeConversion,
}

impl MergeMethod {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Average => "AVERAGE",
            Self::Slerp => "SLERP",
            Self::TaskArithmetic => "TASK ARITHMETIC",
            Self::Frankenmerge => "FRANKENMERGE",
            Self::Dare => "DARE",
            Self::Ties => "TIES",
            Self::Della => "DELLA",
            Self::Passthrough => "PASSTHROUGH",
            Self::ComponentMerge => "COMPONENT MERGE",
            Self::TensorSurgery => "TENSOR SURGERY",
            Self::ParameterSlice => "PARAMETER SLICE",
            Self::MoeConversion => "MOE CONVERSION",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Average => "Simple weighted average of tensors",
            Self::Slerp => "Spherical linear interpolation between two models",
            Self::TaskArithmetic => "Add task vectors to a base model",
            Self::Frankenmerge => "Stack layers from different models",
            Self::Dare => "Drop and rescale: random dropout with rescaling",
            Self::Ties => "Trim, elect sign, merge: task-specific merging",
            Self::Della => "Density-based DARE with lambda interpolation",
            Self::Passthrough => "Direct copy of tensors from a single parent",
            Self::ComponentMerge => "Route attention/MLP/norm to different parents",
            Self::TensorSurgery => "Per-tensor source mapping from parents",
            Self::ParameterSlice => "Dimensional slicing across parents",
            Self::MoeConversion => "Convert dense models to Mixture-of-Experts",
        }
    }

    pub fn requires_base(&self) -> bool {
        matches!(self, Self::TaskArithmetic | Self::Dare | Self::Ties | Self::Della)
    }

    pub fn min_parents(&self) -> usize {
        match self {
            Self::Passthrough => 1,
            _ => 2,
        }
    }

    pub fn difficulty(&self) -> &'static str {
        match self {
            Self::Average | Self::Slerp | Self::Passthrough => "easy",
            Self::TaskArithmetic | Self::Frankenmerge | Self::Dare | Self::Ties => "intermediate",
            Self::Della | Self::ComponentMerge | Self::TensorSurgery | Self::ParameterSlice | Self::MoeConversion => "advanced",
        }
    }

    pub fn all() -> &'static [MergeMethod] {
        &[
            Self::Average,
            Self::Slerp,
            Self::TaskArithmetic,
            Self::Frankenmerge,
            Self::Dare,
            Self::Ties,
            Self::Della,
            Self::Passthrough,
            Self::ComponentMerge,
            Self::TensorSurgery,
            Self::ParameterSlice,
            Self::MoeConversion,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    SafeTensors,
    Gguf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: OutputFormat,
    pub path: String,
    pub model_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentWeight {
    pub parent_id: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAssignment {
    pub layer_index: u64,
    pub source_parent_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentType {
    Attention,
    Mlp,
    Norm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentOverride {
    pub component: ComponentType,
    pub parent_id: String,
    pub layer_start: u64,
    pub layer_end: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOverride {
    pub tensor_name: String,
    pub parent_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodParams {
    // SLERP
    pub t: Option<f64>,
    // Task Arithmetic
    pub scaling: Option<f64>,
    // DARE
    pub density: Option<f64>,
    // TIES
    pub majority_sign_method: Option<String>,
    pub trim_threshold: Option<f64>,
    // DELLA
    pub lambda: Option<f64>,
    pub della_density: Option<f64>,
    // MoE
    pub num_experts: Option<usize>,
    pub experts_per_token: Option<usize>,
    // Parameter Slice
    pub slice_dim: Option<usize>,
    pub slice_ranges: Option<Vec<(usize, usize)>>,
}

impl Default for MethodParams {
    fn default() -> Self {
        Self {
            t: None,
            scaling: None,
            density: None,
            majority_sign_method: None,
            trim_threshold: None,
            lambda: None,
            della_density: None,
            num_experts: None,
            experts_per_token: None,
            slice_dim: None,
            slice_ranges: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    pub parents: Vec<ParentWeight>,
    pub method: MergeMethod,
    pub params: MethodParams,
    pub base_parent_id: Option<String>,
    pub layer_assignments: Vec<LayerAssignment>,
    pub component_overrides: Vec<ComponentOverride>,
    pub tensor_overrides: Vec<TensorOverride>,
    pub output: OutputConfig,
    #[serde(default)]
    pub memory_limit_mb: Option<u64>,
    #[serde(default)]
    pub projection_strategy: Option<String>,
    #[serde(default)]
    pub skip_layers: Vec<u64>,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

fn default_batch_size() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeMethodInfo {
    pub id: MergeMethod,
    pub name: String,
    pub description: String,
    pub requires_base: bool,
    pub min_parents: usize,
    pub difficulty: String,
}

impl From<MergeMethod> for MergeMethodInfo {
    fn from(m: MergeMethod) -> Self {
        Self {
            id: m,
            name: m.display_name().to_string(),
            description: m.description().to_string(),
            requires_base: m.requires_base(),
            min_parents: m.min_parents(),
            difficulty: m.difficulty().to_string(),
        }
    }
}
