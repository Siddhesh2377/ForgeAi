use std::path::PathBuf;
use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};

use crate::merge::compatibility;
use crate::merge::config::{MergeConfig, MergeMethod, MergeMethodInfo};
use crate::merge::executor::{self, MergeResult};
use crate::merge::planner;
use crate::merge::profiler;
use crate::merge::registry::ParentModel;
use crate::model::error::ModelError;
use crate::model::inspect;
use crate::model::state::AppState;
use crate::model::ModelFormat;

// ── Serializable types for frontend ──────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentModelInfo {
    pub id: String,
    pub slot: usize,
    pub name: String,
    pub file_path: String,
    pub format: String,
    pub file_size: u64,
    pub file_size_display: String,
    pub parameter_count: u64,
    pub parameter_count_display: String,
    pub layer_count: Option<u64>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub color: String,
    pub tensor_count: usize,
}

impl From<&ParentModel> for ParentModelInfo {
    fn from(p: &ParentModel) -> Self {
        Self {
            id: p.id.clone(),
            slot: p.slot,
            name: p.name.clone(),
            file_path: p.file_path.clone(),
            format: match &p.format {
                ModelFormat::SafeTensors => "safe_tensors".to_string(),
                ModelFormat::Gguf => "gguf".to_string(),
            },
            file_size: p.file_size,
            file_size_display: p.file_size_display.clone(),
            parameter_count: p.parameter_count,
            parameter_count_display: p.parameter_count_display.clone(),
            layer_count: p.layer_count,
            architecture: p.architecture.clone(),
            quantization: p.quantization.clone(),
            color: p.color.clone(),
            tensor_count: p.compat.tensor_names.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerComponentInfo {
    pub layer_index: u64,
    pub attention_tensors: Vec<String>,
    pub mlp_tensors: Vec<String>,
    pub norm_tensors: Vec<String>,
    pub other_tensors: Vec<String>,
    pub total_params: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorComparisonEntry {
    pub tensor_name: String,
    pub present_in: Vec<String>,
    pub shapes_match: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorComparison {
    pub entries: Vec<TensorComparisonEntry>,
    pub shared_count: usize,
    pub unique_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

// ── Parent Management Commands ───────────────────────────

#[tauri::command]
pub fn merge_load_parent(
    path: String,
    slot: usize,
    state: State<'_, AppState>,
) -> Result<ParentModelInfo, ModelError> {
    let path = PathBuf::from(&path);

    if !path.exists() {
        return Err(ModelError::FileNotFound(path.to_string_lossy().to_string()));
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let info = match ext.as_deref() {
        Some("safetensors") => crate::model::safetensors::parse(&path)?,
        Some("gguf") => crate::model::gguf::parse(&path)?,
        _ => return Err(ModelError::UnsupportedFormat(
            ext.unwrap_or_else(|| "unknown".to_string()),
        )),
    };

    let mut registry = state.merge_parents.lock().unwrap();
    let parent = registry.add(info, slot, false)
        .map_err(|e| ModelError::MergeError(e))?;

    Ok(ParentModelInfo::from(&parent))
}

#[tauri::command]
pub fn merge_load_parent_dir(
    path: String,
    slot: usize,
    state: State<'_, AppState>,
) -> Result<ParentModelInfo, ModelError> {
    let path = PathBuf::from(&path);

    if !path.exists() || !path.is_dir() {
        return Err(ModelError::FileNotFound(path.to_string_lossy().to_string()));
    }

    let info = crate::model::safetensors::parse_dir(&path)?;

    let mut registry = state.merge_parents.lock().unwrap();
    let parent = registry.add(info, slot, true)
        .map_err(|e| ModelError::MergeError(e))?;

    Ok(ParentModelInfo::from(&parent))
}

#[tauri::command]
pub fn merge_remove_parent(
    parent_id: String,
    state: State<'_, AppState>,
) -> Result<(), ModelError> {
    let mut registry = state.merge_parents.lock().unwrap();
    if !registry.remove(&parent_id) {
        return Err(ModelError::ParentNotFound(parent_id));
    }
    Ok(())
}

#[tauri::command]
pub fn merge_get_parents(state: State<'_, AppState>) -> Vec<ParentModelInfo> {
    let registry = state.merge_parents.lock().unwrap();
    registry.all().iter().map(ParentModelInfo::from).collect()
}

#[tauri::command]
pub fn merge_clear_parents(state: State<'_, AppState>) {
    let mut registry = state.merge_parents.lock().unwrap();
    registry.clear();
}

// ── Validation Commands ──────────────────────────────────

#[tauri::command]
pub fn merge_check_compatibility(
    state: State<'_, AppState>,
) -> crate::merge::compatibility::CompatReport {
    let registry = state.merge_parents.lock().unwrap();
    compatibility::check_compatibility(&registry)
}

#[tauri::command]
pub fn merge_validate_config(
    config: MergeConfig,
    state: State<'_, AppState>,
) -> ValidationResult {
    let registry = state.merge_parents.lock().unwrap();
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Check minimum parents
    let min = config.method.min_parents();
    if registry.len() < min {
        errors.push(format!("{} requires at least {} parents", config.method.display_name(), min));
    }

    // Check base model for methods that require it
    if config.method.requires_base() && config.base_parent_id.is_none() {
        errors.push(format!("{} requires a base model", config.method.display_name()));
    }

    // Validate parent weights sum
    let total_weight: f64 = config.parents.iter().map(|p| p.weight).sum();
    if (total_weight - 1.0).abs() > 0.01 && total_weight > 0.0 {
        warnings.push(format!("Weights sum to {:.2}, will be normalized to 1.0", total_weight));
    }

    // Validate all parent_ids exist
    for pw in &config.parents {
        if registry.get(&pw.parent_id).is_none() {
            errors.push(format!("Parent '{}' not found", pw.parent_id));
        }
    }

    // Check output path
    if config.output.path.is_empty() {
        errors.push("Output path is required".to_string());
    }

    // Frankenmerge / Passthrough require layer assignments
    if matches!(config.method, MergeMethod::Frankenmerge) && config.layer_assignments.is_empty() {
        errors.push("FRANKENMERGE requires layer assignments. Assign each offspring layer to a parent in the Layers tab.".to_string());
    }
    if matches!(config.method, MergeMethod::Passthrough) && config.layer_assignments.is_empty() {
        warnings.push("PASSTHROUGH with no layer assignments will copy all layers from the first parent.".to_string());
    }

    // Warn about naive averaging
    if matches!(config.method, MergeMethod::Average) && registry.len() >= 2 {
        warnings.push("AVERAGE merges all tensors by weighted mean. For dissimilar models this may produce incoherent output. Consider SLERP or FRANKENMERGE for better results.".to_string());
    }

    ValidationResult {
        valid: errors.is_empty(),
        errors,
        warnings,
    }
}

// ── Execution Commands ───────────────────────────────────

#[tauri::command]
pub async fn merge_execute(
    app: AppHandle,
    config: MergeConfig,
    state: State<'_, AppState>,
) -> Result<MergeResult, ModelError> {
    // Check if merge is already active
    if state.merge_active.load(Ordering::Relaxed) {
        return Err(ModelError::MergeError("A merge is already in progress".to_string()));
    }

    state.merge_active.store(true, Ordering::Relaxed);
    state.merge_cancel.store(false, Ordering::Relaxed);

    let cancel = state.merge_cancel.clone();

    // Snapshot the registry data and release the lock IMMEDIATELY.
    // This prevents all other commands from blocking while the merge runs.
    let snapshot = {
        let registry = state.merge_parents.lock().unwrap();
        crate::merge::registry::ParentRegistry::from_snapshot(registry.all().to_vec())
    };

    // Build plan from snapshot (lock already released)
    let plan = planner::build_plan(&config, &snapshot)?;

    let merge_active = state.merge_active.clone();

    // Run the heavy merge on a blocking thread so it doesn't freeze the async runtime
    let merge_result = tauri::async_runtime::spawn_blocking(move || {
        let result = executor::execute_merge(&app, &config, &plan, &snapshot, cancel);
        merge_active.store(false, Ordering::Relaxed);
        result
    })
    .await
    .map_err(|e| {
        state.merge_active.store(false, Ordering::Relaxed);
        ModelError::MergeError(format!("Task join error: {}", e))
    })?;

    merge_result
}

#[tauri::command]
pub fn merge_cancel(state: State<'_, AppState>) {
    state.merge_cancel.store(true, Ordering::Relaxed);
}

// ── Profiling Commands ───────────────────────────────────

#[tauri::command]
pub async fn merge_profile_layers(
    app: AppHandle,
    parent_id: String,
    state: State<'_, AppState>,
) -> Result<profiler::ProfileResult, ModelError> {
    state.profiler_cancel.store(false, Ordering::Relaxed);
    let cancel = state.profiler_cancel.clone();

    let (tensor_names, total_layers, pid) = {
        let registry = state.merge_parents.lock().unwrap();
        let parent = registry.get(&parent_id).ok_or_else(|| {
            ModelError::ParentNotFound(parent_id.clone())
        })?;
        (parent.compat.tensor_names.clone(), parent.layer_count.unwrap_or(32), parent_id.clone())
    };

    tauri::async_runtime::spawn_blocking(move || {
        profiler::logit_lens::profile_layers(&app, &pid, &tensor_names, total_layers, cancel)
    })
    .await
    .map_err(|e| ModelError::MergeError(format!("Task join error: {}", e)))?
}

#[tauri::command]
pub fn merge_profile_cancel(state: State<'_, AppState>) {
    state.profiler_cancel.store(true, Ordering::Relaxed);
}

// ── Utility Commands ─────────────────────────────────────

#[tauri::command]
pub fn merge_preview(
    config: MergeConfig,
    state: State<'_, AppState>,
) -> Result<planner::MergePreview, ModelError> {
    let registry = state.merge_parents.lock().unwrap();
    planner::preview_plan(&config, &registry)
}

#[tauri::command]
pub fn merge_get_methods() -> Vec<MergeMethodInfo> {
    MergeMethod::all().iter().map(|&m| MergeMethodInfo::from(m)).collect()
}

#[tauri::command]
pub fn merge_compare_tensors(
    state: State<'_, AppState>,
) -> TensorComparison {
    let registry = state.merge_parents.lock().unwrap();
    let parents = registry.all();

    if parents.is_empty() {
        return TensorComparison {
            entries: vec![],
            shared_count: 0,
            unique_count: 0,
        };
    }

    // Collect all tensor names with which parents have them
    let mut tensor_parents: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for parent in parents {
        for name in &parent.compat.tensor_names {
            tensor_parents
                .entry(name.clone())
                .or_default()
                .push(parent.id.clone());
        }
    }

    let total_parents = parents.len();
    let mut shared_count = 0;
    let mut unique_count = 0;

    let entries: Vec<TensorComparisonEntry> = tensor_parents
        .into_iter()
        .map(|(name, present_in)| {
            let is_shared = present_in.len() == total_parents;
            if is_shared {
                shared_count += 1;
            } else {
                unique_count += 1;
            }
            TensorComparisonEntry {
                tensor_name: name,
                present_in,
                shapes_match: true, // Simplified; full impl would compare shapes
            }
        })
        .collect();

    TensorComparison {
        entries,
        shared_count,
        unique_count,
    }
}

#[tauri::command]
pub async fn merge_analyze_layers(
    app: AppHandle,
    parent_id: String,
    state: State<'_, AppState>,
) -> Result<profiler::tensor_analysis::AnalysisResult, ModelError> {
    state.profiler_cancel.store(false, Ordering::Relaxed);
    let cancel = state.profiler_cancel.clone();

    let parent = {
        let registry = state.merge_parents.lock().unwrap();
        registry.get(&parent_id).ok_or_else(|| {
            ModelError::ParentNotFound(parent_id.clone())
        })?.clone()
    };

    // Run on a blocking thread to avoid freezing the UI
    tauri::async_runtime::spawn_blocking(move || {
        profiler::tensor_analysis::analyze_parent(&app, &parent, cancel)
    })
    .await
    .map_err(|e| ModelError::MergeError(format!("Task join error: {}", e)))?
}

#[tauri::command]
pub fn merge_get_categories() -> Vec<profiler::tensor_analysis::LayerCategory> {
    profiler::tensor_analysis::all_categories()
}

#[tauri::command]
pub fn merge_get_layer_components(
    parent_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<LayerComponentInfo>, ModelError> {
    let registry = state.merge_parents.lock().unwrap();
    let parent = registry.get(&parent_id).ok_or_else(|| {
        ModelError::ParentNotFound(parent_id.clone())
    })?;

    let total_layers = parent.layer_count.unwrap_or(0);
    let mut layers = Vec::new();

    for idx in 0..total_layers {
        let mut attn = Vec::new();
        let mut mlp = Vec::new();
        let mut norm = Vec::new();
        let mut other = Vec::new();
        let total_params: u64 = 0;

        for name in &parent.compat.tensor_names {
            if let Some(layer_idx) = inspect::extract_layer_index(name) {
                if layer_idx == idx {
                    let component = inspect::classify_tensor(name);
                    match component {
                        "attention" => attn.push(name.clone()),
                        "mlp" => mlp.push(name.clone()),
                        "norm" => norm.push(name.clone()),
                        _ => other.push(name.clone()),
                    }
                }
            }
        }

        layers.push(LayerComponentInfo {
            layer_index: idx,
            attention_tensors: attn,
            mlp_tensors: mlp,
            norm_tensors: norm,
            other_tensors: other,
            total_params,
        });
    }

    Ok(layers)
}
