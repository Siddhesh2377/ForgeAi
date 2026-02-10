use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use tauri::{AppHandle, State};

use crate::model::error::ModelError;
use crate::model::state::AppState;
use crate::model::{self, ModelFormat, TensorInfo};
use crate::merge::capabilities;
use crate::training::config::{
    DatasetFullInfo, DatasetInfo, LayerCapabilityMapping, SurgeryConfig, SurgeryResult,
    TargetModuleGroup, TrainingConfig, TrainingDepsStatus, TrainingLayerDetail,
    TrainingResult, LayerTensorInfo,
};
use crate::training::{datasets, executor, surgery, venv};

// ── Check Dependencies ──────────────────────────────

#[tauri::command]
pub async fn training_check_deps(app: AppHandle) -> Result<TrainingDepsStatus, ModelError> {
    let training_dir = venv::get_training_dir(&app)?;
    Ok(venv::check_training_deps(&training_dir))
}

// ── Setup Environment ───────────────────────────────

#[tauri::command]
pub async fn training_setup(app: AppHandle) -> Result<(), ModelError> {
    venv::setup_training_env(&app).await
}

// ── Dataset Detection ───────────────────────────────

#[tauri::command]
pub fn training_detect_dataset(path: String) -> Result<DatasetInfo, ModelError> {
    datasets::detect_dataset(&path)
}

// ── Run Training ────────────────────────────────────

#[tauri::command]
pub async fn training_run(
    config: TrainingConfig,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<TrainingResult, ModelError> {
    let cancel = state.training_cancel.clone();
    cancel.store(false, Ordering::Relaxed);

    let pid_store = Arc::new(Mutex::new(None::<u32>));

    // Store PID reference in AppState
    {
        let mut pid_lock = state.training_pid.lock().unwrap();
        *pid_lock = None;
    }

    let result = executor::run_training(app, config, cancel.clone(), pid_store).await;

    // Clean up PID
    {
        let mut pid_lock = state.training_pid.lock().unwrap();
        *pid_lock = None;
    }

    result
}

// ── Cancel Training ─────────────────────────────────

#[tauri::command]
pub fn training_cancel(state: State<'_, AppState>) -> Result<(), ModelError> {
    state.training_cancel.store(true, Ordering::Relaxed);

    // Also try to kill the process directly
    if let Some(pid) = *state.training_pid.lock().unwrap() {
        #[cfg(unix)]
        {
            let _ = std::process::Command::new("kill")
                .arg(pid.to_string())
                .output();
        }
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .output();
        }
    }

    Ok(())
}

// ── Surgery ─────────────────────────────────────────

#[tauri::command]
pub fn training_surgery_run(
    config: SurgeryConfig,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<SurgeryResult, ModelError> {
    let cancel = state.surgery_cancel.clone();
    cancel.store(false, Ordering::Relaxed);

    surgery::execute_surgery(&app, &config, cancel)
}

#[tauri::command]
pub fn training_surgery_cancel(state: State<'_, AppState>) -> Result<(), ModelError> {
    state.surgery_cancel.store(true, Ordering::Relaxed);
    Ok(())
}

// ── Clean Environment ────────────────────────────────

#[tauri::command]
pub async fn training_clean_env(app: AppHandle) -> Result<(), ModelError> {
    let training_dir = venv::get_training_dir(&app)?;
    if training_dir.exists() {
        std::fs::remove_dir_all(&training_dir)
            .map_err(|e| ModelError::TrainingError(format!("Failed to remove training environment: {}", e)))?;
    }
    Ok(())
}

// ── Helpers ─────────────────────────────────────────

/// Quick-parse a model path to get tensor names and layer count without storing in AppState.
fn quick_parse_model(model_path: &str) -> Result<(Vec<TensorInfo>, Option<u64>), ModelError> {
    let p = Path::new(model_path);

    if p.is_dir() {
        let info = model::safetensors::parse_dir(p)?;
        let layer_count = info.layer_count.or_else(|| model::derive_layer_count(&info.all_tensors));
        Ok((info.all_tensors, layer_count))
    } else if model_path.ends_with(".gguf") {
        let info = model::gguf::parse(p)?;
        let layer_count = info.layer_count.or_else(|| model::derive_layer_count(&info.all_tensors));
        Ok((info.all_tensors, layer_count))
    } else if model_path.ends_with(".safetensors") {
        let info = model::safetensors::parse(p)?;
        let layer_count = info.layer_count.or_else(|| model::derive_layer_count(&info.all_tensors));
        Ok((info.all_tensors, layer_count))
    } else {
        Err(ModelError::TrainingError("Unsupported model format. Use a SafeTensors directory or GGUF file.".into()))
    }
}

// ── Target Modules ──────────────────────────────────

#[tauri::command]
pub fn training_get_target_modules(
    model_path: String,
) -> Result<Vec<TargetModuleGroup>, ModelError> {
    let (tensors, _) = quick_parse_model(&model_path)?;

    // Analyze tensor names to detect module groups
    let mut has_q_proj = false;
    let mut has_k_proj = false;
    let mut has_v_proj = false;
    let mut has_o_proj = false;
    let mut has_gate_proj = false;
    let mut has_up_proj = false;
    let mut has_down_proj = false;
    let mut has_w1 = false;
    let mut has_w2 = false;
    let mut has_w3 = false;
    let mut has_qkv = false;

    for tensor in &tensors {
        let name = &tensor.name;
        if name.contains("q_proj") { has_q_proj = true; }
        if name.contains("k_proj") { has_k_proj = true; }
        if name.contains("v_proj") { has_v_proj = true; }
        if name.contains("o_proj") { has_o_proj = true; }
        if name.contains("gate_proj") { has_gate_proj = true; }
        if name.contains("up_proj") { has_up_proj = true; }
        if name.contains("down_proj") { has_down_proj = true; }
        if name.contains(".w1") { has_w1 = true; }
        if name.contains(".w2") { has_w2 = true; }
        if name.contains(".w3") { has_w3 = true; }
        if name.contains("qkv_proj") || name.contains("query_key_value") { has_qkv = true; }
    }

    let mut groups = vec![];

    // Llama/Mistral style
    if has_q_proj || has_k_proj || has_v_proj {
        let mut modules = vec![];
        if has_q_proj { modules.push("q_proj".to_string()); }
        if has_k_proj { modules.push("k_proj".to_string()); }
        if has_v_proj { modules.push("v_proj".to_string()); }
        groups.push(TargetModuleGroup {
            name: "Attention Q/K/V".into(),
            modules,
        });
    }

    if has_o_proj {
        groups.push(TargetModuleGroup {
            name: "Attention Output".into(),
            modules: vec!["o_proj".to_string()],
        });
    }

    if has_gate_proj || has_up_proj || has_down_proj {
        let mut modules = vec![];
        if has_gate_proj { modules.push("gate_proj".to_string()); }
        if has_up_proj { modules.push("up_proj".to_string()); }
        if has_down_proj { modules.push("down_proj".to_string()); }
        groups.push(TargetModuleGroup {
            name: "MLP".into(),
            modules,
        });
    }

    // GPT-NeoX / Falcon style
    if has_qkv {
        groups.push(TargetModuleGroup {
            name: "Fused QKV".into(),
            modules: vec!["query_key_value".to_string()],
        });
    }

    // GPT-2 / older style
    if has_w1 || has_w2 || has_w3 {
        let mut modules = vec![];
        if has_w1 { modules.push("w1".to_string()); }
        if has_w2 { modules.push("w2".to_string()); }
        if has_w3 { modules.push("w3".to_string()); }
        groups.push(TargetModuleGroup {
            name: "Feed Forward".into(),
            modules,
        });
    }

    // Fallback
    if groups.is_empty() {
        groups.push(TargetModuleGroup {
            name: "Default".into(),
            modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        });
    }

    Ok(groups)
}

// ── Layer Capabilities ──────────────────────────────

#[tauri::command]
pub fn training_get_layer_capabilities(
    model_path: String,
) -> Result<Vec<LayerCapabilityMapping>, ModelError> {
    let (_, layer_count) = quick_parse_model(&model_path)?;
    let total_layers = layer_count.unwrap_or(32);

    let capability_ids = [
        ("tool_calling", "TOOL CALLING"),
        ("reasoning", "REASONING / COT"),
        ("code", "CODE GENERATION"),
        ("math", "MATHEMATICS"),
        ("multilingual", "MULTILINGUAL"),
        ("instruct", "INSTRUCTION FOLLOWING"),
        ("safety", "SAFETY & ALIGNMENT"),
    ];

    let mappings: Vec<LayerCapabilityMapping> = capability_ids
        .iter()
        .map(|&(id, name)| {
            let layers = capabilities::compute_affected_layers(id, total_layers);
            LayerCapabilityMapping {
                capability: id.to_string(),
                name: name.to_string(),
                layers,
            }
        })
        .collect();

    Ok(mappings)
}

// ── Layer Details (Surgery View) ────────────────────

#[tauri::command]
pub fn training_get_layer_details(
    model_path: String,
) -> Result<Vec<TrainingLayerDetail>, ModelError> {
    use std::collections::BTreeMap;
    use crate::model::inspect;

    let (tensors, layer_count) = quick_parse_model(&model_path)?;
    let total_layers = layer_count.unwrap_or(32);

    // Build capability lookup: layer_index -> Vec<capability_id>
    let capability_ids = [
        "tool_calling", "reasoning", "code", "math", "multilingual", "instruct", "safety",
    ];
    let mut cap_map: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    for id in &capability_ids {
        for layer_idx in capabilities::compute_affected_layers(id, total_layers) {
            cap_map.entry(layer_idx).or_default().push(id.to_string());
        }
    }

    // Group tensors by layer
    let mut layer_tensors: BTreeMap<u64, Vec<&TensorInfo>> = BTreeMap::new();
    for t in &tensors {
        if let Some(idx) = inspect::extract_layer_index(&t.name) {
            layer_tensors.entry(idx).or_default().push(t);
        }
    }

    let mut details: Vec<TrainingLayerDetail> = Vec::new();

    for layer_idx in 0..total_layers {
        let tensors_in_layer = layer_tensors.get(&layer_idx).cloned().unwrap_or_default();

        let mut attention_count = 0u32;
        let mut mlp_count = 0u32;
        let mut norm_count = 0u32;
        let mut other_count = 0u32;
        let mut attention_bytes = 0u64;
        let mut mlp_bytes = 0u64;
        let mut norm_bytes = 0u64;
        let mut other_bytes = 0u64;
        let mut total_bytes = 0u64;
        let mut tensor_infos: Vec<LayerTensorInfo> = Vec::new();

        for t in &tensors_in_layer {
            let component = inspect::classify_tensor(&t.name);
            let bytes = inspect::tensor_memory_bytes(&t.dtype, &t.shape);
            total_bytes += bytes;

            match component {
                "attention" => { attention_count += 1; attention_bytes += bytes; }
                "mlp" => { mlp_count += 1; mlp_bytes += bytes; }
                "norm" => { norm_count += 1; norm_bytes += bytes; }
                _ => { other_count += 1; other_bytes += bytes; }
            }

            // Extract short name from full tensor name
            let short_name = t.name.rsplit('.').next().unwrap_or(&t.name).to_string();

            tensor_infos.push(LayerTensorInfo {
                name: short_name,
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                memory_display: inspect::format_bytes(bytes),
                component: component.to_string(),
            });
        }

        let caps = cap_map.get(&layer_idx).cloned().unwrap_or_default();

        details.push(TrainingLayerDetail {
            index: layer_idx,
            total_bytes,
            display: inspect::format_bytes(total_bytes),
            attention_count,
            mlp_count,
            norm_count,
            other_count,
            attention_bytes,
            mlp_bytes,
            norm_bytes,
            other_bytes,
            capabilities: caps,
            tensors: tensor_infos,
        });
    }

    Ok(details)
}

// ── Full Dataset Detection (DataStudio) ─────────────

#[tauri::command]
pub fn training_detect_dataset_full(path: String) -> Result<DatasetFullInfo, ModelError> {
    datasets::detect_dataset_full(&path, 50)
}
