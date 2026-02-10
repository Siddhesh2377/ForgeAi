use serde::{Deserialize, Serialize};
use std::path::Path;

use super::registry::ParentModel;

// ── Capability Structs ──────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub id: String,
    pub name: String,
    pub detected: bool,
    pub confidence: f64,
    pub evidence: Vec<String>,
    pub affected_layers: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityReport {
    pub parent_id: String,
    pub parent_name: String,
    pub capabilities: Vec<Capability>,
    pub total_detected: usize,
}

// ── Layer Range Mapping ─────────────────────────────────

/// Returns (start_fraction, end_fraction) for where a capability's
/// affected layers typically reside in a transformer model.
pub(crate) fn capability_layer_range(id: &str) -> (f64, f64) {
    match id {
        "tool_calling" => (0.60, 0.90),
        "reasoning" => (0.55, 0.85),
        "code" => (0.60, 0.88),
        "math" => (0.55, 0.85),
        "multilingual" => (0.15, 0.45),
        "instruct" => (0.20, 0.55),
        "safety" => (0.75, 0.95),
        "multimodal" => (0.0, 0.30),
        "moe" => (0.0, 1.0), // all layers
        _ => (0.0, 1.0),
    }
}

pub(crate) fn compute_affected_layers(id: &str, total_layers: u64) -> Vec<u64> {
    if total_layers == 0 {
        return vec![];
    }
    let (start, end) = capability_layer_range(id);
    let first = (start * total_layers as f64).floor() as u64;
    let last = (end * total_layers as f64).ceil().min(total_layers as f64) as u64;
    (first..last).collect()
}

// ── Helper: Read tokenizer tokens ───────────────────────

fn read_tokenizer_tokens(dir_path: &str) -> Vec<String> {
    let tokenizer_path = Path::new(dir_path).join("tokenizer.json");
    if !tokenizer_path.exists() {
        return vec![];
    }

    let content = match std::fs::read_to_string(&tokenizer_path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    let mut tokens = Vec::new();

    // Extract added_tokens[].content
    if let Some(added) = json.get("added_tokens").and_then(|a| a.as_array()) {
        for token in added {
            if let Some(content) = token.get("content").and_then(|c| c.as_str()) {
                tokens.push(content.to_string());
            }
        }
    }

    tokens
}

/// Read config.json from the model directory.
fn read_config_json(dir_path: &str) -> Option<serde_json::Value> {
    let config_path = Path::new(dir_path).join("config.json");
    if !config_path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&config_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Resolve the model directory for a parent model.
/// For directories: the dir itself. For files: parent dir.
fn resolve_model_dir(parent: &ParentModel) -> Option<String> {
    if parent.is_dir {
        Some(parent.file_path.clone())
    } else {
        Path::new(&parent.file_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
    }
}

// ── Detection Engine ────────────────────────────────────

pub fn detect_capabilities(parent: &ParentModel) -> CapabilityReport {
    let total_layers = parent.layer_count.unwrap_or(0);
    let model_name = parent.name.to_lowercase();
    let dir = resolve_model_dir(parent);

    // Gather evidence sources
    let tokenizer_tokens = dir.as_deref().map(read_tokenizer_tokens).unwrap_or_default();
    let config_json = dir.as_deref().and_then(read_config_json);
    let tensor_names: Vec<String> = parent.compat.tensor_names();

    // GGUF metadata (chat_template, etc.)
    // We don't have direct access to raw metadata here, but the compat info
    // contains some architecture data. We'll use tensor names + tokenizer.

    let mut capabilities = vec![
        detect_tool_calling(&tokenizer_tokens, &model_name, total_layers),
        detect_reasoning(&tokenizer_tokens, &model_name, total_layers),
        detect_code(&tokenizer_tokens, &model_name, total_layers),
        detect_math(&tokenizer_tokens, &model_name, total_layers),
        detect_multilingual(&tokenizer_tokens, parent.compat.vocab_size, total_layers),
        detect_instruct(&tokenizer_tokens, &model_name, total_layers),
        detect_safety(&model_name, total_layers),
        detect_multimodal(&config_json, &tensor_names, total_layers),
        detect_moe(&tensor_names, &parent.compat, total_layers),
    ];

    let total_detected = capabilities.iter().filter(|c| c.detected).count();

    // Sort: detected first, then by confidence descending
    capabilities.sort_by(|a, b| {
        b.detected.cmp(&a.detected)
            .then(b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal))
    });

    CapabilityReport {
        parent_id: parent.id.clone(),
        parent_name: parent.name.clone(),
        capabilities,
        total_detected,
    }
}

// ── Individual Detectors ────────────────────────────────

fn detect_tool_calling(tokens: &[String], model_name: &str, total_layers: u64) -> Capability {
    let mut evidence = Vec::new();
    let mut confidence: f64 = 0.0;

    let tool_tokens = [
        "<|tool_call|>", "<function>", "<|python_tag|>", "<tool_call>",
        "</tool_call>", "<|tool_start|>", "<|tool_end|>", "<|plugin|>",
        "[TOOL_CALLS]", "<|startofaction|>",
    ];

    for pattern in &tool_tokens {
        if tokens.iter().any(|t| t.contains(pattern)) {
            evidence.push(format!("Token: {}", pattern));
            confidence += 0.3;
        }
    }

    if model_name.contains("function") || model_name.contains("tool") {
        evidence.push("Model name suggests tool-calling".into());
        confidence += 0.2;
    }

    confidence = confidence.min(1.0);
    let detected = confidence >= 0.2;

    Capability {
        id: "tool_calling".into(),
        name: "TOOL CALLING".into(),
        detected,
        confidence: if detected { confidence } else { 0.0 },
        evidence,
        affected_layers: if detected { compute_affected_layers("tool_calling", total_layers) } else { vec![] },
    }
}

fn detect_reasoning(tokens: &[String], model_name: &str, total_layers: u64) -> Capability {
    let mut evidence = Vec::new();
    let mut confidence: f64 = 0.0;

    let reasoning_tokens = [
        "<think>", "</think>", "<|thinking|>", "<|/thinking|>",
        "<reasoning>", "</reasoning>", "<|thought|>",
    ];

    for pattern in &reasoning_tokens {
        if tokens.iter().any(|t| t.contains(pattern)) {
            evidence.push(format!("Token: {}", pattern));
            confidence += 0.35;
        }
    }

    let name_hints = ["r1", "cot", "think", "reason", "o1", "o3"];
    for hint in &name_hints {
        if model_name.contains(hint) {
            evidence.push(format!("Model name contains '{}'", hint));
            confidence += 0.25;
        }
    }

    confidence = confidence.min(1.0);
    let detected = confidence >= 0.2;

    Capability {
        id: "reasoning".into(),
        name: "REASONING / COT".into(),
        detected,
        confidence: if detected { confidence } else { 0.0 },
        evidence,
        affected_layers: if detected { compute_affected_layers("reasoning", total_layers) } else { vec![] },
    }
}

fn detect_code(tokens: &[String], model_name: &str, total_layers: u64) -> Capability {
    let mut evidence = Vec::new();
    let mut confidence: f64 = 0.0;

    let code_tokens = [
        "<|code|>", "<|python|>", "<|javascript|>", "<|java|>",
        "<|code_start|>", "<|code_end|>", "<|fim_prefix|>", "<|fim_suffix|>",
        "<|fim_middle|>",
    ];

    for pattern in &code_tokens {
        if tokens.iter().any(|t| t.contains(pattern)) {
            evidence.push(format!("Token: {}", pattern));
            confidence += 0.3;
        }
    }

    let name_hints = ["code", "coder", "codex", "starcoder", "deepseek-coder", "codellama"];
    for hint in &name_hints {
        if model_name.contains(hint) {
            evidence.push(format!("Model name contains '{}'", hint));
            confidence += 0.35;
        }
    }

    confidence = confidence.min(1.0);
    let detected = confidence >= 0.2;

    Capability {
        id: "code".into(),
        name: "CODE GENERATION".into(),
        detected,
        confidence: if detected { confidence } else { 0.0 },
        evidence,
        affected_layers: if detected { compute_affected_layers("code", total_layers) } else { vec![] },
    }
}

fn detect_math(tokens: &[String], model_name: &str, total_layers: u64) -> Capability {
    let mut evidence = Vec::new();
    let mut confidence: f64 = 0.0;

    let math_tokens = ["<|math|>", "<|equation|>", "<|latex|>"];
    for pattern in &math_tokens {
        if tokens.iter().any(|t| t.contains(pattern)) {
            evidence.push(format!("Token: {}", pattern));
            confidence += 0.3;
        }
    }

    let name_hints = ["math", "mathcoder", "wizard-math", "llemma", "minerva"];
    for hint in &name_hints {
        if model_name.contains(hint) {
            evidence.push(format!("Model name contains '{}'", hint));
            confidence += 0.35;
        }
    }

    confidence = confidence.min(1.0);
    let detected = confidence >= 0.2;

    Capability {
        id: "math".into(),
        name: "MATHEMATICAL REASONING".into(),
        detected,
        confidence: if detected { confidence } else { 0.0 },
        evidence,
        affected_layers: if detected { compute_affected_layers("math", total_layers) } else { vec![] },
    }
}

fn detect_multilingual(tokens: &[String], vocab_size: Option<u64>, total_layers: u64) -> Capability {
    let mut evidence = Vec::new();
    let mut confidence: f64 = 0.0;

    if let Some(vs) = vocab_size {
        if vs > 100_000 {
            evidence.push(format!("Large vocabulary: {} tokens", vs));
            confidence += 0.4;
        } else if vs > 64_000 {
            evidence.push(format!("Medium-large vocabulary: {} tokens", vs));
            confidence += 0.2;
        }
    }

    // Look for CJK, Arabic, or other non-Latin script tokens
    let script_indicators = ["▁", "##", "Ġ"]; // common multilingual subword markers
    let mut multilingual_count = 0;
    for token in tokens {
        if token.chars().any(|c| c > '\u{2E80}') || script_indicators.iter().any(|s| token.contains(s)) {
            multilingual_count += 1;
        }
    }
    if multilingual_count > 20 {
        evidence.push(format!("{} multilingual tokens detected", multilingual_count));
        confidence += 0.3;
    }

    confidence = confidence.min(1.0);
    let detected = confidence >= 0.3;

    Capability {
        id: "multilingual".into(),
        name: "MULTILINGUAL".into(),
        detected,
        confidence: if detected { confidence } else { 0.0 },
        evidence,
        affected_layers: if detected { compute_affected_layers("multilingual", total_layers) } else { vec![] },
    }
}

fn detect_instruct(tokens: &[String], model_name: &str, total_layers: u64) -> Capability {
    let mut evidence = Vec::new();
    let mut confidence: f64 = 0.0;

    let instruct_tokens = [
        "<|im_start|>", "<|im_end|>", "[INST]", "[/INST]",
        "<|system|>", "<|user|>", "<|assistant|>",
        "<|start_header_id|>", "<|end_header_id|>",
        "<|begin_of_text|>", "<s>",
    ];

    for pattern in &instruct_tokens {
        if tokens.iter().any(|t| t.contains(pattern)) {
            evidence.push(format!("Token: {}", pattern));
            confidence += 0.2;
        }
    }

    let name_hints = ["instruct", "chat", "it", "dpo", "rlhf", "sft"];
    for hint in &name_hints {
        if model_name.contains(hint) {
            evidence.push(format!("Model name contains '{}'", hint));
            confidence += 0.2;
        }
    }

    confidence = confidence.min(1.0);
    let detected = confidence >= 0.2;

    Capability {
        id: "instruct".into(),
        name: "INSTRUCTION FOLLOWING".into(),
        detected,
        confidence: if detected { confidence } else { 0.0 },
        evidence,
        affected_layers: if detected { compute_affected_layers("instruct", total_layers) } else { vec![] },
    }
}

fn detect_safety(model_name: &str, total_layers: u64) -> Capability {
    let mut evidence = Vec::new();
    let mut confidence: f64 = 0.0;

    let name_hints = ["rlhf", "aligned", "safe", "guard", "moderation"];
    for hint in &name_hints {
        if model_name.contains(hint) {
            evidence.push(format!("Model name contains '{}'", hint));
            confidence += 0.3;
        }
    }

    confidence = confidence.min(1.0);
    let detected = confidence >= 0.2;

    Capability {
        id: "safety".into(),
        name: "SAFETY ALIGNMENT".into(),
        detected,
        confidence: if detected { confidence } else { 0.0 },
        evidence,
        affected_layers: if detected { compute_affected_layers("safety", total_layers) } else { vec![] },
    }
}

fn detect_multimodal(
    config_json: &Option<serde_json::Value>,
    tensor_names: &[String],
    total_layers: u64,
) -> Capability {
    let mut evidence = Vec::new();
    let mut confidence: f64 = 0.0;

    // Check config.json for vision_config
    if let Some(config) = config_json {
        if config.get("vision_config").is_some() {
            evidence.push("vision_config found in config.json".into());
            confidence += 0.5;
        }
        if config.get("image_size").is_some() || config.get("patch_size").is_some() {
            evidence.push("Image processing config found".into());
            confidence += 0.3;
        }
    }

    // Check tensor names for visual/image patterns
    let visual_count = tensor_names.iter().filter(|n| {
        let lower = n.to_lowercase();
        lower.contains("visual") || lower.contains("image_") || lower.contains("vision_")
            || lower.contains("vit.") || lower.contains("clip.")
    }).count();

    if visual_count > 0 {
        evidence.push(format!("{} visual tensors found", visual_count));
        confidence += 0.4;
    }

    confidence = confidence.min(1.0);
    let detected = confidence >= 0.3;

    Capability {
        id: "multimodal".into(),
        name: "MULTIMODAL / VISION".into(),
        detected,
        confidence: if detected { confidence } else { 0.0 },
        evidence,
        affected_layers: if detected { compute_affected_layers("multimodal", total_layers) } else { vec![] },
    }
}

fn detect_moe(
    tensor_names: &[String],
    compat: &super::registry::CompatInfo,
    total_layers: u64,
) -> Capability {
    let mut evidence = Vec::new();
    let mut confidence: f64 = 0.0;

    // Check tensor names for expert/router patterns
    let expert_count = tensor_names.iter().filter(|n| {
        let lower = n.to_lowercase();
        lower.contains("experts.") || lower.contains("expert_")
    }).count();

    let router_count = tensor_names.iter().filter(|n| {
        let lower = n.to_lowercase();
        lower.contains("router.") || lower.contains("gate.weight")
    }).count();

    if expert_count > 0 {
        evidence.push(format!("{} expert tensors found", expert_count));
        confidence += 0.5;
    }
    if router_count > 0 {
        evidence.push(format!("{} router tensors found", router_count));
        confidence += 0.3;
    }

    // Check for num_experts in architecture metadata (we approximate via tensor count patterns)
    if compat.architecture.as_deref() == Some("mixtral") || compat.architecture.as_deref() == Some("qwen2moe") {
        evidence.push("MoE architecture detected".into());
        confidence += 0.5;
    }

    confidence = confidence.min(1.0);
    let detected = confidence >= 0.3;

    Capability {
        id: "moe".into(),
        name: "MIXTURE OF EXPERTS".into(),
        detected,
        confidence: if detected { confidence } else { 0.0 },
        evidence,
        affected_layers: if detected { compute_affected_layers("moe", total_layers) } else { vec![] },
    }
}
