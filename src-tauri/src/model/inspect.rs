use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::TensorInfo;

/// Bits per weight for each GGML/quantization type.
pub fn bits_per_weight(dtype: &str) -> f64 {
    match dtype {
        "F32" => 32.0,
        "F16" => 16.0,
        "BF16" => 16.0,
        "F64" => 64.0,
        "Q4_0" => 4.5,
        "Q4_1" => 5.0,
        "Q5_0" => 5.5,
        "Q5_1" => 6.0,
        "Q8_0" => 8.5,
        "Q8_1" => 9.0,
        "Q2_K" => 2.56,
        "Q3_K" => 3.44,
        "Q4_K" => 4.5,
        "Q5_K" => 5.5,
        "Q6_K" => 6.56,
        "Q8_K" => 8.5,
        "IQ1_S" => 1.56,
        "IQ1_M" => 1.75,
        "IQ2_XXS" => 2.06,
        "IQ2_XS" => 2.31,
        "IQ2_S" => 2.5,
        "IQ3_XXS" => 3.06,
        "IQ3_S" => 3.44,
        "IQ4_NL" => 4.5,
        "IQ4_XS" => 4.25,
        "I8" => 8.0,
        "I16" => 16.0,
        "I32" => 32.0,
        "I64" => 64.0,
        _ => 16.0, // default to F16 if unknown
    }
}

/// Calculate memory in bytes for a tensor given its dtype and shape.
pub fn tensor_memory_bytes(dtype: &str, shape: &[u64]) -> u64 {
    if shape.is_empty() {
        return 0;
    }
    let elements: u64 = shape.iter().product();
    let bpw = bits_per_weight(dtype);
    ((elements as f64 * bpw) / 8.0).ceil() as u64
}

pub fn format_bytes(bytes: u64) -> String {
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

/// Classify a tensor by its component type based on name patterns.
pub fn classify_tensor(name: &str) -> &'static str {
    let lower = name.to_lowercase();

    // Embeddings
    if lower.contains("token_embd")
        || lower.contains("embed_tokens")
        || lower.contains("wte")
        || lower.contains("word_embedding")
    {
        return "embedding";
    }

    // Output head
    if (lower.contains("output") && !lower.contains("output.weight") == false)
        && !lower.contains("attn_output")
        && !lower.contains("o_proj")
        && !lower.contains("out_proj")
    {
        // Careful: "output.weight" is the LM head, but "attn_output" is attention
    }
    if lower == "output.weight"
        || lower.contains("lm_head")
        || lower.contains("output_norm")
        || (lower.starts_with("output.") && !lower.contains("attn"))
    {
        return "output";
    }

    // Attention
    if lower.contains("attn_q")
        || lower.contains("attn_k")
        || lower.contains("attn_v")
        || lower.contains("attn_output")
        || lower.contains("q_proj")
        || lower.contains("k_proj")
        || lower.contains("v_proj")
        || lower.contains("o_proj")
        || lower.contains("self_attn")
        || lower.contains("qkv_proj")
        || lower.contains("attn.c_attn")
        || lower.contains("attn.c_proj")
    {
        return "attention";
    }

    // Layer norms
    if lower.contains("norm")
        || lower.contains("layernorm")
        || lower.contains("ln_")
        || lower.contains("layer_norm")
        || lower.contains("rms_norm")
        || lower.contains("input_layernorm")
        || lower.contains("post_attention_layernorm")
    {
        return "norm";
    }

    // MLP / Feed-forward
    if lower.contains("ffn_")
        || lower.contains("mlp")
        || lower.contains("gate_proj")
        || lower.contains("up_proj")
        || lower.contains("down_proj")
        || lower.contains("fc1")
        || lower.contains("fc2")
        || lower.contains("c_fc")
        || lower.contains("c_proj")
        || lower.contains("feed_forward")
    {
        return "mlp";
    }

    "other"
}

/// Extract layer index from a tensor name.
pub fn extract_layer_index(name: &str) -> Option<u64> {
    let patterns = ["blk.", "layers.", "blocks.", "h.", "layer."];
    for pat in &patterns {
        if let Some(pos) = name.find(pat) {
            let after = &name[pos + pat.len()..];
            if let Some(end) = after.find('.') {
                if let Ok(idx) = after[..end].parse::<u64>() {
                    return Some(idx);
                }
            } else if let Ok(idx) = after.parse::<u64>() {
                return Some(idx);
            }
        }
    }
    None
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub memory_bytes: u64,
    pub memory_display: String,
    pub component: String,
    pub params: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryComponent {
    pub name: String,
    pub bytes: u64,
    pub display: String,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantEntry {
    pub dtype: String,
    pub count: u64,
    pub total_bytes: u64,
    pub display: String,
    pub percentage: f64,
    pub total_params: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGroup {
    pub index: u64,
    pub total_bytes: u64,
    pub display: String,
    pub attention: Vec<InspectTensor>,
    pub mlp: Vec<InspectTensor>,
    pub norms: Vec<InspectTensor>,
    pub other: Vec<InspectTensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionInfo {
    pub attention_type: String,
    pub q_heads: Option<u64>,
    pub kv_heads: Option<u64>,
    pub head_dim: Option<u64>,
    pub gqa_ratio: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigEntry {
    pub label: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub entries: Vec<ConfigEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialToken {
    pub role: String,
    pub id: u64,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerInfo {
    pub tokenizer_type: Option<String>,
    pub vocab_size: Option<u64>,
    pub special_tokens: Vec<SpecialToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectData {
    pub memory_breakdown: Vec<MemoryComponent>,
    pub total_memory_bytes: u64,
    pub total_memory_display: String,
    pub quant_distribution: Vec<QuantEntry>,
    pub layers: Vec<LayerGroup>,
    pub other_tensors: Vec<InspectTensor>,
    pub attention_info: Option<AttentionInfo>,
    pub model_config: ModelConfig,
    pub tokenizer_info: Option<TokenizerInfo>,
    pub tensor_count: u64,
    pub total_params: u64,
    pub total_params_display: String,
}

fn extract_model_config(metadata: &HashMap<String, String>) -> ModelConfig {
    let arch = metadata
        .get("general.architecture")
        .cloned()
        .unwrap_or_default();

    let keys: Vec<(String, &str)> = vec![
        (format!("{}.context_length", arch), "CONTEXT LENGTH"),
        (format!("{}.embedding_length", arch), "EMBEDDING DIM"),
        (format!("{}.feed_forward_length", arch), "FFN LENGTH"),
        (format!("{}.block_count", arch), "BLOCKS"),
        ("tokenizer.ggml.tokens_count".to_string(), "VOCAB SIZE"),
        (
            format!("{}.attention.head_count", arch),
            "ATTENTION HEADS",
        ),
        (format!("{}.attention.head_count_kv", arch), "KV HEADS"),
        (format!("{}.rope.freq_base", arch), "ROPE FREQ BASE"),
        (
            format!("{}.rope.dimension_count", arch),
            "ROPE DIMENSIONS",
        ),
        (
            format!("{}.attention.layer_norm_rms_epsilon", arch),
            "NORM EPSILON",
        ),
    ];

    let entries: Vec<ConfigEntry> = keys
        .into_iter()
        .filter_map(|(key, label)| {
            metadata.get(&key).map(|value| ConfigEntry {
                label: label.to_string(),
                value: value.clone(),
            })
        })
        .collect();

    ModelConfig { entries }
}

fn extract_tokenizer_info(metadata: &HashMap<String, String>) -> Option<TokenizerInfo> {
    let tokenizer_type = metadata.get("tokenizer.ggml.model").cloned();
    let vocab_size = metadata
        .get("tokenizer.ggml.tokens_count")
        .and_then(|v| v.parse::<u64>().ok());

    if tokenizer_type.is_none() && vocab_size.is_none() {
        return None;
    }

    let token_defs = [
        ("BOS", "tokenizer.ggml.bos_token_id"),
        ("EOS", "tokenizer.ggml.eos_token_id"),
        ("PAD", "tokenizer.ggml.padding_token_id"),
        ("UNK", "tokenizer.ggml.unknown_token_id"),
    ];

    let mut special_tokens = Vec::new();
    for (role, key) in &token_defs {
        if let Some(id_str) = metadata.get(*key) {
            if let Ok(id) = id_str.parse::<u64>() {
                let resolved = metadata.get(&format!("{}_resolved", key)).cloned();
                special_tokens.push(SpecialToken {
                    role: role.to_string(),
                    id,
                    token: resolved,
                });
            }
        }
    }

    Some(TokenizerInfo {
        tokenizer_type,
        vocab_size,
        special_tokens,
    })
}

pub fn analyze(tensors: &[TensorInfo], metadata: &HashMap<String, String>) -> InspectData {
    // Build inspect tensors with memory calculation
    let mut inspect_tensors: Vec<InspectTensor> = tensors
        .iter()
        .map(|t| {
            let mem = tensor_memory_bytes(&t.dtype, &t.shape);
            let params: u64 = if t.shape.is_empty() {
                0
            } else {
                t.shape.iter().product()
            };
            InspectTensor {
                name: t.name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                memory_bytes: mem,
                memory_display: format_bytes(mem),
                component: classify_tensor(&t.name).to_string(),
                params,
            }
        })
        .collect();

    inspect_tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let total_memory: u64 = inspect_tensors.iter().map(|t| t.memory_bytes).sum();
    let total_params: u64 = inspect_tensors.iter().map(|t| t.params).sum();

    // Memory breakdown by component
    let mut component_bytes: HashMap<&str, u64> = HashMap::new();
    for t in &inspect_tensors {
        *component_bytes.entry(leak_str(&t.component)).or_insert(0) += t.memory_bytes;
    }

    let component_order = ["embedding", "attention", "mlp", "norm", "output", "other"];
    let component_labels = [
        "Token Embeddings",
        "Attention Weights",
        "MLP / Feed-Forward",
        "Layer Norms",
        "Output Head",
        "Other",
    ];

    let memory_breakdown: Vec<MemoryComponent> = component_order
        .iter()
        .zip(component_labels.iter())
        .filter_map(|(key, label)| {
            let bytes = *component_bytes.get(key).unwrap_or(&0);
            if bytes == 0 {
                return None;
            }
            let pct = if total_memory > 0 {
                (bytes as f64 / total_memory as f64) * 100.0
            } else {
                0.0
            };
            Some(MemoryComponent {
                name: label.to_string(),
                bytes,
                display: format_bytes(bytes),
                percentage: (pct * 10.0).round() / 10.0,
            })
        })
        .collect();

    // Quantization distribution
    let mut quant_map: HashMap<String, (u64, u64, u64)> = HashMap::new(); // dtype -> (count, bytes, params)
    for t in &inspect_tensors {
        let entry = quant_map.entry(t.dtype.clone()).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += t.memory_bytes;
        entry.2 += t.params;
    }

    let mut quant_distribution: Vec<QuantEntry> = quant_map
        .into_iter()
        .map(|(dtype, (count, bytes, params))| {
            let pct = if total_memory > 0 {
                (bytes as f64 / total_memory as f64) * 100.0
            } else {
                0.0
            };
            QuantEntry {
                dtype,
                count,
                total_bytes: bytes,
                display: format_bytes(bytes),
                percentage: (pct * 10.0).round() / 10.0,
                total_params: params,
            }
        })
        .collect();

    quant_distribution.sort_by(|a, b| b.total_bytes.cmp(&a.total_bytes));

    // Layer grouping
    let mut layer_map: HashMap<u64, Vec<&InspectTensor>> = HashMap::new();
    let mut other_tensors: Vec<InspectTensor> = Vec::new();

    for t in &inspect_tensors {
        if let Some(idx) = extract_layer_index(&t.name) {
            layer_map.entry(idx).or_default().push(t);
        } else {
            other_tensors.push(t.clone());
        }
    }

    let mut layers: Vec<LayerGroup> = layer_map
        .into_iter()
        .map(|(idx, tensors_in_layer)| {
            let mut attention = Vec::new();
            let mut mlp = Vec::new();
            let mut norms = Vec::new();
            let mut other = Vec::new();

            for t in &tensors_in_layer {
                match t.component.as_str() {
                    "attention" => attention.push((*t).clone()),
                    "mlp" => mlp.push((*t).clone()),
                    "norm" => norms.push((*t).clone()),
                    _ => other.push((*t).clone()),
                }
            }

            let total_bytes: u64 = tensors_in_layer.iter().map(|t| t.memory_bytes).sum();

            LayerGroup {
                index: idx,
                total_bytes,
                display: format_bytes(total_bytes),
                attention,
                mlp,
                norms,
                other,
            }
        })
        .collect();

    layers.sort_by_key(|l| l.index);

    // Attention architecture detection
    let attention_info = detect_attention_arch(&inspect_tensors, metadata);

    // Model config and tokenizer info
    let model_config = extract_model_config(metadata);
    let tokenizer_info = extract_tokenizer_info(metadata);

    InspectData {
        memory_breakdown,
        total_memory_bytes: total_memory,
        total_memory_display: format_bytes(total_memory),
        quant_distribution,
        layers,
        other_tensors,
        attention_info,
        model_config,
        tokenizer_info,
        tensor_count: inspect_tensors.len() as u64,
        total_params,
        total_params_display: super::format_param_count(total_params),
    }
}

fn detect_attention_arch(
    tensors: &[InspectTensor],
    metadata: &HashMap<String, String>,
) -> Option<AttentionInfo> {
    // Try metadata first (GGUF stores this explicitly)
    let arch = metadata
        .get("general.architecture")
        .cloned()
        .unwrap_or_default();

    let q_heads_meta = metadata
        .get(&format!("{}.attention.head_count", arch))
        .and_then(|v| v.parse::<u64>().ok());
    let kv_heads_meta = metadata
        .get(&format!("{}.attention.head_count_kv", arch))
        .and_then(|v| v.parse::<u64>().ok());
    let embedding_length = metadata
        .get(&format!("{}.embedding_length", arch))
        .and_then(|v| v.parse::<u64>().ok());

    if let (Some(q_heads), Some(kv_heads)) = (q_heads_meta, kv_heads_meta) {
        let head_dim = embedding_length.map(|e| e / q_heads);
        let (attn_type, gqa_ratio) = if kv_heads == 1 {
            ("MQA".to_string(), Some(q_heads))
        } else if kv_heads == q_heads {
            ("MHA".to_string(), None)
        } else {
            (
                "GQA".to_string(),
                Some(q_heads / kv_heads),
            )
        };
        return Some(AttentionInfo {
            attention_type: attn_type,
            q_heads: Some(q_heads),
            kv_heads: Some(kv_heads),
            head_dim,
            gqa_ratio,
        });
    }

    // Fallback: detect from tensor shapes
    // Find Q and K tensors from first layer
    let q_tensor = tensors.iter().find(|t| {
        let lower = t.name.to_lowercase();
        (lower.contains("blk.0.attn_q") || lower.contains("layers.0.self_attn.q_proj"))
            && lower.contains("weight")
    });
    let k_tensor = tensors.iter().find(|t| {
        let lower = t.name.to_lowercase();
        (lower.contains("blk.0.attn_k") || lower.contains("layers.0.self_attn.k_proj"))
            && lower.contains("weight")
    });

    if let (Some(q), Some(k)) = (q_tensor, k_tensor) {
        if q.shape.len() >= 2 && k.shape.len() >= 2 {
            let q_out = q.shape[0]; // output dim of Q projection
            let k_out = k.shape[0]; // output dim of K projection

            if q_out > 0 && k_out > 0 && q_out >= k_out {
                let ratio = q_out / k_out;
                let (attn_type, gqa_ratio) = if ratio == 1 {
                    ("MHA".to_string(), None)
                } else if k_out == q.shape.get(1).copied().unwrap_or(0) / q_out {
                    ("MQA".to_string(), Some(ratio))
                } else {
                    ("GQA".to_string(), Some(ratio))
                };

                // Estimate head dim from embedding size
                let hidden = q.shape.get(1).copied().unwrap_or(q.shape[0]);
                let estimated_heads = q_out / k_out;
                let head_dim = if estimated_heads > 0 {
                    Some(hidden / (q_out / (hidden / q_out.max(1)).max(1)).max(1))
                } else {
                    None
                };

                return Some(AttentionInfo {
                    attention_type: attn_type,
                    q_heads: None,
                    kv_heads: None,
                    head_dim,
                    gqa_ratio,
                });
            }
        }
    }

    None
}

// Helper to convert &str component to &'static str for HashMap key
fn leak_str(s: &str) -> &'static str {
    match s {
        "embedding" => "embedding",
        "attention" => "attention",
        "mlp" => "mlp",
        "norm" => "norm",
        "output" => "output",
        _ => "other",
    }
}
