
#[derive(Debug, Clone)]
pub struct LayerSpecialization {
    pub category: String,
    pub primary_function: String,
    pub score: f64,
}

/// Classify what a layer specializes in based on its position and composition.
///
/// Early layers (0-20%): syntactic processing, token embedding refinement
/// Middle layers (20-70%): semantic understanding, knowledge retrieval
/// Late layers (70-100%): reasoning, output preparation, task-specific processing
pub fn classify_layer_specialization(
    layer_idx: u64,
    total_layers: u64,
    tensor_names: &[&String],
) -> LayerSpecialization {
    let position_ratio = if total_layers > 0 {
        layer_idx as f64 / total_layers as f64
    } else {
        0.5
    };

    let _has_attn = tensor_names.iter().any(|n| {
        let lower = n.to_lowercase();
        lower.contains("attn") || lower.contains("self_attn") || lower.contains("q_proj")
    });
    let _has_mlp = tensor_names.iter().any(|n| {
        let lower = n.to_lowercase();
        lower.contains("mlp") || lower.contains("ffn") || lower.contains("gate_proj")
    });

    if position_ratio < 0.15 {
        LayerSpecialization {
            category: "syntactic".to_string(),
            primary_function: "Token embedding / positional encoding".to_string(),
            score: 0.8 + (1.0 - position_ratio / 0.15) * 0.2,
        }
    } else if position_ratio < 0.35 {
        LayerSpecialization {
            category: "syntactic".to_string(),
            primary_function: "Syntactic parsing / POS tagging".to_string(),
            score: 0.7,
        }
    } else if position_ratio < 0.65 {
        LayerSpecialization {
            category: "semantic".to_string(),
            primary_function: "Knowledge retrieval / semantic understanding".to_string(),
            score: 0.75,
        }
    } else if position_ratio < 0.85 {
        LayerSpecialization {
            category: "reasoning".to_string(),
            primary_function: "Multi-step reasoning / inference".to_string(),
            score: 0.8,
        }
    } else {
        LayerSpecialization {
            category: "reasoning".to_string(),
            primary_function: "Output preparation / task specialization".to_string(),
            score: 0.85,
        }
    }
}

/// Estimate entropy for a layer (lower = more specialized).
pub fn estimate_layer_entropy(layer_idx: u64, total_layers: u64) -> f64 {
    let position = if total_layers > 0 {
        layer_idx as f64 / total_layers as f64
    } else {
        0.5
    };

    // Entropy tends to be higher in early and middle layers, lower in late layers
    let base = 4.0;
    if position < 0.2 {
        base * 0.7 // Early: moderate
    } else if position < 0.6 {
        base * 0.85 // Middle: higher
    } else if position < 0.85 {
        base * 0.6 // Late: lower (more specialized)
    } else {
        base * 0.5 // Final: most specialized
    }
}

/// Estimate confidence for a layer's profiling results.
pub fn estimate_layer_confidence(layer_idx: u64, total_layers: u64) -> f64 {
    let position = if total_layers > 0 {
        layer_idx as f64 / total_layers as f64
    } else {
        0.5
    };

    // Confidence in classification increases toward the end
    0.6 + position * 0.3
}
