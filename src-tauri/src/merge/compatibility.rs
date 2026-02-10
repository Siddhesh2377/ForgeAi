use serde::{Deserialize, Serialize};

use super::registry::ParentRegistry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionMismatch {
    pub dimension_name: String,
    pub values: Vec<(String, u64)>,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionStrategy {
    pub name: String,
    pub description: String,
    pub applicable_to: Vec<String>,
    pub estimated_size_change_bytes: i64,
    pub quality_estimate: String,
    pub requires_training: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatReport {
    pub compatible: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub shared_tensor_count: usize,
    pub total_tensor_count: usize,
    pub architecture_match: bool,
    pub dimension_match: bool,
    pub layer_count_match: bool,
    #[serde(default)]
    pub dimension_details: Vec<DimensionMismatch>,
    #[serde(default)]
    pub resolution_strategies: Vec<ResolutionStrategy>,
}

pub fn check_compatibility(registry: &ParentRegistry) -> CompatReport {
    let parents = registry.all();
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut dimension_details = Vec::new();

    if parents.len() < 2 {
        errors.push("At least 2 parent models required".to_string());
        return CompatReport {
            compatible: false,
            warnings,
            errors,
            shared_tensor_count: 0,
            total_tensor_count: 0,
            architecture_match: false,
            dimension_match: false,
            layer_count_match: false,
            dimension_details: vec![],
            resolution_strategies: vec![],
        };
    }

    // Architecture match
    let architectures: Vec<Option<&String>> = parents
        .iter()
        .map(|p| p.compat.architecture.as_ref())
        .collect();
    let arch_match = {
        let known: Vec<&String> = architectures.iter().filter_map(|a| *a).collect();
        if known.len() < 2 {
            warnings.push("Could not verify architecture compatibility (metadata missing)".to_string());
            true
        } else {
            let first = known[0].to_lowercase();
            let all_same = known.iter().all(|a| a.to_lowercase() == first);
            if !all_same {
                errors.push(format!(
                    "Architecture mismatch: {}",
                    known.iter().map(|a| a.as_str()).collect::<Vec<_>>().join(" vs ")
                ));
            }
            all_same
        }
    };

    // Hidden dimension match
    let hidden_sizes: Vec<Option<u64>> = parents
        .iter()
        .map(|p| p.compat.hidden_size)
        .collect();
    let dim_match = {
        let known: Vec<u64> = hidden_sizes.iter().filter_map(|h| *h).collect();
        if known.len() < 2 {
            warnings.push("Could not verify hidden dimension compatibility".to_string());
            true
        } else {
            let all_same = known.iter().all(|&h| h == known[0]);
            if !all_same {
                errors.push(format!(
                    "Hidden dimension mismatch: {}",
                    known.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(" vs ")
                ));
                dimension_details.push(DimensionMismatch {
                    dimension_name: "hidden_dim".into(),
                    values: parents.iter()
                        .filter_map(|p| p.compat.hidden_size.map(|h| (p.name.clone(), h)))
                        .collect(),
                    severity: "error".into(),
                });
            }
            all_same
        }
    };

    // Layer count
    let layer_counts: Vec<Option<u64>> = parents
        .iter()
        .map(|p| p.compat.num_layers)
        .collect();
    let layer_match = {
        let known: Vec<u64> = layer_counts.iter().filter_map(|l| *l).collect();
        if known.len() < 2 {
            warnings.push("Could not verify layer count compatibility".to_string());
            true
        } else {
            let all_same = known.iter().all(|&l| l == known[0]);
            if !all_same {
                warnings.push(format!(
                    "Layer count differs: {} (Frankenmerge may be required)",
                    known.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" vs ")
                ));
                dimension_details.push(DimensionMismatch {
                    dimension_name: "num_layers".into(),
                    values: parents.iter()
                        .filter_map(|p| p.compat.num_layers.map(|l| (p.name.clone(), l)))
                        .collect(),
                    severity: "warning".into(),
                });
            }
            all_same
        }
    };

    // ── Detailed dimension analysis ─────────────────────
    // Vocab size
    let vocab_sizes: Vec<u64> = parents
        .iter()
        .filter_map(|p| p.compat.vocab_size)
        .collect();
    if vocab_sizes.len() >= 2 && !vocab_sizes.iter().all(|&v| v == vocab_sizes[0]) {
        warnings.push(format!(
            "Vocab size mismatch: {}",
            vocab_sizes.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" vs ")
        ));
        dimension_details.push(DimensionMismatch {
            dimension_name: "vocab_size".into(),
            values: parents.iter()
                .filter_map(|p| p.compat.vocab_size.map(|v| (p.name.clone(), v)))
                .collect(),
            severity: "warning".into(),
        });
    }

    // Attention heads
    let attn_heads: Vec<u64> = parents
        .iter()
        .filter_map(|p| p.compat.num_attention_heads)
        .collect();
    if attn_heads.len() >= 2 && !attn_heads.iter().all(|&h| h == attn_heads[0]) {
        warnings.push(format!(
            "Attention head count differs: {}",
            attn_heads.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(" vs ")
        ));
        dimension_details.push(DimensionMismatch {
            dimension_name: "num_attention_heads".into(),
            values: parents.iter()
                .filter_map(|p| p.compat.num_attention_heads.map(|h| (p.name.clone(), h)))
                .collect(),
            severity: "warning".into(),
        });
    }

    // KV heads
    let kv_heads: Vec<u64> = parents
        .iter()
        .filter_map(|p| p.compat.num_kv_heads)
        .collect();
    if kv_heads.len() >= 2 && !kv_heads.iter().all(|&h| h == kv_heads[0]) {
        warnings.push(format!(
            "KV head count differs: {}",
            kv_heads.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(" vs ")
        ));
        dimension_details.push(DimensionMismatch {
            dimension_name: "num_kv_heads".into(),
            values: parents.iter()
                .filter_map(|p| p.compat.num_kv_heads.map(|h| (p.name.clone(), h)))
                .collect(),
            severity: "warning".into(),
        });
    }

    // Context length
    let ctx_lengths: Vec<u64> = parents
        .iter()
        .filter_map(|p| p.compat.context_length)
        .collect();
    if ctx_lengths.len() >= 2 && !ctx_lengths.iter().all(|&c| c == ctx_lengths[0]) {
        warnings.push(format!(
            "Context length differs: {}",
            ctx_lengths.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" vs ")
        ));
        dimension_details.push(DimensionMismatch {
            dimension_name: "context_length".into(),
            values: parents.iter()
                .filter_map(|p| p.compat.context_length.map(|c| (p.name.clone(), c)))
                .collect(),
            severity: "warning".into(),
        });
    }

    // Tensor name overlap
    let shared = registry.shared_tensor_names();
    let shared_count = shared.len();
    let max_tensor_count = parents
        .iter()
        .map(|p| p.compat.tensor_metas.len())
        .max()
        .unwrap_or(0);

    if max_tensor_count > 0 {
        let overlap_pct = (shared_count as f64 / max_tensor_count as f64) * 100.0;
        if overlap_pct < 50.0 {
            errors.push(format!(
                "Very low tensor name overlap: {:.0}% ({}/{})",
                overlap_pct, shared_count, max_tensor_count
            ));
        } else if overlap_pct < 80.0 {
            warnings.push(format!(
                "Tensor name overlap below 80%: {:.0}% ({}/{})",
                overlap_pct, shared_count, max_tensor_count
            ));
        }
    }

    // Mixed format warning
    let formats: Vec<&str> = parents
        .iter()
        .map(|p| match &p.format {
            crate::model::ModelFormat::SafeTensors => "SafeTensors",
            crate::model::ModelFormat::Gguf => "GGUF",
        })
        .collect();
    let has_mixed = formats.windows(2).any(|w| w[0] != w[1]);
    if has_mixed {
        warnings.push("Mixed formats detected (GGUF tensors will be dequantized to F32)".to_string());
    }

    let compatible = errors.is_empty();

    // Generate resolution strategies for any dimension mismatches
    let resolution_strategies = generate_resolution_strategies(&dimension_details);

    CompatReport {
        compatible,
        warnings,
        errors,
        shared_tensor_count: shared_count,
        total_tensor_count: max_tensor_count,
        architecture_match: arch_match,
        dimension_match: dim_match,
        layer_count_match: layer_match,
        dimension_details,
        resolution_strategies,
    }
}

fn generate_resolution_strategies(mismatches: &[DimensionMismatch]) -> Vec<ResolutionStrategy> {
    let mut strategies = Vec::new();

    for mismatch in mismatches {
        match mismatch.dimension_name.as_str() {
            "hidden_dim" => {
                strategies.push(ResolutionStrategy {
                    name: "zero_padding".into(),
                    description: "Pad smaller model's hidden dimension with zeros to match larger model".into(),
                    applicable_to: vec!["hidden_dim".into()],
                    estimated_size_change_bytes: 0, // depends on actual sizes
                    quality_estimate: "medium".into(),
                    requires_training: false,
                });
                strategies.push(ResolutionStrategy {
                    name: "interpolation".into(),
                    description: "Interpolate tensor dimensions to match target size".into(),
                    applicable_to: vec!["hidden_dim".into()],
                    estimated_size_change_bytes: 0,
                    quality_estimate: "medium".into(),
                    requires_training: false,
                });
                strategies.push(ResolutionStrategy {
                    name: "truncation".into(),
                    description: "Truncate larger model's dimensions to match smaller model".into(),
                    applicable_to: vec!["hidden_dim".into()],
                    estimated_size_change_bytes: 0,
                    quality_estimate: "low".into(),
                    requires_training: false,
                });
            }
            "vocab_size" => {
                strategies.push(ResolutionStrategy {
                    name: "zero_padding".into(),
                    description: "Extend embedding matrix with zero rows for missing tokens".into(),
                    applicable_to: vec!["vocab_size".into()],
                    estimated_size_change_bytes: 0,
                    quality_estimate: "high".into(),
                    requires_training: false,
                });
                strategies.push(ResolutionStrategy {
                    name: "truncation".into(),
                    description: "Truncate vocabulary to smaller model's size".into(),
                    applicable_to: vec!["vocab_size".into()],
                    estimated_size_change_bytes: 0,
                    quality_estimate: "medium".into(),
                    requires_training: false,
                });
            }
            _ => {}
        }
    }

    // Universal fallback for any mismatch
    if !mismatches.is_empty() {
        let all_dims: Vec<String> = mismatches.iter().map(|m| m.dimension_name.clone()).collect();
        strategies.push(ResolutionStrategy {
            name: "moe_routing".into(),
            description: "Convert to Mixture-of-Experts: each parent becomes an expert, router handles dimension differences".into(),
            applicable_to: all_dims,
            estimated_size_change_bytes: 0,
            quality_estimate: "high".into(),
            requires_training: false,
        });
    }

    strategies
}
