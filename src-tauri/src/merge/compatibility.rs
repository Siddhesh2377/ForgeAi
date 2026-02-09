use serde::{Deserialize, Serialize};

use super::registry::ParentRegistry;

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
}

pub fn check_compatibility(registry: &ParentRegistry) -> CompatReport {
    let parents = registry.all();
    let mut warnings = Vec::new();
    let mut errors = Vec::new();

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
            }
            all_same
        }
    };

    // Tensor name overlap
    let shared = registry.shared_tensor_names();
    let shared_count = shared.len();
    let max_tensor_count = parents
        .iter()
        .map(|p| p.compat.tensor_names.len())
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

    // Vocab size warning
    let vocab_sizes: Vec<u64> = parents
        .iter()
        .filter_map(|p| p.compat.vocab_size)
        .collect();
    if vocab_sizes.len() >= 2 && !vocab_sizes.iter().all(|&v| v == vocab_sizes[0]) {
        warnings.push(format!(
            "Vocab size mismatch: {}",
            vocab_sizes.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" vs ")
        ));
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

    CompatReport {
        compatible,
        warnings,
        errors,
        shared_tensor_count: shared_count,
        total_tensor_count: max_tensor_count,
        architecture_match: arch_match,
        dimension_match: dim_match,
        layer_count_match: layer_match,
    }
}
