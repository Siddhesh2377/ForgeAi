use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::merge::registry::ParentModel;
use crate::merge::tensor_io;
use crate::model::error::ModelError;
use crate::model::inspect;

// ── Layer Categories ────────────────────────────────────

/// All possible layer categories with display metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCategory {
    pub id: String,
    pub label: String,
    pub color: String,
    pub description: String,
}

/// The 8 categories a layer can be classified into.
pub fn all_categories() -> Vec<LayerCategory> {
    vec![
        LayerCategory {
            id: "embed".into(),
            label: "EMBED".into(),
            color: "#a78bfa".into(),
            description: "Token embedding & positional encoding".into(),
        },
        LayerCategory {
            id: "syntax".into(),
            label: "SYNTAX".into(),
            color: "#60a5fa".into(),
            description: "Syntactic parsing & grammar structure".into(),
        },
        LayerCategory {
            id: "language".into(),
            label: "LANG".into(),
            color: "#38bdf8".into(),
            description: "Language understanding & context".into(),
        },
        LayerCategory {
            id: "knowledge".into(),
            label: "KNOW".into(),
            color: "#34d399".into(),
            description: "Factual knowledge storage & retrieval".into(),
        },
        LayerCategory {
            id: "reasoning".into(),
            label: "REASON".into(),
            color: "#fbbf24".into(),
            description: "Multi-step reasoning & inference".into(),
        },
        LayerCategory {
            id: "expert".into(),
            label: "EXPERT".into(),
            color: "#fb923c".into(),
            description: "Specialized task processing (math/code)".into(),
        },
        LayerCategory {
            id: "synthesis".into(),
            label: "SYNTH".into(),
            color: "#f87171".into(),
            description: "Output synthesis & generation".into(),
        },
        LayerCategory {
            id: "head".into(),
            label: "HEAD".into(),
            color: "#94a3b8".into(),
            description: "Output head & final projection".into(),
        },
    ]
}

fn get_category(id: &str) -> LayerCategory {
    all_categories()
        .into_iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| LayerCategory {
            id: "unknown".into(),
            label: "UNK".into(),
            color: "#6b7280".into(),
            description: "Unknown layer function".into(),
        })
}

// ── Layer Analysis Result ───────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAnalysis {
    pub layer_index: u64,
    pub category: String,
    pub label: String,
    pub color: String,
    pub description: String,
    pub confidence: f64,
    pub attn_tensors: usize,
    pub mlp_tensors: usize,
    pub norm_tensors: usize,
    pub norm_l2: f64,
    pub norm_variance: f64,
    pub mlp_dominance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub parent_id: String,
    pub parent_name: String,
    pub layers: Vec<LayerAnalysis>,
    pub total_layers: u64,
    pub categories: Vec<LayerCategory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisProgress {
    pub parent_id: String,
    pub layer_index: u64,
    pub total_layers: u64,
    pub percent: f64,
    pub stage: String,
}

// ── Tensor Statistics ───────────────────────────────────

struct NormStats {
    l2: f64,
    mean: f64,
    variance: f64,
}

/// Compute statistics from a 1D tensor's values.
fn compute_stats(data: &[f32]) -> NormStats {
    if data.is_empty() {
        return NormStats { l2: 0.0, mean: 0.0, variance: 0.0 };
    }
    let n = data.len() as f64;
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / n;
    let variance = data.iter().map(|&x| {
        let d = x as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    let l2 = data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    NormStats { l2, mean, variance }
}

/// Check if a tensor name is a norm/layernorm tensor.
fn is_norm_tensor(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("norm") || lower.contains("layernorm") || lower.contains("ln_")
        || lower.contains("rms_norm") || lower.contains("input_layernorm")
        || lower.contains("post_attention_layernorm")
}

// ── Analysis Engine ─────────────────────────────────────

/// Analyze all layers of a parent model by loading norm tensors and computing stats.
pub fn analyze_parent(
    app: &AppHandle,
    parent: &ParentModel,
    cancel: Arc<AtomicBool>,
) -> Result<AnalysisResult, ModelError> {
    let total_layers = parent.layer_count.unwrap_or(0);
    if total_layers == 0 {
        return Ok(AnalysisResult {
            parent_id: parent.id.clone(),
            parent_name: parent.name.clone(),
            layers: vec![],
            total_layers: 0,
            categories: all_categories(),
        });
    }

    // Phase 1: Collect raw stats for all layers
    let mut raw_stats: Vec<(u64, f64, f64, usize, usize, usize)> = Vec::new();

    for layer_idx in 0..total_layers {
        if cancel.load(Ordering::Relaxed) {
            return Err(ModelError::MergeCancelled);
        }

        let percent = ((layer_idx as f64 + 0.5) / total_layers as f64) * 50.0;
        let _ = app.emit("merge:analysis-progress", AnalysisProgress {
            parent_id: parent.id.clone(),
            layer_index: layer_idx,
            total_layers,
            percent,
            stage: "Loading tensors".into(),
        });

        // Find tensors for this layer
        let layer_prefix1 = format!("blk.{}", layer_idx);
        let layer_prefix2 = format!("layers.{}", layer_idx);

        let mut attn_count = 0usize;
        let mut mlp_count = 0usize;
        let mut norm_names = Vec::new();

        for name in &parent.compat.tensor_names {
            if !name.contains(&layer_prefix1) && !name.contains(&layer_prefix2) {
                continue;
            }
            let class = inspect::classify_tensor(name);
            match class {
                "attention" => attn_count += 1,
                "mlp" => mlp_count += 1,
                "norm" => {
                    if is_norm_tensor(name) {
                        norm_names.push(name.clone());
                    }
                }
                _ => {}
            }
        }

        // Load norm tensors and compute statistics
        let mut layer_l2 = 0.0f64;
        let mut layer_var = 0.0f64;
        let mut loaded_count = 0usize;

        for norm_name in &norm_names {
            match tensor_io::load_tensor(parent, norm_name) {
                Ok(tensor) => {
                    // Flatten to 1D and convert to f32
                    if let Ok(flat) = tensor.flatten_all() {
                        if let Ok(data) = flat.to_vec1::<f32>() {
                            let stats = compute_stats(&data);
                            layer_l2 += stats.l2;
                            layer_var += stats.variance;
                            loaded_count += 1;
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        // Average the stats if we loaded multiple norm tensors
        if loaded_count > 0 {
            layer_l2 /= loaded_count as f64;
            layer_var /= loaded_count as f64;
        }

        raw_stats.push((layer_idx, layer_l2, layer_var, attn_count, mlp_count, norm_names.len()));
    }

    // Phase 2: Normalize stats across all layers
    let all_l2: Vec<f64> = raw_stats.iter().map(|s| s.1).collect();
    let all_var: Vec<f64> = raw_stats.iter().map(|s| s.2).collect();

    let l2_mean = all_l2.iter().sum::<f64>() / all_l2.len().max(1) as f64;
    let l2_std = (all_l2.iter().map(|x| (x - l2_mean).powi(2)).sum::<f64>()
        / all_l2.len().max(1) as f64).sqrt().max(0.001);

    let var_mean = all_var.iter().sum::<f64>() / all_var.len().max(1) as f64;
    let var_std = (all_var.iter().map(|x| (x - var_mean).powi(2)).sum::<f64>()
        / all_var.len().max(1) as f64).sqrt().max(0.001);

    // Phase 3: Classify each layer
    let mut layers = Vec::new();

    for (layer_idx, l2, var, attn_count, mlp_count, norm_count) in &raw_stats {
        if cancel.load(Ordering::Relaxed) {
            return Err(ModelError::MergeCancelled);
        }

        let percent = 50.0 + ((*layer_idx as f64 + 1.0) / total_layers as f64) * 50.0;
        let _ = app.emit("merge:analysis-progress", AnalysisProgress {
            parent_id: parent.id.clone(),
            layer_index: *layer_idx,
            total_layers,
            percent,
            stage: "Classifying".into(),
        });

        let position = *layer_idx as f64 / total_layers as f64;
        let l2_z = (l2 - l2_mean) / l2_std;
        let var_z = (var - var_mean) / var_std;

        // MLP dominance: ratio of MLP tensors to total
        let total_tensors = (*attn_count + *mlp_count).max(1) as f64;
        let mlp_dominance = *mlp_count as f64 / total_tensors;

        // Classification using position + normalized tensor stats
        let (cat_id, confidence) = classify_layer(position, l2_z, var_z, mlp_dominance);
        let cat = get_category(cat_id);

        let analysis = LayerAnalysis {
            layer_index: *layer_idx,
            category: cat.id.clone(),
            label: cat.label.clone(),
            color: cat.color.clone(),
            description: cat.description.clone(),
            confidence,
            attn_tensors: *attn_count,
            mlp_tensors: *mlp_count,
            norm_tensors: *norm_count,
            norm_l2: *l2,
            norm_variance: *var,
            mlp_dominance,
        };

        let _ = app.emit("merge:layer-analyzed", &analysis);
        layers.push(analysis);
    }

    Ok(AnalysisResult {
        parent_id: parent.id.clone(),
        parent_name: parent.name.clone(),
        layers,
        total_layers,
        categories: all_categories(),
    })
}

/// Classify a single layer based on position and normalized statistics.
///
/// Position is the primary signal (well-established from research).
/// Tensor stats refine the boundaries: layers with high MLP norm L2
/// indicate knowledge storage; high variance indicates specialization;
/// high MLP dominance in late layers suggests expert/task processing.
fn classify_layer(position: f64, l2_z: f64, var_z: f64, mlp_dom: f64) -> (&'static str, f64) {
    // Very early: embedding
    if position < 0.05 {
        return ("embed", 0.95);
    }

    // Early: syntax vs language
    if position < 0.20 {
        if l2_z > 0.5 || var_z > 0.5 {
            // High norm activity → active syntactic processing
            return ("syntax", 0.80 + var_z.abs().min(0.15));
        }
        return ("syntax", 0.75);
    }

    // Early-mid: language understanding
    if position < 0.35 {
        if mlp_dom > 0.55 && l2_z > 0.3 {
            // MLP-heavy early-mid → knowledge emerging
            return ("knowledge", 0.70);
        }
        return ("language", 0.75 + (position - 0.20) / 0.15 * 0.1);
    }

    // Mid: knowledge vs context
    if position < 0.55 {
        if mlp_dom > 0.5 && l2_z > 0.0 {
            // High MLP dominance + high norms → factual knowledge storage
            return ("knowledge", 0.80 + l2_z.min(0.15));
        }
        if var_z > 0.5 {
            // High variance → beginning of reasoning
            return ("reasoning", 0.65);
        }
        return ("language", 0.70);
    }

    // Mid-late: reasoning
    if position < 0.72 {
        if var_z > 0.8 || l2_z > 1.0 {
            // Very high activity → expert processing emerging
            return ("expert", 0.70);
        }
        return ("reasoning", 0.80 + (position - 0.55) / 0.17 * 0.1);
    }

    // Late: expert vs synthesis
    if position < 0.88 {
        if l2_z > 0.5 && var_z > 0.3 {
            // High activity + high variance → specialized expert
            return ("expert", 0.80 + l2_z.min(0.15));
        }
        if mlp_dom > 0.55 {
            // MLP-heavy late → task-specific computation
            return ("expert", 0.75);
        }
        return ("synthesis", 0.75);
    }

    // Final: synthesis / head
    if position < 0.97 {
        return ("synthesis", 0.85);
    }

    ("head", 0.90)
}
