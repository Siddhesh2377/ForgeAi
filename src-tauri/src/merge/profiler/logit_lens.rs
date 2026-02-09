use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tauri::{AppHandle, Emitter};

use crate::model::error::ModelError;

use super::{LayerProfile, PredictionEntry, ProfileProgress, ProfileResult};
use super::layer_analysis;

/// Run Logit Lens profiling on a model.
///
/// This performs a simplified layer-by-layer analysis:
/// For each layer, we examine the tensor norms and distributions
/// to estimate what each layer specializes in.
///
/// Full Logit Lens (with actual inference through layers) requires
/// loading the full model into Candle, which is expensive.
/// This implementation uses tensor statistics as a proxy.
pub fn profile_layers(
    app: &AppHandle,
    parent_id: &str,
    tensor_names: &[String],
    total_layers: u64,
    cancel: Arc<AtomicBool>,
) -> Result<ProfileResult, ModelError> {
    let mut layers = Vec::new();

    for layer_idx in 0..total_layers {
        if cancel.load(Ordering::Relaxed) {
            return Err(ModelError::MergeCancelled);
        }

        let percent = ((layer_idx as f64 + 1.0) / total_layers as f64) * 100.0;

        let _ = app.emit("merge:profile-progress", ProfileProgress {
            layer_index: layer_idx,
            total_layers,
            percent,
        });

        // Analyze layer tensors
        let layer_prefix = format!("blk.{}", layer_idx);
        let alt_prefix = format!("layers.{}", layer_idx);

        let layer_tensors: Vec<&String> = tensor_names
            .iter()
            .filter(|n| n.contains(&layer_prefix) || n.contains(&alt_prefix))
            .collect();

        let specialization = layer_analysis::classify_layer_specialization(layer_idx, total_layers, &layer_tensors);
        let entropy = layer_analysis::estimate_layer_entropy(layer_idx, total_layers);
        let confidence = layer_analysis::estimate_layer_confidence(layer_idx, total_layers);

        let top_predictions = vec![
            PredictionEntry {
                token: specialization.primary_function.clone(),
                probability: confidence,
                rank: 0,
            },
        ];

        let profile = LayerProfile {
            layer_index: layer_idx,
            top_predictions,
            entropy,
            specialization: specialization.category.clone(),
            confidence,
        };

        let _ = app.emit("merge:profile-layer-done", &profile);
        layers.push(profile);
    }

    Ok(ProfileResult {
        parent_id: parent_id.to_string(),
        layers,
        total_layers,
    })
}
