pub mod layer_analysis;
pub mod logit_lens;
pub mod tensor_analysis;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProfile {
    pub layer_index: u64,
    pub top_predictions: Vec<PredictionEntry>,
    pub entropy: f64,
    pub specialization: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionEntry {
    pub token: String,
    pub probability: f64,
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    pub parent_id: String,
    pub layers: Vec<LayerProfile>,
    pub total_layers: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileProgress {
    pub layer_index: u64,
    pub total_layers: u64,
    pub percent: f64,
}
