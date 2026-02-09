use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use super::error::ModelError;
use super::{
    derive_layer_count, format_file_size, format_param_count, ModelFormat, ModelInfo, TensorInfo,
};

/// Parse a single safetensors file header, returning tensors and metadata.
fn parse_header(path: &Path) -> Result<(Vec<TensorInfo>, HashMap<String, String>, u64), ModelError> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();

    if file_size < 8 {
        return Err(ModelError::FileTooSmall(file_size));
    }

    let mmap = unsafe { Mmap::map(&file)? };

    let header_len = u64::from_le_bytes(
        mmap[0..8]
            .try_into()
            .map_err(|_| ModelError::ParseError {
                format: "SafeTensors".into(),
                reason: "Failed to read header length".into(),
            })?,
    ) as usize;

    if header_len == 0 || header_len + 8 > file_size as usize {
        return Err(ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: format!("Header length {} is invalid for file size {}", header_len, file_size),
        });
    }

    let header_json: serde_json::Value =
        serde_json::from_slice(&mmap[8..8 + header_len]).map_err(|e| ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: format!("Invalid JSON header: {}", e),
        })?;

    let header_map = header_json
        .as_object()
        .ok_or_else(|| ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: "Header is not a JSON object".into(),
        })?;

    let mut metadata = HashMap::new();
    let mut tensors = Vec::new();

    for (key, value) in header_map {
        if key == "__metadata__" {
            if let Some(meta_obj) = value.as_object() {
                for (mk, mv) in meta_obj {
                    metadata.insert(mk.clone(), mv.as_str().unwrap_or("").to_string());
                }
            }
            continue;
        }

        if let Some(obj) = value.as_object() {
            let dtype = obj
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let shape: Vec<u64> = obj
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default();

            tensors.push(TensorInfo {
                name: key.clone(),
                dtype,
                shape,
            });
        }
    }

    Ok((tensors, metadata, file_size))
}

/// Parse a directory of safetensors shards + config.json + tokenizer files.
pub fn parse_dir(dir: &Path) -> Result<ModelInfo, ModelError> {
    let mut shard_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: format!("Cannot read directory: {}", e),
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase() == "safetensors")
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();

    if shard_files.is_empty() {
        return Err(ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: "No .safetensors files found in directory".into(),
        });
    }

    shard_files.sort();
    let shard_count = shard_files.len() as u32;

    let mut all_tensors: Vec<TensorInfo> = Vec::new();
    let mut all_metadata: HashMap<String, String> = HashMap::new();
    let mut total_file_size: u64 = 0;
    let mut total_params: u64 = 0;

    for shard_path in &shard_files {
        let (tensors, metadata, file_size) = parse_header(shard_path)?;

        for t in &tensors {
            let param_count: u64 = if t.shape.is_empty() { 0 } else { t.shape.iter().product() };
            total_params += param_count;
        }

        all_tensors.extend(tensors);
        for (k, v) in metadata {
            all_metadata.entry(k).or_insert(v);
        }
        total_file_size += file_size;
    }

    let tensor_count = all_tensors.len() as u64;
    all_tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let layer_count = derive_layer_count(&all_tensors);
    let quantization = detect_quantization(&all_tensors);

    // Read config.json if present
    let config_path = dir.join("config.json");
    let (has_config, model_type, architecture, context_length, embedding_size, vocab_size) =
        if config_path.exists() {
            parse_config_json(&config_path)
        } else {
            (false, None, None, None, None, None)
        };

    // Check tokenizer files
    let has_tokenizer = dir.join("tokenizer.json").exists()
        || dir.join("tokenizer_config.json").exists()
        || dir.join("tokenizer.model").exists();

    let dir_name = dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let arch = architecture
        .clone()
        .or_else(|| all_metadata.get("model_type").cloned())
        .or_else(|| all_metadata.get("architecture").cloned());

    Ok(ModelInfo {
        file_name: dir_name,
        file_path: dir.to_string_lossy().to_string(),
        file_size: total_file_size,
        file_size_display: format_file_size(total_file_size),
        format: ModelFormat::SafeTensors,
        tensor_count,
        parameter_count: total_params,
        parameter_count_display: format_param_count(total_params),
        layer_count,
        quantization,
        architecture: arch,
        context_length,
        embedding_size,
        metadata: all_metadata,
        tensor_preview: all_tensors.iter().take(50).cloned().collect(),
        all_tensors,
        shard_count: Some(shard_count),
        has_tokenizer: Some(has_tokenizer),
        has_config: Some(has_config),
        model_type,
        vocab_size,
    })
}

fn parse_config_json(
    path: &Path,
) -> (bool, Option<String>, Option<String>, Option<u64>, Option<u64>, Option<u64>) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return (false, None, None, None, None, None),
    };
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return (false, None, None, None, None, None),
    };

    let model_type = json.get("model_type").and_then(|v| v.as_str()).map(|s| s.to_string());

    let architecture = json
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| model_type.clone());

    let context_length = json
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64());

    let embedding_size = json.get("hidden_size").and_then(|v| v.as_u64());

    let vocab_size = json.get("vocab_size").and_then(|v| v.as_u64());

    (true, model_type, architecture, context_length, embedding_size, vocab_size)
}

pub fn parse(path: &Path) -> Result<ModelInfo, ModelError> {
    let (mut tensors, metadata, file_size) = parse_header(path)?;

    let mut total_params: u64 = 0;
    for t in &tensors {
        let param_count: u64 = if t.shape.is_empty() { 0 } else { t.shape.iter().product() };
        total_params += param_count;
    }

    let tensor_count = tensors.len() as u64;

    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let layer_count = derive_layer_count(&tensors);
    let quantization = detect_quantization(&tensors);

    Ok(ModelInfo {
        file_name: path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        file_path: path.to_string_lossy().to_string(),
        file_size,
        file_size_display: format_file_size(file_size),
        format: ModelFormat::SafeTensors,
        tensor_count,
        parameter_count: total_params,
        parameter_count_display: format_param_count(total_params),
        layer_count,
        quantization,
        architecture: metadata
            .get("model_type")
            .cloned()
            .or_else(|| metadata.get("architecture").cloned()),
        context_length: None,
        embedding_size: None,
        metadata,
        tensor_preview: tensors.iter().take(50).cloned().collect(),
        all_tensors: tensors,
        shard_count: None,
        has_tokenizer: None,
        has_config: None,
        model_type: None,
        vocab_size: None,
    })
}

fn detect_quantization(tensors: &[TensorInfo]) -> Option<String> {
    if tensors.is_empty() {
        return None;
    }

    let mut dtype_counts: HashMap<&str, usize> = HashMap::new();
    for t in tensors {
        *dtype_counts.entry(&t.dtype).or_insert(0) += 1;
    }

    // Find the most common dtype (excluding small tensors like norms)
    let dominant = dtype_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(dtype, _)| *dtype)?;

    Some(dominant.to_string())
}
