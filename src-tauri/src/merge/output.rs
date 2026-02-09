use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use candle_core::{DType, Tensor};

use crate::merge::registry::CompatInfo;
use crate::model::error::ModelError;

/// Write merged tensors to a SafeTensors file.
pub fn write_safetensors(
    output_path: &str,
    tensors: &[(String, Tensor)],
) -> Result<(), ModelError> {
    let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());

    // Build the safetensors data structure
    // Header format: { "tensor_name": { "dtype": "F32", "shape": [...], "data_offsets": [start, end] }, ... }
    let mut tensor_data: Vec<(&str, Vec<u8>, &str, Vec<usize>)> = Vec::new();
    let _current_offset: usize = 0;

    for (name, tensor) in tensors {
        let tensor_f32 = tensor.to_dtype(DType::F32).map_err(map_err)?;
        let flat: Vec<f32> = tensor_f32.flatten_all().map_err(map_err)?
            .to_vec1::<f32>().map_err(map_err)?;

        let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        let shape: Vec<usize> = tensor.shape().dims().to_vec();
        let dtype = "F32";

        tensor_data.push((name.as_str(), bytes, dtype, shape));
    }

    // Build header JSON
    let mut header_entries: Vec<String> = Vec::new();
    let mut data_offset = 0usize;

    for (name, bytes, dtype, shape) in &tensor_data {
        let end_offset = data_offset + bytes.len();
        let shape_str = shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");

        header_entries.push(format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name, dtype, shape_str, data_offset, end_offset
        ));

        data_offset = end_offset;
    }

    // Add metadata
    header_entries.push(
        "\"__metadata__\":{\"format\":\"pt\",\"source\":\"forgeai-merge\"}".to_string()
    );

    let header_json = format!("{{{}}}", header_entries.join(","));
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    // Write file
    let file = File::create(output_path).map_err(ModelError::IoError)?;
    let mut writer = BufWriter::new(file);

    // Write header length (8 bytes, little-endian)
    writer.write_all(&header_len.to_le_bytes()).map_err(ModelError::IoError)?;

    // Write header JSON
    writer.write_all(header_bytes).map_err(ModelError::IoError)?;

    // Write tensor data
    for (_, bytes, _, _) in &tensor_data {
        writer.write_all(bytes).map_err(ModelError::IoError)?;
    }

    writer.flush().map_err(ModelError::IoError)?;

    Ok(())
}

/// Write merged tensors to a GGUF file (F32 unquantized).
///
/// Metadata source priority:
/// 1. `source_gguf_path` — copies raw metadata from an existing GGUF file
/// 2. `compat` + config.json from parent directory — constructs metadata from
///    HuggingFace config fields (handles safetensors → GGUF conversion)
/// 3. Minimal fallback with just architecture/name/file_type
pub fn write_gguf(
    output_path: &str,
    tensors: &[(String, Tensor)],
    model_name: &str,
    source_gguf_path: Option<&str>,
    compat: Option<&CompatInfo>,
    config_json_dir: Option<&str>,
) -> Result<(), ModelError> {
    let map_err = |e: candle_core::Error| ModelError::CandleError(e.to_string());

    // If we have a source GGUF, extract its raw metadata section
    let source_metadata = source_gguf_path.and_then(|path| {
        extract_gguf_metadata(path).ok()
    });

    let file = File::create(output_path).map_err(ModelError::IoError)?;
    let mut writer = BufWriter::new(file);

    // Track bytes written for alignment
    let mut bytes_written: usize = 0;

    // GGUF Magic
    writer.write_all(b"GGUF").map_err(ModelError::IoError)?;
    bytes_written += 4;

    // Version 3
    writer.write_all(&3u32.to_le_bytes()).map_err(ModelError::IoError)?;
    bytes_written += 4;

    // Tensor count
    writer.write_all(&(tensors.len() as u64).to_le_bytes()).map_err(ModelError::IoError)?;
    bytes_written += 8;

    if let Some(ref meta) = source_metadata {
        // Write original metadata KV count + raw bytes
        writer.write_all(&(meta.kv_count as u64).to_le_bytes()).map_err(ModelError::IoError)?;
        bytes_written += 8;
        writer.write_all(&meta.raw_kv_bytes).map_err(ModelError::IoError)?;
        bytes_written += meta.raw_kv_bytes.len();
    } else {
        // Build metadata from CompatInfo + config.json
        let metadata_kvs = build_gguf_metadata(model_name, compat, config_json_dir);

        writer.write_all(&(metadata_kvs.len() as u64).to_le_bytes()).map_err(ModelError::IoError)?;
        bytes_written += 8;

        for (key, value) in &metadata_kvs {
            bytes_written += write_gguf_string(&mut writer, key)?;
            bytes_written += write_gguf_value(&mut writer, value)?;
        }
    }

    // Compute tensor data offsets
    let mut tensor_infos: Vec<(String, Vec<usize>, u64)> = Vec::new();
    let mut offset: u64 = 0;

    for (name, tensor) in tensors {
        let shape: Vec<usize> = tensor.shape().dims().to_vec();
        let elem_count: usize = shape.iter().product();
        let byte_size = (elem_count * 4) as u64; // F32

        tensor_infos.push((name.clone(), shape, offset));
        offset += byte_size;
    }

    // Write tensor info entries
    for (name, shape, data_offset) in &tensor_infos {
        bytes_written += write_gguf_string(&mut writer, name)?;
        writer.write_all(&(shape.len() as u32).to_le_bytes()).map_err(ModelError::IoError)?;
        bytes_written += 4;
        for &dim in shape {
            writer.write_all(&(dim as u64).to_le_bytes()).map_err(ModelError::IoError)?;
            bytes_written += 8;
        }
        writer.write_all(&0u32.to_le_bytes()).map_err(ModelError::IoError)?; // ggml_type F32
        bytes_written += 4;
        writer.write_all(&data_offset.to_le_bytes()).map_err(ModelError::IoError)?;
        bytes_written += 8;
    }

    // Align to 32 bytes
    let alignment = 32;
    let padding = (alignment - (bytes_written % alignment)) % alignment;
    for _ in 0..padding {
        writer.write_all(&[0u8]).map_err(ModelError::IoError)?;
    }

    // Write tensor data
    for (_, tensor) in tensors {
        let tensor_f32 = tensor.to_dtype(DType::F32).map_err(map_err)?;
        let flat: Vec<f32> = tensor_f32.flatten_all().map_err(map_err)?
            .to_vec1::<f32>().map_err(map_err)?;

        let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer.write_all(&bytes).map_err(ModelError::IoError)?;
    }

    writer.flush().map_err(ModelError::IoError)?;

    Ok(())
}

struct GgufSourceMetadata {
    kv_count: usize,
    raw_kv_bytes: Vec<u8>,
}

/// Extract raw metadata KV bytes from a GGUF file.
fn extract_gguf_metadata(path: &str) -> Result<GgufSourceMetadata, ModelError> {
    use memmap2::Mmap;

    let file = File::open(path).map_err(ModelError::IoError)?;
    let mmap = unsafe { Mmap::map(&file).map_err(ModelError::IoError)? };

    if mmap.len() < 24 || &mmap[0..4] != b"GGUF" {
        return Err(ModelError::ParseError {
            format: "GGUF".into(),
            reason: "Invalid GGUF magic".into(),
        });
    }

    let _version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
    let _tensor_count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
    let kv_count = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;

    // Parse through metadata KVs to find where they end
    let mut pos = 24usize;

    for _ in 0..kv_count {
        // Read key string
        if pos + 8 > mmap.len() { break; }
        let key_len = u64::from_le_bytes(mmap[pos..pos+8].try_into().unwrap()) as usize;
        pos += 8 + key_len;

        // Read value type
        if pos + 4 > mmap.len() { break; }
        let vtype = u32::from_le_bytes(mmap[pos..pos+4].try_into().unwrap());
        pos += 4;

        // Skip value based on type
        pos = skip_gguf_value(&mmap, pos, vtype)?;
    }

    // The raw KV bytes are from position 24 to pos
    let raw_kv_bytes = mmap[24..pos].to_vec();

    Ok(GgufSourceMetadata {
        kv_count,
        raw_kv_bytes,
    })
}

/// Skip a GGUF metadata value and return the new position.
fn skip_gguf_value(data: &[u8], mut pos: usize, vtype: u32) -> Result<usize, ModelError> {
    match vtype {
        0 => { pos += 1; }  // UINT8
        1 => { pos += 1; }  // INT8
        2 => { pos += 2; }  // UINT16
        3 => { pos += 2; }  // INT16
        4 => { pos += 4; }  // UINT32
        5 => { pos += 4; }  // INT32
        6 => { pos += 4; }  // FLOAT32
        7 => { pos += 1; }  // BOOL
        8 => {              // STRING
            if pos + 8 > data.len() { return Ok(pos); }
            let len = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()) as usize;
            pos += 8 + len;
        }
        9 => {              // ARRAY
            if pos + 12 > data.len() { return Ok(pos); }
            let arr_type = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            let arr_len = u64::from_le_bytes(data[pos+4..pos+12].try_into().unwrap()) as usize;
            pos += 12;
            for _ in 0..arr_len {
                pos = skip_gguf_value(data, pos, arr_type)?;
            }
        }
        10 => { pos += 8; } // UINT64
        11 => { pos += 8; } // INT64
        12 => { pos += 8; } // FLOAT64
        _ => { pos += 4; }  // Unknown, skip 4
    }
    Ok(pos)
}

enum GgufMetaValue {
    String(String),
    U32(u32),
    F32(f32),
    StringArray(Vec<String>),
    F32Array(Vec<f32>),
    I32Array(Vec<i32>),
}

fn write_gguf_string<W: Write>(writer: &mut W, s: &str) -> Result<usize, ModelError> {
    let bytes = s.as_bytes();
    writer.write_all(&(bytes.len() as u64).to_le_bytes()).map_err(ModelError::IoError)?;
    writer.write_all(bytes).map_err(ModelError::IoError)?;
    Ok(8 + bytes.len())
}

fn write_gguf_value<W: Write>(writer: &mut W, value: &GgufMetaValue) -> Result<usize, ModelError> {
    let mut written = 0usize;
    match value {
        GgufMetaValue::String(s) => {
            // Type tag: 8 = GGUF_TYPE_STRING
            writer.write_all(&8u32.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
            written += write_gguf_string(writer, s)?;
        }
        GgufMetaValue::U32(v) => {
            // Type tag: 4 = GGUF_TYPE_UINT32
            writer.write_all(&4u32.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
            writer.write_all(&v.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
        }
        GgufMetaValue::F32(v) => {
            // Type tag: 6 = GGUF_TYPE_FLOAT32
            writer.write_all(&6u32.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
            writer.write_all(&v.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
        }
        GgufMetaValue::StringArray(arr) => {
            // Type tag: 9 = GGUF_TYPE_ARRAY
            writer.write_all(&9u32.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
            // Element type: 8 = STRING
            writer.write_all(&8u32.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
            // Count
            writer.write_all(&(arr.len() as u64).to_le_bytes()).map_err(ModelError::IoError)?;
            written += 8;
            for s in arr {
                written += write_gguf_string(writer, s)?;
            }
        }
        GgufMetaValue::F32Array(arr) => {
            writer.write_all(&9u32.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
            // Element type: 6 = FLOAT32
            writer.write_all(&6u32.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
            writer.write_all(&(arr.len() as u64).to_le_bytes()).map_err(ModelError::IoError)?;
            written += 8;
            for v in arr {
                writer.write_all(&v.to_le_bytes()).map_err(ModelError::IoError)?;
                written += 4;
            }
        }
        GgufMetaValue::I32Array(arr) => {
            writer.write_all(&9u32.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
            // Element type: 5 = INT32
            writer.write_all(&5u32.to_le_bytes()).map_err(ModelError::IoError)?;
            written += 4;
            writer.write_all(&(arr.len() as u64).to_le_bytes()).map_err(ModelError::IoError)?;
            written += 8;
            for v in arr {
                writer.write_all(&v.to_le_bytes()).map_err(ModelError::IoError)?;
                written += 4;
            }
        }
    }
    Ok(written)
}

/// Build comprehensive GGUF metadata from CompatInfo + config.json.
/// This handles the safetensors → GGUF conversion case where there's
/// no source GGUF to copy metadata from.
fn build_gguf_metadata(
    model_name: &str,
    compat: Option<&CompatInfo>,
    config_json_dir: Option<&str>,
) -> Vec<(String, GgufMetaValue)> {
    // Try to read config.json for extra fields not in CompatInfo
    let config = config_json_dir.and_then(|dir| {
        let path = Path::new(dir).join("config.json");
        std::fs::read_to_string(&path).ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    });

    // Determine architecture string (e.g. "llama", "gemma", "mistral")
    let arch = compat
        .and_then(|c| c.architecture.as_deref())
        .or_else(|| config.as_ref()
            .and_then(|c| c.get("model_type"))
            .and_then(|v| v.as_str()))
        .unwrap_or("llama");

    // Normalize architecture for GGUF (strip "ForCausalLM" suffix, lowercase)
    let arch_clean = arch
        .to_lowercase()
        .replace("forcausallm", "")
        .replace("forsequenceclassification", "")
        .trim()
        .to_string();
    let arch_key = if arch_clean.is_empty() { "llama".to_string() } else { arch_clean };

    let mut kvs: Vec<(String, GgufMetaValue)> = Vec::new();

    // Required general keys
    kvs.push(("general.architecture".into(), GgufMetaValue::String(arch_key.clone())));
    kvs.push(("general.name".into(), GgufMetaValue::String(model_name.into())));
    kvs.push(("general.file_type".into(), GgufMetaValue::U32(0))); // 0 = F32

    // Architecture-specific keys from CompatInfo
    if let Some(c) = compat {
        if let Some(ctx) = c.context_length {
            kvs.push((format!("{}.context_length", arch_key), GgufMetaValue::U32(ctx as u32)));
        }
        if let Some(emb) = c.hidden_size {
            kvs.push((format!("{}.embedding_length", arch_key), GgufMetaValue::U32(emb as u32)));
        }
        if let Some(layers) = c.num_layers {
            kvs.push((format!("{}.block_count", arch_key), GgufMetaValue::U32(layers as u32)));
        }
        if let Some(heads) = c.num_attention_heads {
            kvs.push((format!("{}.attention.head_count", arch_key), GgufMetaValue::U32(heads as u32)));
        }
        if let Some(kv_heads) = c.num_kv_heads {
            kvs.push((format!("{}.attention.head_count_kv", arch_key), GgufMetaValue::U32(kv_heads as u32)));
        }
    }

    // Extra keys from config.json that CompatInfo doesn't carry
    if let Some(ref cfg) = config {
        // Feed-forward / intermediate size
        if let Some(ff) = cfg.get("intermediate_size").and_then(|v| v.as_u64()) {
            kvs.push((format!("{}.feed_forward_length", arch_key), GgufMetaValue::U32(ff as u32)));
        }

        // RoPE frequency base
        if let Some(theta) = cfg.get("rope_theta").and_then(|v| v.as_f64()) {
            kvs.push((format!("{}.rope.freq_base", arch_key), GgufMetaValue::F32(theta as f32)));
        }

        // Layer norm epsilon
        if let Some(eps) = cfg.get("rms_norm_eps").and_then(|v| v.as_f64()) {
            kvs.push((format!("{}.attention.layer_norm_rms_epsilon", arch_key), GgufMetaValue::F32(eps as f32)));
        }

        // Fill in any fields that CompatInfo didn't have (config.json as fallback)
        let fill_u32 = |kvs: &mut Vec<(String, GgufMetaValue)>, key: String, cfg: &serde_json::Value, json_field: &str| {
            if !kvs.iter().any(|(k, _)| *k == key) {
                if let Some(v) = cfg.get(json_field).and_then(|v| v.as_u64()) {
                    kvs.push((key, GgufMetaValue::U32(v as u32)));
                }
            }
        };

        fill_u32(&mut kvs, format!("{}.context_length", arch_key), cfg, "max_position_embeddings");
        fill_u32(&mut kvs, format!("{}.embedding_length", arch_key), cfg, "hidden_size");
        fill_u32(&mut kvs, format!("{}.block_count", arch_key), cfg, "num_hidden_layers");
        fill_u32(&mut kvs, format!("{}.attention.head_count", arch_key), cfg, "num_attention_heads");
        fill_u32(&mut kvs, format!("{}.attention.head_count_kv", arch_key), cfg, "num_key_value_heads");
    }

    // Embed tokenizer data from tokenizer.json
    if let Some(dir) = config_json_dir {
        if let Some(tok_kvs) = parse_tokenizer_for_gguf(dir) {
            kvs.extend(tok_kvs);
        }
    }

    kvs
}

/// Parse tokenizer.json + tokenizer_config.json and produce GGUF tokenizer metadata.
fn parse_tokenizer_for_gguf(dir: &str) -> Option<Vec<(String, GgufMetaValue)>> {
    let dir_path = Path::new(dir);

    // Read tokenizer.json
    let tok_path = dir_path.join("tokenizer.json");
    let tok_json: serde_json::Value = std::fs::read_to_string(&tok_path).ok()
        .and_then(|s| serde_json::from_str(&s).ok())?;

    // Read tokenizer_config.json for special token IDs
    let tok_config: Option<serde_json::Value> = std::fs::read_to_string(dir_path.join("tokenizer_config.json")).ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    let model = tok_json.get("model")?;
    let model_type = model.get("type").and_then(|v| v.as_str()).unwrap_or("BPE");

    // Map HF tokenizer type to GGUF tokenizer.ggml.model
    let ggml_model = match model_type {
        "BPE" => "gpt2",
        "Unigram" => "llama",  // SentencePiece unigram
        "WordPiece" => "bert",
        _ => "gpt2",
    };

    // Extract vocab: map of token → id
    let vocab = model.get("vocab").and_then(|v| v.as_object())?;

    // Get added_tokens for special token detection
    let added_tokens = tok_json.get("added_tokens")
        .and_then(|v| v.as_array());

    // Build token list sorted by ID
    let mut token_list: Vec<(u32, String, bool)> = Vec::with_capacity(vocab.len());
    for (token, id_val) in vocab {
        if let Some(id) = id_val.as_u64() {
            token_list.push((id as u32, token.clone(), false));
        }
    }

    // Add added_tokens (may override or supplement vocab)
    if let Some(added) = added_tokens {
        for entry in added {
            let id = entry.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let content = entry.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let special = entry.get("special").and_then(|v| v.as_bool()).unwrap_or(false);

            // Check if this ID already exists in our list
            if let Some(existing) = token_list.iter_mut().find(|(tid, _, _)| *tid == id) {
                existing.2 = special; // mark as special
            } else {
                token_list.push((id, content.to_string(), special));
            }
        }
    }

    token_list.sort_by_key(|(id, _, _)| *id);

    let vocab_size = token_list.last().map(|(id, _, _)| *id + 1).unwrap_or(0) as usize;

    // Build dense arrays (fill gaps with empty tokens)
    let mut tokens: Vec<String> = vec![String::new(); vocab_size];
    let mut scores: Vec<f32> = vec![0.0; vocab_size];
    let mut token_types: Vec<i32> = vec![1; vocab_size]; // 1 = normal

    for (id, token, special) in &token_list {
        let idx = *id as usize;
        if idx < vocab_size {
            tokens[idx] = token.clone();
            if *special {
                token_types[idx] = 3; // 3 = control
            }
        }
    }

    // Detect BOS/EOS token IDs
    let bos_token_id = tok_config.as_ref()
        .and_then(|c| {
            // Can be a string like "<s>" or an object with "content"
            c.get("bos_token")
                .and_then(|v| v.as_str().map(|s| s.to_string())
                    .or_else(|| v.get("content").and_then(|c| c.as_str().map(|s| s.to_string()))))
        })
        .and_then(|tok| token_list.iter().find(|(_, t, _)| t == &tok).map(|(id, _, _)| *id));

    let eos_token_id = tok_config.as_ref()
        .and_then(|c| {
            c.get("eos_token")
                .and_then(|v| v.as_str().map(|s| s.to_string())
                    .or_else(|| v.get("content").and_then(|c| c.as_str().map(|s| s.to_string()))))
        })
        .and_then(|tok| token_list.iter().find(|(_, t, _)| t == &tok).map(|(id, _, _)| *id));

    let mut kvs: Vec<(String, GgufMetaValue)> = Vec::new();

    kvs.push(("tokenizer.ggml.model".into(), GgufMetaValue::String(ggml_model.into())));
    kvs.push(("tokenizer.ggml.tokens".into(), GgufMetaValue::StringArray(tokens)));
    kvs.push(("tokenizer.ggml.scores".into(), GgufMetaValue::F32Array(scores)));
    kvs.push(("tokenizer.ggml.token_type".into(), GgufMetaValue::I32Array(token_types)));

    if let Some(bos) = bos_token_id {
        kvs.push(("tokenizer.ggml.bos_token_id".into(), GgufMetaValue::U32(bos)));
    }
    if let Some(eos) = eos_token_id {
        kvs.push(("tokenizer.ggml.eos_token_id".into(), GgufMetaValue::U32(eos)));
    }

    // Add merges if BPE
    if model_type == "BPE" {
        if let Some(merges) = model.get("merges").and_then(|v| v.as_array()) {
            let merge_strs: Vec<String> = merges.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            if !merge_strs.is_empty() {
                kvs.push(("tokenizer.ggml.merges".into(), GgufMetaValue::StringArray(merge_strs)));
            }
        }
    }

    Some(kvs)
}
