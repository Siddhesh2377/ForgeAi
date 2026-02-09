use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use super::error::ModelError;
use super::{
    derive_layer_count, format_file_size, format_param_count, ModelFormat, ModelInfo, TensorInfo,
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

// GGUF metadata value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

/// A simple cursor over a byte slice for sequential reads.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], ModelError> {
        if self.pos + n > self.data.len() {
            return Err(ModelError::ParseError {
                format: "GGUF".into(),
                reason: format!(
                    "Unexpected end of file at offset {} (need {} bytes, have {})",
                    self.pos,
                    n,
                    self.remaining()
                ),
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, ModelError> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, ModelError> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_i8(&mut self) -> Result<i8, ModelError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_i16(&mut self) -> Result<i16, ModelError> {
        Ok(i16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, ModelError> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, ModelError> {
        Ok(i64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, ModelError> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, ModelError> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String, ModelError> {
        let len = self.read_u64()? as usize;
        if len > 1_000_000 {
            return Err(ModelError::ParseError {
                format: "GGUF".into(),
                reason: format!("String length {} is unreasonably large", len),
            });
        }
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| ModelError::ParseError {
            format: "GGUF".into(),
            reason: format!("Invalid UTF-8 string: {}", e),
        })
    }

    fn read_bool(&mut self) -> Result<bool, ModelError> {
        Ok(self.read_u8()? != 0)
    }
}

#[derive(Debug, Clone)]
enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    #[allow(dead_code)]
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    fn as_string(&self) -> Option<String> {
        match self {
            GgufValue::String(s) => Some(s.clone()),
            GgufValue::Uint8(v) => Some(v.to_string()),
            GgufValue::Int8(v) => Some(v.to_string()),
            GgufValue::Uint16(v) => Some(v.to_string()),
            GgufValue::Int16(v) => Some(v.to_string()),
            GgufValue::Uint32(v) => Some(v.to_string()),
            GgufValue::Int32(v) => Some(v.to_string()),
            GgufValue::Float32(v) => Some(v.to_string()),
            GgufValue::Bool(v) => Some(v.to_string()),
            GgufValue::Uint64(v) => Some(v.to_string()),
            GgufValue::Int64(v) => Some(v.to_string()),
            GgufValue::Float64(v) => Some(v.to_string()),
            GgufValue::Array(_) => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint8(v) => Some(*v as u64),
            GgufValue::Uint16(v) => Some(*v as u64),
            GgufValue::Uint32(v) => Some(*v as u64),
            GgufValue::Uint64(v) => Some(*v),
            GgufValue::Int8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::Int16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::Int32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
}

fn read_value(reader: &mut Reader, value_type: u32) -> Result<GgufValue, ModelError> {
    match value_type {
        GGUF_TYPE_UINT8 => Ok(GgufValue::Uint8(reader.read_u8()?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::Int8(reader.read_i8()?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::Uint16(reader.read_u16()?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::Int16(reader.read_i16()?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::Uint32(reader.read_u32()?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::Int32(reader.read_i32()?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::Float32(reader.read_f32()?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(reader.read_bool()?)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(reader.read_string()?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = reader.read_u32()?;
            let count = reader.read_u64()? as usize;
            if count > 10_000_000 {
                return Err(ModelError::ParseError {
                    format: "GGUF".into(),
                    reason: format!("Array length {} is unreasonably large", count),
                });
            }
            let mut items = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                items.push(read_value(reader, elem_type)?);
            }
            Ok(GgufValue::Array(items))
        }
        GGUF_TYPE_UINT64 => Ok(GgufValue::Uint64(reader.read_u64()?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::Int64(reader.read_i64()?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::Float64(reader.read_f64()?)),
        _ => Err(ModelError::ParseError {
            format: "GGUF".into(),
            reason: format!("Unknown metadata value type: {}", value_type),
        }),
    }
}

pub fn parse(path: &Path) -> Result<ModelInfo, ModelError> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();

    if file_size < 24 {
        return Err(ModelError::FileTooSmall(file_size));
    }

    let mmap = unsafe { Mmap::map(&file)? };
    let mut reader = Reader::new(&mmap);

    // Validate magic
    let magic = reader.read_bytes(4)?;
    if magic != GGUF_MAGIC {
        return Err(ModelError::ParseError {
            format: "GGUF".into(),
            reason: format!(
                "Invalid magic bytes: {:?} (expected {:?})",
                magic, GGUF_MAGIC
            ),
        });
    }

    // Version
    let version = reader.read_u32()?;
    if version < 2 || version > 3 {
        return Err(ModelError::ParseError {
            format: "GGUF".into(),
            reason: format!("Unsupported GGUF version: {} (supported: 2, 3)", version),
        });
    }

    // Tensor count and metadata KV count
    let tensor_count: u64;
    let metadata_kv_count: u64;

    if version >= 3 {
        tensor_count = reader.read_u64()?;
        metadata_kv_count = reader.read_u64()?;
    } else {
        tensor_count = reader.read_u32()? as u64;
        metadata_kv_count = reader.read_u32()? as u64;
    }

    // Parse metadata KV pairs
    let mut kv_map: HashMap<String, GgufValue> = HashMap::new();

    for _ in 0..metadata_kv_count {
        let key = reader.read_string()?;
        let value_type = reader.read_u32()?;
        let value = read_value(&mut reader, value_type)?;
        kv_map.insert(key, value);
    }

    // Parse tensor info entries
    let mut tensors = Vec::new();
    let mut total_params: u64 = 0;

    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        let n_dims = reader.read_u32()?;

        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(reader.read_u64()?);
        }

        let ggml_type = reader.read_u32()?;

        // offset — u64
        let _offset = reader.read_u64()?;

        let param_count: u64 = if shape.is_empty() {
            0
        } else {
            shape.iter().product()
        };
        total_params += param_count;

        tensors.push(TensorInfo {
            name,
            dtype: ggml_type_name(ggml_type).to_string(),
            shape,
        });
    }

    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    // Extract well-known metadata
    let architecture = kv_map
        .get("general.architecture")
        .and_then(|v| v.as_string());

    let arch_prefix = architecture.clone().unwrap_or_default();

    let context_length = kv_map
        .get(&format!("{}.context_length", arch_prefix))
        .and_then(|v| v.as_u64());

    let embedding_size = kv_map
        .get(&format!("{}.embedding_length", arch_prefix))
        .and_then(|v| v.as_u64());

    let block_count = kv_map
        .get(&format!("{}.block_count", arch_prefix))
        .and_then(|v| v.as_u64());

    let file_type = kv_map.get("general.file_type").and_then(|v| v.as_u64());
    let quantization = file_type.map(|ft| gguf_file_type_name(ft).to_string());

    // Build general metadata for display
    let mut metadata = HashMap::new();
    for (key, value) in &kv_map {
        if let Some(s) = value.as_string() {
            // Skip very long values
            if s.len() <= 500 {
                metadata.insert(key.clone(), s);
            }
        }
    }

    // Extract tokenizer vocab size and resolve special token names
    if let Some(GgufValue::Array(tokens)) = kv_map.get("tokenizer.ggml.tokens") {
        metadata.insert(
            "tokenizer.ggml.tokens_count".to_string(),
            tokens.len().to_string(),
        );

        let special_keys = [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.unknown_token_id",
        ];
        for key in &special_keys {
            if let Some(id_val) = kv_map.get(*key).and_then(|v| v.as_u64()) {
                let idx = id_val as usize;
                if idx < tokens.len() {
                    if let GgufValue::String(token_str) = &tokens[idx] {
                        metadata.insert(format!("{}_resolved", key), token_str.clone());
                    }
                }
            }
        }
    }

    Ok(ModelInfo {
        file_name: path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        file_path: path.to_string_lossy().to_string(),
        file_size,
        file_size_display: format_file_size(file_size),
        format: ModelFormat::Gguf,
        tensor_count,
        parameter_count: total_params,
        parameter_count_display: format_param_count(total_params),
        layer_count: block_count.or_else(|| derive_layer_count(&tensors)),
        quantization,
        architecture,
        context_length,
        embedding_size,
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

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "UNKNOWN",
    }
}

fn gguf_file_type_name(ft: u64) -> &'static str {
    match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "IQ3_XXS",
        22 => "IQ1_S",
        23 => "IQ4_NL",
        24 => "IQ3_S",
        25 => "IQ2_S",
        26 => "IQ4_XS",
        27 => "IQ1_M",
        28 => "BF16",
        _ => "UNKNOWN",
    }
}
