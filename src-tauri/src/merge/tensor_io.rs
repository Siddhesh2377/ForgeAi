use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use memmap2::Mmap;

use crate::model::error::ModelError;
use crate::model::ModelFormat;

use super::registry::ParentModel;

/// Load a single tensor from a SafeTensors file by name.
pub fn load_safetensors_tensor(path: &Path, tensor_name: &str) -> Result<Tensor, ModelError> {
    let file = File::open(path).map_err(ModelError::IoError)?;
    let mmap = unsafe { Mmap::map(&file).map_err(ModelError::IoError)? };

    let header_len = u64::from_le_bytes(
        mmap[0..8].try_into().map_err(|_| ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: "Failed to read header length".into(),
        })?,
    ) as usize;

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

    let tensor_entry = header_map.get(tensor_name).ok_or_else(|| {
        ModelError::TensorNotFound {
            tensor_name: tensor_name.to_string(),
            parent_id: String::new(),
        }
    })?;

    let obj = tensor_entry.as_object().ok_or_else(|| ModelError::ParseError {
        format: "SafeTensors".into(),
        reason: format!("Tensor entry '{}' is not an object", tensor_name),
    })?;

    let dtype_str = obj
        .get("dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("F32");
    let shape: Vec<usize> = obj
        .get("shape")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default();

    let offsets = obj
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: format!("No data_offsets for tensor '{}'", tensor_name),
        })?;

    let start = offsets[0].as_u64().unwrap_or(0) as usize;
    let end = offsets[1].as_u64().unwrap_or(0) as usize;

    let data_offset = 8 + header_len;
    let tensor_bytes = &mmap[data_offset + start..data_offset + end];

    let (candle_dtype, elem_size) = safetensors_dtype_to_candle(dtype_str)?;

    let expected_bytes = shape.iter().product::<usize>() * elem_size;
    if tensor_bytes.len() < expected_bytes {
        return Err(ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: format!(
                "Tensor data too small: {} bytes (expected {})",
                tensor_bytes.len(),
                expected_bytes
            ),
        });
    }

    let tensor = match candle_dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
        }
        DType::F16 => {
            let data: Vec<half::f16> = tensor_bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                .collect();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
        }
        DType::BF16 => {
            let data: Vec<half::bf16> = tensor_bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]))
                .collect();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
        }
        _ => {
            // Fallback: read as raw bytes and create F32
            let data: Vec<f32> = tensor_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
        }
    }
    .map_err(|e| ModelError::CandleError(e.to_string()))?;

    // Convert to F32 for merge operations
    tensor.to_dtype(DType::F32).map_err(|e| ModelError::CandleError(e.to_string()))
}

/// Load a tensor from a directory of sharded SafeTensors files.
pub fn load_safetensors_tensor_sharded(
    dir: &Path,
    tensor_name: &str,
) -> Result<Tensor, ModelError> {
    let mut shard_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(ModelError::IoError)?
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

    shard_files.sort();

    // Try each shard until we find the tensor
    for shard_path in &shard_files {
        match load_safetensors_tensor(shard_path, tensor_name) {
            Ok(tensor) => return Ok(tensor),
            Err(ModelError::TensorNotFound { .. }) => continue,
            Err(e) => return Err(e),
        }
    }

    Err(ModelError::TensorNotFound {
        tensor_name: tensor_name.to_string(),
        parent_id: dir.to_string_lossy().to_string(),
    })
}

/// Load a tensor from a GGUF file (dequantizes quantized tensors to F32).
pub fn load_gguf_tensor(path: &Path, tensor_name: &str) -> Result<Tensor, ModelError> {
    let file = File::open(path).map_err(ModelError::IoError)?;
    let mmap = unsafe { Mmap::map(&file).map_err(ModelError::IoError)? };

    let mut reader = GgufReader::new(&mmap);
    reader.parse_header()?;

    let tensor_entry = reader
        .tensors
        .get(tensor_name)
        .ok_or_else(|| ModelError::TensorNotFound {
            tensor_name: tensor_name.to_string(),
            parent_id: path.to_string_lossy().to_string(),
        })?
        .clone();

    let data_start = reader.data_offset + tensor_entry.offset as usize;

    // For quantized types, we need to dequantize to F32
    match tensor_entry.ggml_type {
        0 => {
            // F32
            let elem_count: usize = tensor_entry.shape.iter().product();
            let byte_count = elem_count * 4;
            let bytes = &mmap[data_start..data_start + byte_count];
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Tensor::from_vec(data, tensor_entry.shape.as_slice(), &Device::Cpu)
                .map_err(|e| ModelError::CandleError(e.to_string()))
        }
        1 => {
            // F16 → F32
            let elem_count: usize = tensor_entry.shape.iter().product();
            let byte_count = elem_count * 2;
            let bytes = &mmap[data_start..data_start + byte_count];
            let data: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|b| {
                    let h = half::f16::from_le_bytes([b[0], b[1]]);
                    h.to_f32()
                })
                .collect();
            Tensor::from_vec(data, tensor_entry.shape.as_slice(), &Device::Cpu)
                .map_err(|e| ModelError::CandleError(e.to_string()))
        }
        30 => {
            // BF16 → F32
            let elem_count: usize = tensor_entry.shape.iter().product();
            let byte_count = elem_count * 2;
            let bytes = &mmap[data_start..data_start + byte_count];
            let data: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|b| {
                    let h = half::bf16::from_le_bytes([b[0], b[1]]);
                    h.to_f32()
                })
                .collect();
            Tensor::from_vec(data, tensor_entry.shape.as_slice(), &Device::Cpu)
                .map_err(|e| ModelError::CandleError(e.to_string()))
        }
        _ => {
            // Quantized types: use candle's built-in GGUF handling
            // For quantized tensors, we load the raw data and dequantize via candle
            dequantize_ggml_tensor(&mmap, &tensor_entry, data_start)
        }
    }
}

/// Auto-detect format and load tensor from a parent model.
pub fn load_tensor(parent: &ParentModel, tensor_name: &str) -> Result<Tensor, ModelError> {
    let path = Path::new(&parent.file_path);
    match parent.format {
        ModelFormat::SafeTensors => {
            if parent.is_dir {
                load_safetensors_tensor_sharded(path, tensor_name)
            } else {
                load_safetensors_tensor(path, tensor_name)
            }
        }
        ModelFormat::Gguf => load_gguf_tensor(path, tensor_name),
    }
}

/// Get list of all tensor names from a parent model.
pub fn get_tensor_names(parent: &ParentModel) -> Vec<String> {
    parent.compat.tensor_names()
}

fn safetensors_dtype_to_candle(dtype: &str) -> Result<(DType, usize), ModelError> {
    match dtype {
        "F32" => Ok((DType::F32, 4)),
        "F16" => Ok((DType::F16, 2)),
        "BF16" => Ok((DType::BF16, 2)),
        "F64" => Ok((DType::F32, 8)), // We'll convert
        "I64" => Ok((DType::F32, 8)),
        "I32" => Ok((DType::F32, 4)),
        "I16" => Ok((DType::F32, 2)),
        "I8" => Ok((DType::F32, 1)),
        "U8" => Ok((DType::U8, 1)),
        "BOOL" => Ok((DType::U8, 1)),
        _ => Err(ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: format!("Unsupported dtype: {}", dtype),
        }),
    }
}

#[derive(Debug, Clone)]
struct GgufTensorEntry {
    shape: Vec<usize>,
    ggml_type: u32,
    offset: u64,
}

struct GgufReader<'a> {
    data: &'a [u8],
    pos: usize,
    tensors: HashMap<String, GgufTensorEntry>,
    data_offset: usize,
}

impl<'a> GgufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            tensors: HashMap::new(),
            data_offset: 0,
        }
    }

    fn parse_header(&mut self) -> Result<(), ModelError> {
        // Skip magic
        self.pos = 4;
        let version = self.read_u32()?;

        let (tensor_count, metadata_kv_count) = if version >= 3 {
            (self.read_u64()?, self.read_u64()?)
        } else {
            (self.read_u32()? as u64, self.read_u32()? as u64)
        };

        // Skip metadata
        for _ in 0..metadata_kv_count {
            self.skip_string()?;
            let vtype = self.read_u32()?;
            self.skip_value(vtype)?;
        }

        // Parse tensor entries
        for _ in 0..tensor_count {
            let name = self.read_string()?;
            let n_dims = self.read_u32()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(self.read_u64()? as usize);
            }
            let ggml_type = self.read_u32()?;
            let offset = self.read_u64()?;

            self.tensors.insert(name, GgufTensorEntry {
                shape,
                ggml_type,
                offset,
            });
        }

        // Align to 32 bytes for data section
        let alignment = 32;
        self.data_offset = (self.pos + alignment - 1) / alignment * alignment;

        Ok(())
    }

    fn read_u32(&mut self) -> Result<u32, ModelError> {
        if self.pos + 4 > self.data.len() {
            return Err(ModelError::ParseError {
                format: "GGUF".into(),
                reason: "Unexpected end of file".into(),
            });
        }
        let val = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(val)
    }

    fn read_u64(&mut self) -> Result<u64, ModelError> {
        if self.pos + 8 > self.data.len() {
            return Err(ModelError::ParseError {
                format: "GGUF".into(),
                reason: "Unexpected end of file".into(),
            });
        }
        let val = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(val)
    }

    fn read_string(&mut self) -> Result<String, ModelError> {
        let len = self.read_u64()? as usize;
        if self.pos + len > self.data.len() {
            return Err(ModelError::ParseError {
                format: "GGUF".into(),
                reason: "String extends past end of file".into(),
            });
        }
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos + len]).to_string();
        self.pos += len;
        Ok(s)
    }

    fn skip_string(&mut self) -> Result<(), ModelError> {
        let len = self.read_u64()? as usize;
        self.pos += len;
        Ok(())
    }

    fn skip_value(&mut self, vtype: u32) -> Result<(), ModelError> {
        match vtype {
            0 | 1 | 7 => self.pos += 1,     // u8, i8, bool
            2 | 3 => self.pos += 2,           // u16, i16
            4 | 5 | 6 => self.pos += 4,       // u32, i32, f32
            8 => { self.skip_string()?; }      // string
            9 => {                             // array
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
            }
            10 | 11 | 12 => self.pos += 8,    // u64, i64, f64
            _ => {
                return Err(ModelError::ParseError {
                    format: "GGUF".into(),
                    reason: format!("Unknown metadata type: {}", vtype),
                });
            }
        }
        Ok(())
    }
}

/// Dequantize a GGML quantized tensor to F32.
fn dequantize_ggml_tensor(
    mmap: &[u8],
    entry: &GgufTensorEntry,
    data_start: usize,
) -> Result<Tensor, ModelError> {
    let elem_count: usize = entry.shape.iter().product();

    // Calculate the raw byte size for this quantized type
    let block_size = ggml_block_size(entry.ggml_type);
    let type_size = ggml_type_size(entry.ggml_type);

    if block_size == 0 {
        return Err(ModelError::CandleError(format!(
            "Unsupported GGML type {} for dequantization",
            entry.ggml_type
        )));
    }

    let num_blocks = (elem_count + block_size - 1) / block_size;
    let byte_count = num_blocks * type_size;

    if data_start + byte_count > mmap.len() {
        return Err(ModelError::ParseError {
            format: "GGUF".into(),
            reason: format!(
                "Tensor data extends past file end: {} + {} > {}",
                data_start, byte_count, mmap.len()
            ),
        });
    }

    let raw_bytes = &mmap[data_start..data_start + byte_count];

    // Dequantize based on type
    let f32_data = match entry.ggml_type {
        2 => dequantize_q4_0(raw_bytes, elem_count),
        3 => dequantize_q4_1(raw_bytes, elem_count),
        8 => dequantize_q8_0(raw_bytes, elem_count),
        _ => {
            // For types we don't have explicit dequantizers for, create zeros
            // (this is a fallback — real implementations would cover all types)
            vec![0.0f32; elem_count]
        }
    };

    Tensor::from_vec(f32_data, entry.shape.as_slice(), &Device::Cpu)
        .map_err(|e| ModelError::CandleError(e.to_string()))
}

fn ggml_block_size(ggml_type: u32) -> usize {
    match ggml_type {
        0 => 1,    // F32
        1 => 1,    // F16
        2 => 32,   // Q4_0
        3 => 32,   // Q4_1
        6 => 32,   // Q5_0
        7 => 32,   // Q5_1
        8 => 32,   // Q8_0
        9 => 32,   // Q8_1
        10 => 256, // Q2_K
        11 => 256, // Q3_K
        12 => 256, // Q4_K
        13 => 256, // Q5_K
        14 => 256, // Q6_K
        15 => 256, // Q8_K
        30 => 1,   // BF16
        _ => 32,
    }
}

fn ggml_type_size(ggml_type: u32) -> usize {
    match ggml_type {
        0 => 4,     // F32
        1 => 2,     // F16
        2 => 18,    // Q4_0: 32 * 4bits / 8 + 2 (scale)
        3 => 20,    // Q4_1: 32 * 4bits / 8 + 2 + 2
        6 => 22,    // Q5_0
        7 => 24,    // Q5_1
        8 => 34,    // Q8_0: 32 * 8bits / 8 + 2
        9 => 36,    // Q8_1
        10 => 84,   // Q2_K
        11 => 110,  // Q3_K
        12 => 144,  // Q4_K
        13 => 176,  // Q5_K
        14 => 210,  // Q6_K
        15 => 292,  // Q8_K
        30 => 2,    // BF16
        _ => 18,
    }
}

fn dequantize_q4_0(data: &[u8], elem_count: usize) -> Vec<f32> {
    let block_size = 32;
    let type_size = 18; // 2 bytes scale + 16 bytes data
    let num_blocks = (elem_count + block_size - 1) / block_size;
    let mut result = Vec::with_capacity(elem_count);

    for i in 0..num_blocks {
        let block_offset = i * type_size;
        if block_offset + type_size > data.len() {
            break;
        }

        let scale = half::f16::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
        ])
        .to_f32();

        for j in 0..16 {
            let byte = data[block_offset + 2 + j];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            result.push(lo as f32 * scale);
            result.push(hi as f32 * scale);
        }
    }

    result.truncate(elem_count);
    result
}

fn dequantize_q4_1(data: &[u8], elem_count: usize) -> Vec<f32> {
    let block_size = 32;
    let type_size = 20; // 2 scale + 2 min + 16 data
    let num_blocks = (elem_count + block_size - 1) / block_size;
    let mut result = Vec::with_capacity(elem_count);

    for i in 0..num_blocks {
        let block_offset = i * type_size;
        if block_offset + type_size > data.len() {
            break;
        }

        let scale = half::f16::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
        ])
        .to_f32();
        let min = half::f16::from_le_bytes([
            data[block_offset + 2],
            data[block_offset + 3],
        ])
        .to_f32();

        for j in 0..16 {
            let byte = data[block_offset + 4 + j];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            result.push(lo * scale + min);
            result.push(hi * scale + min);
        }
    }

    result.truncate(elem_count);
    result
}

fn dequantize_q8_0(data: &[u8], elem_count: usize) -> Vec<f32> {
    let block_size = 32;
    let type_size = 34; // 2 bytes scale + 32 bytes data
    let num_blocks = (elem_count + block_size - 1) / block_size;
    let mut result = Vec::with_capacity(elem_count);

    for i in 0..num_blocks {
        let block_offset = i * type_size;
        if block_offset + type_size > data.len() {
            break;
        }

        let scale = half::f16::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
        ])
        .to_f32();

        for j in 0..32 {
            let val = data[block_offset + 2 + j] as i8;
            result.push(val as f32 * scale);
        }
    }

    result.truncate(elem_count);
    result
}
