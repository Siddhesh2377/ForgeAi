use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::model::error::ModelError;
use super::config::{ColumnAnalysis, DatasetFormat, DatasetFullInfo, DatasetInfo};

/// Detect dataset format, row count, columns, and preview.
pub fn detect_dataset(path: &str) -> Result<DatasetInfo, ModelError> {
    let (rows, columns, preview, format, size) = parse_dataset_core(path, 5)?;
    let detected_template = detect_template(&columns, &preview);
    let size_display = format_size(size);

    Ok(DatasetInfo {
        path: path.to_string(),
        format,
        rows,
        columns,
        preview,
        size,
        size_display,
        detected_template,
    })
}

/// Full dataset detection with extended preview and column analysis (for DataStudio).
pub fn detect_dataset_full(path: &str, max_preview: usize) -> Result<DatasetFullInfo, ModelError> {
    let (rows, columns, preview, format, size) = parse_dataset_core(path, max_preview)?;
    let detected_template = detect_template(&columns, &preview);
    let size_display = format_size(size);
    let column_analysis = analyze_columns(&columns, &preview);

    Ok(DatasetFullInfo {
        path: path.to_string(),
        format,
        rows,
        columns,
        column_analysis,
        preview,
        size,
        size_display,
        detected_template,
    })
}

fn parse_dataset_core(
    path: &str,
    max_preview: usize,
) -> Result<(u64, Vec<String>, Vec<serde_json::Value>, DatasetFormat, u64), ModelError> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(ModelError::FileNotFound(path.to_string()));
    }

    let meta = fs::metadata(p).map_err(ModelError::IoError)?;
    let size = meta.len();

    let ext = p.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let format = match ext.as_str() {
        "json" => DatasetFormat::Json,
        "jsonl" | "ndjson" => DatasetFormat::Jsonl,
        "csv" => DatasetFormat::Csv,
        "parquet" => DatasetFormat::Parquet,
        _ => return Err(ModelError::UnsupportedFormat(format!(
            "Unsupported dataset format: .{}", ext
        ))),
    };

    let (rows, columns, preview) = match &format {
        DatasetFormat::Json => parse_json(p, max_preview)?,
        DatasetFormat::Jsonl => parse_jsonl(p, max_preview)?,
        DatasetFormat::Csv => parse_csv(p, max_preview)?,
        DatasetFormat::Parquet => parse_parquet(p, max_preview)?
    };

    Ok((rows, columns, preview, format, size))
}

fn parse_json(path: &Path, max_preview: usize) -> Result<(u64, Vec<String>, Vec<serde_json::Value>), ModelError> {
    let content = fs::read_to_string(path).map_err(ModelError::IoError)?;
    let val: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| ModelError::TrainingError(format!("Invalid JSON: {}", e)))?;

    match val {
        serde_json::Value::Array(arr) => {
            let rows = arr.len() as u64;
            let columns = if let Some(first) = arr.first() {
                extract_columns(first)
            } else {
                vec![]
            };
            let preview: Vec<serde_json::Value> = arr.into_iter().take(max_preview).collect();
            Ok((rows, columns, preview))
        }
        serde_json::Value::Object(_) => {
            if let Some(arr) = val.get("data").or_else(|| val.get("rows")).and_then(|v| v.as_array()) {
                let rows = arr.len() as u64;
                let columns = if let Some(first) = arr.first() {
                    extract_columns(first)
                } else {
                    vec![]
                };
                let preview: Vec<serde_json::Value> = arr.iter().take(max_preview).cloned().collect();
                Ok((rows, columns, preview))
            } else {
                Ok((1, extract_columns(&val), vec![val]))
            }
        }
        _ => Err(ModelError::TrainingError("JSON root must be an array or object".into())),
    }
}

fn parse_jsonl(path: &Path, max_preview: usize) -> Result<(u64, Vec<String>, Vec<serde_json::Value>), ModelError> {
    let file = fs::File::open(path).map_err(ModelError::IoError)?;
    let reader = BufReader::new(file);

    let mut rows = 0u64;
    let mut columns = vec![];
    let mut preview = vec![];

    for line in reader.lines() {
        let line = line.map_err(ModelError::IoError)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if rows == 0 {
                columns = extract_columns(&val);
            }
            if (rows as usize) < max_preview {
                preview.push(val);
            }
            rows += 1;
        }
    }

    Ok((rows, columns, preview))
}

fn parse_csv(path: &Path, max_preview: usize) -> Result<(u64, Vec<String>, Vec<serde_json::Value>), ModelError> {
    let content = fs::read_to_string(path).map_err(ModelError::IoError)?;
    let mut lines = content.lines();

    let header = match lines.next() {
        Some(h) => h,
        None => return Ok((0, vec![], vec![])),
    };

    let columns: Vec<String> = header.split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect();

    let mut rows = 0u64;
    let mut preview = vec![];

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        if (rows as usize) <= max_preview {
            let values: Vec<&str> = line.split(',').collect();
            let mut obj = serde_json::Map::new();
            for (i, col) in columns.iter().enumerate() {
                let val = values.get(i).unwrap_or(&"").trim().trim_matches('"');
                obj.insert(col.clone(), serde_json::Value::String(val.to_string()));
            }
            preview.push(serde_json::Value::Object(obj));
        }
    }

    Ok((rows, columns, preview))
}

fn parse_parquet(path: &Path, max_preview: usize) -> Result<(u64, Vec<String>, Vec<serde_json::Value>), ModelError> {
    let file = fs::File::open(path).map_err(ModelError::IoError)?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| ModelError::TrainingError(format!("Failed to open parquet: {}", e)))?;

    let total_rows = builder.metadata().file_metadata().num_rows() as u64;

    // Extract column names from parquet schema before consuming builder
    let parquet_schema = builder.metadata().file_metadata().schema_descr();
    let columns: Vec<String> = parquet_schema
        .columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();

    let reader = builder
        .with_batch_size(max_preview)
        .build()
        .map_err(|e| ModelError::TrainingError(format!("Failed to read parquet: {}", e)))?;

    let mut preview: Vec<serde_json::Value> = Vec::new();

    // Use arrow_json to convert record batches to JSON rows
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| ModelError::TrainingError(format!("Parquet batch error: {}", e)))?;

        let mut buf = Vec::new();
        {
            let mut writer = arrow_json::ArrayWriter::new(&mut buf);
            writer.write(&batch)
                .map_err(|e| ModelError::TrainingError(format!("Parquet to JSON error: {}", e)))?;
            writer.finish()
                .map_err(|e| ModelError::TrainingError(format!("Parquet JSON finish error: {}", e)))?;
        }

        let rows: Vec<serde_json::Value> = serde_json::from_slice(&buf)
            .map_err(|e| ModelError::TrainingError(format!("Parquet JSON parse error: {}", e)))?;

        for row in rows {
            preview.push(row);
            if preview.len() >= max_preview {
                break;
            }
        }
        if preview.len() >= max_preview {
            break;
        }
    }

    Ok((total_rows, columns, preview))
}

fn extract_columns(val: &serde_json::Value) -> Vec<String> {
    match val {
        serde_json::Value::Object(map) => map.keys().cloned().collect(),
        _ => vec![],
    }
}

/// Detect common dataset templates from column names and sample data.
pub fn detect_template(columns: &[String], _preview: &[serde_json::Value]) -> Option<String> {
    let cols_lower: Vec<String> = columns.iter().map(|c| c.to_lowercase()).collect();

    // Alpaca format: instruction, input, output
    if cols_lower.contains(&"instruction".into())
        && cols_lower.contains(&"output".into())
    {
        return Some("alpaca".to_string());
    }

    // ShareGPT: conversations array
    if cols_lower.contains(&"conversations".into()) {
        return Some("sharegpt".to_string());
    }

    // ChatML / OpenAI: messages array
    if cols_lower.contains(&"messages".into()) {
        return Some("chat_ml".to_string());
    }

    // DPO: chosen + rejected
    if cols_lower.contains(&"chosen".into()) && cols_lower.contains(&"rejected".into()) {
        return Some("dpo_pairs".to_string());
    }

    // Simple text completion
    if cols_lower.contains(&"text".into()) && columns.len() <= 2 {
        return Some("text".to_string());
    }

    // Prompt/completion
    if cols_lower.contains(&"prompt".into()) && cols_lower.contains(&"completion".into()) {
        return Some("prompt_completion".to_string());
    }

    None
}

fn analyze_columns(columns: &[String], preview: &[serde_json::Value]) -> Vec<ColumnAnalysis> {
    columns.iter().map(|col| {
        let mut non_null = 0u64;
        let mut null_count = 0u64;
        let mut total_len = 0usize;
        let mut string_count = 0u64;
        let mut samples: Vec<String> = Vec::new();
        let mut detected_type = "null";

        for row in preview {
            if let Some(val) = row.get(col) {
                match val {
                    serde_json::Value::Null => { null_count += 1; }
                    serde_json::Value::String(s) => {
                        non_null += 1;
                        total_len += s.len();
                        string_count += 1;
                        detected_type = "string";
                        if samples.len() < 3 {
                            let trunc: String = s.chars().take(80).collect();
                            samples.push(trunc);
                        }
                    }
                    serde_json::Value::Number(_) => {
                        non_null += 1;
                        if detected_type == "null" { detected_type = "number"; }
                        if samples.len() < 3 {
                            samples.push(val.to_string());
                        }
                    }
                    serde_json::Value::Bool(_) => {
                        non_null += 1;
                        if detected_type == "null" { detected_type = "boolean"; }
                        if samples.len() < 3 {
                            samples.push(val.to_string());
                        }
                    }
                    serde_json::Value::Array(_) => {
                        non_null += 1;
                        if detected_type == "null" { detected_type = "array"; }
                        if samples.len() < 3 {
                            let s = val.to_string();
                            let trunc: String = s.chars().take(80).collect();
                            samples.push(trunc);
                        }
                    }
                    serde_json::Value::Object(_) => {
                        non_null += 1;
                        if detected_type == "null" { detected_type = "object"; }
                        if samples.len() < 3 {
                            let s = val.to_string();
                            let trunc: String = s.chars().take(80).collect();
                            samples.push(trunc);
                        }
                    }
                }
            } else {
                null_count += 1;
            }
        }

        let avg_length = if string_count > 0 {
            Some(total_len as f64 / string_count as f64)
        } else {
            None
        };

        ColumnAnalysis {
            name: col.clone(),
            dtype: detected_type.to_string(),
            non_null_count: non_null,
            null_count,
            sample_values: samples,
            avg_length,
        }
    }).collect()
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
