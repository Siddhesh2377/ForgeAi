use std::io::Read;
use std::path::PathBuf;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{Emitter, Manager, State};
use tokio::io::{AsyncBufReadExt, AsyncReadExt};

use crate::model::error::ModelError;
use crate::model::inspect::{self, InspectData};
use crate::model::state::AppState;
use crate::model::{ModelFormat, ModelInfo};

fn detect_format(path: &std::path::Path) -> Result<ModelFormat, ModelError> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .as_deref()
    {
        Some("safetensors") => Ok(ModelFormat::SafeTensors),
        Some("gguf") => Ok(ModelFormat::Gguf),
        Some(ext) => Err(ModelError::UnsupportedFormat(ext.to_string())),
        None => Err(ModelError::UnsupportedFormat("no extension".to_string())),
    }
}

// ── Model Commands ─────────────────────────────────────

#[tauri::command]
pub fn load_model(path: String, state: State<'_, AppState>) -> Result<ModelInfo, ModelError> {
    let path = PathBuf::from(&path);

    if !path.exists() {
        return Err(ModelError::FileNotFound(
            path.to_string_lossy().to_string(),
        ));
    }

    let format = detect_format(&path)?;

    let info = match format {
        ModelFormat::SafeTensors => crate::model::safetensors::parse(&path)?,
        ModelFormat::Gguf => crate::model::gguf::parse(&path)?,
    };

    let mut loaded = state.loaded_model.lock().unwrap();
    *loaded = Some(info.clone());

    Ok(info)
}

#[tauri::command]
pub fn load_model_dir(path: String, state: State<'_, AppState>) -> Result<ModelInfo, ModelError> {
    let path = PathBuf::from(&path);

    if !path.exists() {
        return Err(ModelError::FileNotFound(
            path.to_string_lossy().to_string(),
        ));
    }

    if !path.is_dir() {
        return Err(ModelError::ParseError {
            format: "SafeTensors".into(),
            reason: "Path is not a directory".into(),
        });
    }

    let info = crate::model::safetensors::parse_dir(&path)?;

    let mut loaded = state.loaded_model.lock().unwrap();
    *loaded = Some(info.clone());

    Ok(info)
}

#[tauri::command]
pub fn get_loaded_model(state: State<'_, AppState>) -> Option<ModelInfo> {
    state.loaded_model.lock().unwrap().clone()
}

#[tauri::command]
pub fn unload_model(state: State<'_, AppState>) {
    let mut loaded = state.loaded_model.lock().unwrap();
    *loaded = None;
}

#[tauri::command]
pub fn inspect_model(state: State<'_, AppState>) -> Result<InspectData, ModelError> {
    let loaded = state.loaded_model.lock().unwrap();
    let info = loaded.as_ref().ok_or_else(|| ModelError::ParseError {
        format: "inspect".into(),
        reason: "No model loaded".into(),
    })?;

    Ok(inspect::analyze(&info.all_tensors, &info.metadata))
}

#[tauri::command]
pub fn inspect_capabilities(
    state: State<'_, AppState>,
) -> Result<crate::merge::capabilities::CapabilityReport, ModelError> {
    let loaded = state.loaded_model.lock().unwrap();
    let info = loaded.as_ref().ok_or_else(|| ModelError::ParseError {
        format: "inspect".into(),
        reason: "No model loaded".into(),
    })?;

    // Build a temporary ParentModel from the loaded ModelInfo
    let compat = crate::merge::registry::CompatInfo::from_model_info(info);
    let parent = crate::merge::registry::ParentModel {
        id: "inspect".into(),
        slot: 0,
        name: info.file_name.clone(),
        file_path: info.file_path.clone(),
        format: info.format.clone(),
        file_size: info.file_size,
        file_size_display: info.file_size_display.clone(),
        parameter_count: info.parameter_count,
        parameter_count_display: info.parameter_count_display.clone(),
        layer_count: info.layer_count,
        architecture: info.architecture.clone(),
        quantization: info.quantization.clone(),
        compat,
        color: "#f59e0b".into(),
        is_dir: info.shard_count.map_or(false, |s| s > 0),
    };

    Ok(crate::merge::capabilities::detect_capabilities(&parent))
}

// ── Fingerprint ────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFingerprint {
    pub sha256: String,
    pub file_size_bytes: u64,
    pub tensor_count_verified: bool,
}

#[tauri::command]
pub async fn compute_fingerprint(
    state: State<'_, AppState>,
) -> Result<ModelFingerprint, ModelError> {
    let file_path = {
        let loaded = state.loaded_model.lock().unwrap();
        let info = loaded.as_ref().ok_or_else(|| ModelError::ParseError {
            format: "fingerprint".into(),
            reason: "No model loaded".into(),
        })?;
        info.file_path.clone()
    };

    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut file = std::fs::File::open(&file_path)?;
        let file_size = file.metadata()?.len();

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        let hash = format!("{:x}", hasher.finalize());

        Ok::<_, std::io::Error>((hash, file_size))
    })
    .await
    .map_err(|e| ModelError::ParseError {
        format: "fingerprint".into(),
        reason: format!("Task failed: {}", e),
    })?
    .map_err(ModelError::IoError)?;

    Ok(ModelFingerprint {
        sha256: result.0,
        file_size_bytes: result.1,
        tensor_count_verified: true,
    })
}

// ── GPU Detection ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub has_nvidia: bool,
    pub nvidia_name: Option<String>,
    pub nvidia_vram: Option<String>,
    pub cuda_version: Option<String>,
    pub has_vulkan: bool,
    pub has_metal: bool,
    pub recommended_variant: String,
    pub os: String,
    pub arch: String,
}

fn probe_nvidia() -> (bool, Option<String>, Option<String>, Option<String>) {
    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let line = stdout.lines().next().unwrap_or("").trim().to_string();
            let parts: Vec<&str> = line.split(", ").collect();
            let name = parts.first().map(|s| s.trim().to_string());
            let vram = parts.get(1).map(|s| format!("{} MB", s.trim()));

            // Parse CUDA version from nvidia-smi header output
            let cuda_ver = std::process::Command::new("nvidia-smi")
                .output()
                .ok()
                .and_then(|o| {
                    let out = String::from_utf8_lossy(&o.stdout);
                    out.lines()
                        .find(|l| l.contains("CUDA Version"))
                        .and_then(|l| {
                            l.split("CUDA Version:")
                                .nth(1)
                                .map(|s| s.trim().split_whitespace().next().unwrap_or("").to_string())
                        })
                });

            (true, name, vram, cuda_ver)
        }
        _ => (false, None, None, None),
    }
}

fn probe_vulkan() -> bool {
    if let Ok(output) = std::process::Command::new("vulkaninfo")
        .arg("--summary")
        .output()
    {
        if output.status.success() {
            return true;
        }
    }

    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/usr/lib/libvulkan.so.1").exists()
            || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so.1").exists()
            || std::path::Path::new("/usr/lib64/libvulkan.so.1").exists()
        {
            return true;
        }
    }

    false
}

#[tauri::command]
pub fn detect_gpu() -> GpuInfo {
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    let (has_nvidia, nvidia_name, nvidia_vram, cuda_version) = probe_nvidia();
    let has_vulkan = probe_vulkan();
    let has_metal = cfg!(target_os = "macos");

    let recommended_variant = if has_nvidia {
        "cuda".to_string()
    } else if has_metal {
        "cpu".to_string() // macOS builds include Metal by default
    } else if has_vulkan {
        "vulkan".to_string()
    } else {
        "cpu".to_string()
    };

    GpuInfo {
        has_nvidia,
        nvidia_name,
        nvidia_vram,
        cuda_version,
        has_vulkan,
        has_metal,
        recommended_variant,
        os,
        arch,
    }
}

// ── Tools Management ───────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsStatus {
    pub installed: bool,
    pub version: Option<String>,
    pub variant: Option<String>,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolsManifest {
    version: String,
    variant: String,
    asset_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadResult {
    pub success: bool,
    pub version: String,
    pub variant: String,
    pub path: String,
}

fn get_tools_dir(app: &tauri::AppHandle) -> Result<PathBuf, ModelError> {
    let data_dir = app.path().app_data_dir().map_err(|e| ModelError::ParseError {
        format: "tools".into(),
        reason: format!("Cannot resolve app data dir: {}", e),
    })?;
    Ok(data_dir.join("tools").join("llama-cpp"))
}

fn quantize_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "llama-quantize.exe"
    } else {
        "llama-quantize"
    }
}

fn find_binary_recursive(dir: &std::path::Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_binary_recursive(&path, name) {
                return Some(found);
            }
        } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(path);
        }
    }
    None
}

/// Resolve the llama-quantize binary: bundled first, then PATH fallback.
fn resolve_quantize_binary(app: &tauri::AppHandle) -> PathBuf {
    let name = quantize_binary_name();
    if let Ok(tools_dir) = get_tools_dir(app) {
        if let Some(bundled) = find_binary_recursive(&tools_dir, name) {
            return bundled;
        }
    }
    PathBuf::from(name) // fallback to PATH
}

#[tauri::command]
pub fn get_tools_status(app: tauri::AppHandle) -> Result<ToolsStatus, ModelError> {
    let tools_dir = get_tools_dir(&app)?;

    // Read manifest if it exists
    let manifest_path = tools_dir.join("manifest.json");
    let manifest: Option<ToolsManifest> = std::fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    let name = quantize_binary_name();
    if let Some(binary_path) = find_binary_recursive(&tools_dir, name) {
        Ok(ToolsStatus {
            installed: true,
            version: manifest.as_ref().map(|m| m.version.clone()),
            variant: manifest.as_ref().map(|m| m.variant.clone()),
            path: Some(binary_path.to_string_lossy().to_string()),
        })
    } else {
        Ok(ToolsStatus {
            installed: false,
            version: None,
            variant: None,
            path: None,
        })
    }
}

// ── Download llama.cpp ─────────────────────────────────

#[derive(Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

fn match_asset<'a>(assets: &'a [GitHubAsset], variant: &str) -> Option<&'a GitHubAsset> {
    // Actual release names use "ubuntu" for Linux, "macos", "win" for Windows
    let os_patterns: &[&str] = match std::env::consts::OS {
        "linux" => &["ubuntu", "linux"],
        "macos" => &["macos"],
        "windows" => &["win"],
        _ => return None,
    };

    let arch_pattern = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        _ => return None,
    };

    // Filter to archive assets (.zip or .tar.gz) matching our OS and arch
    let matching: Vec<&GitHubAsset> = assets
        .iter()
        .filter(|a| {
            let n = a.name.to_lowercase();
            let is_archive = n.ends_with(".zip") || n.ends_with(".tar.gz");
            let matches_os = os_patterns.iter().any(|p| n.contains(p));
            let matches_arch = n.contains(arch_pattern);
            is_archive && matches_os && matches_arch
        })
        .collect();

    // Try to find variant-specific build
    if variant == "cuda" {
        if let Some(a) = matching.iter().find(|a| a.name.to_lowercase().contains("cuda")) {
            return Some(a);
        }
        // CUDA builds only exist for Windows; on Linux fall through to vulkan then cpu
        if let Some(a) = matching.iter().find(|a| a.name.to_lowercase().contains("vulkan")) {
            return Some(a);
        }
    }
    if variant == "vulkan" {
        if let Some(a) = matching.iter().find(|a| a.name.to_lowercase().contains("vulkan")) {
            return Some(a);
        }
    }

    // CPU fallback: pick the one without accelerator-specific keywords
    matching
        .iter()
        .find(|a| {
            let n = a.name.to_lowercase();
            !n.contains("cuda") && !n.contains("vulkan") && !n.contains("rocm")
                && !n.contains("hip") && !n.contains("sycl") && !n.contains("opencl")
        })
        .copied()
}

#[tauri::command]
pub async fn download_llama_cpp(
    variant: String,
    app: tauri::AppHandle,
) -> Result<DownloadResult, ModelError> {
    let tools_dir = get_tools_dir(&app)?;

    // 1. Fetch latest release metadata from GitHub
    let client = reqwest::Client::builder()
        .user_agent("ForgeAI")
        .build()
        .map_err(|e| ModelError::ParseError {
            format: "download".into(),
            reason: format!("HTTP client error: {}", e),
        })?;

    let release: GitHubRelease = client
        .get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
        .send()
        .await
        .map_err(|e| ModelError::ParseError {
            format: "download".into(),
            reason: format!("Failed to fetch releases: {}", e),
        })?
        .json()
        .await
        .map_err(|e| ModelError::ParseError {
            format: "download".into(),
            reason: format!("Failed to parse release JSON: {}", e),
        })?;

    // 2. Find matching asset for this platform + variant
    let asset = match_asset(&release.assets, &variant).ok_or_else(|| ModelError::ParseError {
        format: "download".into(),
        reason: format!(
            "No compatible build found for {} / {} / {}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            variant
        ),
    })?;

    let asset_name = asset.name.clone();
    let download_url = asset.browser_download_url.clone();
    let version = release.tag_name.clone();

    // 3. Download the zip
    let zip_bytes = client
        .get(&download_url)
        .send()
        .await
        .map_err(|e| ModelError::ParseError {
            format: "download".into(),
            reason: format!("Download failed: {}", e),
        })?
        .bytes()
        .await
        .map_err(|e| ModelError::ParseError {
            format: "download".into(),
            reason: format!("Download failed: {}", e),
        })?;

    // 4. Extract on a blocking thread
    let td = tools_dir.clone();
    let ver = version.clone();
    let var = variant.clone();
    let aname = asset_name.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        // Clean previous installation
        if td.exists() {
            std::fs::remove_dir_all(&td).map_err(|e| format!("Cleanup failed: {}", e))?;
        }
        std::fs::create_dir_all(&td).map_err(|e| format!("Create dir failed: {}", e))?;

        // Extract archive (zip or tar.gz)
        if aname.ends_with(".tar.gz") {
            let cursor = std::io::Cursor::new(&zip_bytes);
            let gz = flate2::read::GzDecoder::new(cursor);
            let mut archive = tar::Archive::new(gz);

            for entry in archive.entries().map_err(|e| format!("Invalid tar.gz: {}", e))? {
                let mut entry = entry.map_err(|e| format!("Tar entry error: {}", e))?;
                let path = entry.path().map_err(|e| format!("Tar path error: {}", e))?.into_owned();
                let path_str = path.to_string_lossy().to_string();

                // Skip path-traversal attempts
                if path_str.contains("..") {
                    continue;
                }

                let outpath = td.join(&path);
                let entry_type = entry.header().entry_type();

                if entry_type.is_dir() {
                    std::fs::create_dir_all(&outpath)
                        .map_err(|e| format!("mkdir failed: {}", e))?;
                } else if entry_type.is_symlink() || entry_type.is_hard_link() {
                    // Handle symlinks (e.g. libllama.so -> libllama.so.0.0.7974)
                    if let Some(parent) = outpath.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| format!("mkdir failed: {}", e))?;
                    }
                    if let Ok(link_name) = entry.link_name() {
                        if let Some(target) = link_name {
                            let _ = std::fs::remove_file(&outpath);
                            #[cfg(unix)]
                            {
                                std::os::unix::fs::symlink(target.as_ref(), &outpath)
                                    .map_err(|e| format!("Symlink failed: {}", e))?;
                            }
                            #[cfg(not(unix))]
                            {
                                // On non-Unix, copy the resolved target file
                                let resolved = if target.is_relative() {
                                    outpath.parent().unwrap_or(&td).join(&target)
                                } else {
                                    target.into_owned()
                                };
                                let _ = std::fs::copy(&resolved, &outpath);
                            }
                        }
                    }
                } else {
                    if let Some(parent) = outpath.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| format!("mkdir failed: {}", e))?;
                    }
                    let mut outfile = std::fs::File::create(&outpath)
                        .map_err(|e| format!("Create file failed: {}", e))?;
                    std::io::copy(&mut entry, &mut outfile)
                        .map_err(|e| format!("Extract failed: {}", e))?;
                }
            }
        } else {
            let cursor = std::io::Cursor::new(&zip_bytes);
            let mut archive =
                zip::ZipArchive::new(cursor).map_err(|e| format!("Invalid zip: {}", e))?;

            for i in 0..archive.len() {
                let mut file = archive
                    .by_index(i)
                    .map_err(|e| format!("Zip entry error: {}", e))?;

                let name = file.name().to_string();
                // Skip path-traversal attempts
                if name.contains("..") {
                    continue;
                }

                let outpath = td.join(&name);

                if file.is_dir() {
                    std::fs::create_dir_all(&outpath)
                        .map_err(|e| format!("mkdir failed: {}", e))?;
                } else {
                    if let Some(parent) = outpath.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| format!("mkdir failed: {}", e))?;
                    }
                    let mut outfile = std::fs::File::create(&outpath)
                        .map_err(|e| format!("Create file failed: {}", e))?;
                    std::io::copy(&mut file, &mut outfile)
                        .map_err(|e| format!("Extract failed: {}", e))?;
                }
            }
        }

        // Set executable permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let executables = [
                "llama-quantize",
                "llama-cli",
                "llama-server",
                "llama-perplexity",
                "llama-bench",
            ];
            for exe in &executables {
                if let Some(path) = find_binary_recursive(&td, exe) {
                    let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755));
                }
            }
        }

        // Save manifest
        let manifest = ToolsManifest {
            version: ver,
            variant: var,
            asset_name: aname,
        };
        let json =
            serde_json::to_string_pretty(&manifest).map_err(|e| format!("JSON error: {}", e))?;
        std::fs::write(td.join("manifest.json"), json)
            .map_err(|e| format!("Write manifest failed: {}", e))?;

        Ok(())
    })
    .await
    .map_err(|e| ModelError::ParseError {
        format: "download".into(),
        reason: format!("Task join error: {}", e),
    })?
    .map_err(|e| ModelError::ParseError {
        format: "download".into(),
        reason: e,
    })?;

    // 5. Verify the binary exists
    let name = quantize_binary_name();
    let binary_path =
        find_binary_recursive(&tools_dir, name).ok_or_else(|| ModelError::ParseError {
            format: "download".into(),
            reason: "llama-quantize not found in downloaded archive".into(),
        })?;

    Ok(DownloadResult {
        success: true,
        version,
        variant,
        path: binary_path.to_string_lossy().to_string(),
    })
}

#[tauri::command]
pub async fn remove_tools(app: tauri::AppHandle) -> Result<(), ModelError> {
    let tools_dir = get_tools_dir(&app)?;
    if tools_dir.exists() {
        std::fs::remove_dir_all(&tools_dir).map_err(ModelError::IoError)?;
    }
    Ok(())
}

// ── Model Hub ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfRepoInfo {
    pub id: String,
    pub files: Vec<HfFileInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfFileInfo {
    pub rfilename: String,
    pub size: Option<u64>,
    pub size_display: String,
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelEntry {
    pub id: String,
    pub file_name: String,
    pub file_path: String,
    pub file_size: u64,
    pub file_size_display: String,
    pub format: String,
    pub source_repo: Option<String>,
    pub downloaded_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub file_name: String,
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    pub percent: f64,
    pub status: String,
    pub files_done: Option<u32>,
    pub files_total: Option<u32>,
}

fn get_models_dir(app: &tauri::AppHandle) -> Result<PathBuf, ModelError> {
    let data_dir = app.path().app_data_dir().map_err(|e| ModelError::ParseError {
        format: "hub".into(),
        reason: format!("Cannot resolve app data dir: {}", e),
    })?;
    Ok(data_dir.join("models"))
}

fn read_manifest(models_dir: &std::path::Path) -> Vec<LocalModelEntry> {
    let manifest_path = models_dir.join("manifest.json");
    std::fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_manifest(models_dir: &std::path::Path, entries: &[LocalModelEntry]) -> Result<(), ModelError> {
    std::fs::create_dir_all(models_dir).map_err(ModelError::IoError)?;
    let json = serde_json::to_string_pretty(entries).map_err(|e| ModelError::ParseError {
        format: "hub".into(),
        reason: format!("JSON serialize error: {}", e),
    })?;
    std::fs::write(models_dir.join("manifest.json"), json).map_err(ModelError::IoError)?;
    Ok(())
}

fn detect_file_format(filename: &str) -> Option<String> {
    let lower = filename.to_lowercase();
    if lower.ends_with(".gguf") {
        Some("gguf".into())
    } else if lower.ends_with(".safetensors") {
        Some("safetensors".into())
    } else {
        None
    }
}

#[derive(Deserialize)]
struct HfApiSibling {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Deserialize)]
struct HfApiResponse {
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    id: Option<String>,
    siblings: Option<Vec<HfApiSibling>>,
}

async fn fetch_repo_info(repo_id: &str) -> Result<HfRepoInfo, ModelError> {
    let client = reqwest::Client::builder()
        .user_agent("ForgeAI")
        .build()
        .map_err(|e| ModelError::ParseError {
            format: "hub".into(),
            reason: format!("HTTP client error: {}", e),
        })?;

    let url = format!("https://huggingface.co/api/models/{}", repo_id);
    let resp = client.get(&url).send().await.map_err(|e| ModelError::ParseError {
        format: "hub".into(),
        reason: format!("Failed to fetch repo: {}", e),
    })?;

    if !resp.status().is_success() {
        return Err(ModelError::ParseError {
            format: "hub".into(),
            reason: format!("Repository not found or inaccessible (HTTP {})", resp.status()),
        });
    }

    let api_resp: HfApiResponse = resp.json().await.map_err(|e| ModelError::ParseError {
        format: "hub".into(),
        reason: format!("Failed to parse response: {}", e),
    })?;

    let repo_name = api_resp.model_id.or(api_resp.id).unwrap_or_else(|| repo_id.to_string());
    let siblings = api_resp.siblings.unwrap_or_default();

    let mut files: Vec<HfFileInfo> = siblings
        .into_iter()
        .map(|s| {
            let format = detect_file_format(&s.rfilename);
            let size_display = s.size.map(|sz| crate::model::format_file_size(sz)).unwrap_or_else(|| "---".into());
            HfFileInfo {
                rfilename: s.rfilename,
                size: s.size,
                size_display,
                format,
            }
        })
        .collect();

    // Sort: model files first (gguf, safetensors), then by size descending
    files.sort_by(|a, b| {
        let a_is_model = a.format.is_some();
        let b_is_model = b.format.is_some();
        b_is_model.cmp(&a_is_model)
            .then_with(|| b.size.unwrap_or(0).cmp(&a.size.unwrap_or(0)))
    });

    Ok(HfRepoInfo {
        id: repo_name,
        files,
    })
}

fn build_http_client() -> Result<reqwest::Client, ModelError> {
    reqwest::Client::builder()
        .user_agent("ForgeAI")
        .build()
        .map_err(|e| ModelError::ParseError {
            format: "hub".into(),
            reason: format!("HTTP client error: {}", e),
        })
}

fn calculate_dir_size(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += calculate_dir_size(&path);
            } else if let Ok(meta) = path.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

#[tauri::command]
pub async fn hf_fetch_repo(repo_id: String) -> Result<HfRepoInfo, ModelError> {
    fetch_repo_info(&repo_id).await
}

#[tauri::command]
pub async fn hf_download_file(
    repo_id: String,
    filename: String,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<LocalModelEntry, ModelError> {
    let models_dir = get_models_dir(&app)?;
    std::fs::create_dir_all(&models_dir).map_err(ModelError::IoError)?;

    // Reset cancel flag
    let cancel = state.download_cancel.clone();
    cancel.store(false, std::sync::atomic::Ordering::Relaxed);

    let download_url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    );

    let client = build_http_client()?;

    let resp = client.get(&download_url).send().await.map_err(|e| ModelError::ParseError {
        format: "hub".into(),
        reason: format!("Download request failed: {}", e),
    })?;

    if !resp.status().is_success() {
        return Err(ModelError::ParseError {
            format: "hub".into(),
            reason: format!("Download failed (HTTP {})", resp.status()),
        });
    }

    let total_size = resp.content_length().unwrap_or(0);

    // Sanitize filename: take only the last path component
    let safe_name = std::path::Path::new(&filename)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&filename)
        .to_string();
    let file_path = models_dir.join(&safe_name);
    let partial_path = models_dir.join(format!("{}.partial", safe_name));

    let mut file = std::fs::File::create(&partial_path).map_err(ModelError::IoError)?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    while let Some(chunk) = stream.next().await {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(&partial_path);
            let _ = app.emit("hub:download-progress", DownloadProgress {
                file_name: safe_name.clone(),
                bytes_downloaded: downloaded,
                bytes_total: total_size,
                percent: 0.0,
                status: "cancelled".into(),
                files_done: None,
                files_total: None,
            });
            return Err(ModelError::ParseError {
                format: "hub".into(),
                reason: "Download cancelled".into(),
            });
        }

        let bytes = chunk.map_err(|e| ModelError::ParseError {
            format: "hub".into(),
            reason: format!("Download stream error: {}", e),
        })?;

        {
            use std::io::Write;
            file.write_all(&bytes).map_err(ModelError::IoError)?;
        }
        downloaded += bytes.len() as u64;

        let now = std::time::Instant::now();
        if now.duration_since(last_emit).as_millis() >= 500 || (total_size > 0 && downloaded >= total_size) {
            let percent = if total_size > 0 {
                (downloaded as f64 / total_size as f64) * 100.0
            } else {
                0.0
            };
            let _ = app.emit("hub:download-progress", DownloadProgress {
                file_name: safe_name.clone(),
                bytes_downloaded: downloaded,
                bytes_total: total_size,
                percent,
                status: "downloading".into(),
                files_done: None,
                files_total: None,
            });
            last_emit = now;
        }
    }

    drop(file);

    // Rename partial to final
    std::fs::rename(&partial_path, &file_path).map_err(ModelError::IoError)?;

    let file_size = std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
    let format = detect_file_format(&safe_name).unwrap_or_else(|| "unknown".into());
    let id = format!(
        "{}-{}",
        safe_name.replace('.', "-"),
        chrono::Utc::now().timestamp()
    );

    let entry = LocalModelEntry {
        id,
        file_name: safe_name.clone(),
        file_path: file_path.to_string_lossy().to_string(),
        file_size,
        file_size_display: crate::model::format_file_size(file_size),
        format,
        source_repo: Some(repo_id),
        downloaded_at: chrono::Utc::now().to_rfc3339(),
    };

    // Update manifest
    let mut manifest = read_manifest(&models_dir);
    manifest.push(entry.clone());
    write_manifest(&models_dir, &manifest)?;

    // Emit completion
    let _ = app.emit("hub:download-progress", DownloadProgress {
        file_name: safe_name,
        bytes_downloaded: file_size,
        bytes_total: file_size,
        percent: 100.0,
        status: "complete".into(),
        files_done: None,
        files_total: None,
    });

    Ok(entry)
}

#[tauri::command]
pub async fn hf_download_repo(
    repo_id: String,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<LocalModelEntry, ModelError> {
    let models_dir = get_models_dir(&app)?;
    let repo_dir_name = repo_id.replace('/', "--");
    let repo_dir = models_dir.join(&repo_dir_name);
    std::fs::create_dir_all(&repo_dir).map_err(ModelError::IoError)?;

    // Reset cancel flag
    let cancel = state.download_cancel.clone();
    cancel.store(false, std::sync::atomic::Ordering::Relaxed);

    // Fetch repo file listing
    let repo_info = fetch_repo_info(&repo_id).await?;

    let total_size: u64 = repo_info.files.iter().filter_map(|f| f.size).sum();
    let total_files = repo_info.files.len() as u32;
    let mut overall_downloaded: u64 = 0;

    let client = build_http_client()?;

    for (idx, file_info) in repo_info.files.iter().enumerate() {
        // Check cancel before each file
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = app.emit("hub:download-progress", DownloadProgress {
                file_name: repo_dir_name.clone(),
                bytes_downloaded: overall_downloaded,
                bytes_total: total_size,
                percent: 0.0,
                status: "cancelled".into(),
                files_done: Some(idx as u32),
                files_total: Some(total_files),
            });
            return Err(ModelError::ParseError {
                format: "hub".into(),
                reason: "Download cancelled".into(),
            });
        }

        let download_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, file_info.rfilename
        );

        // Preserve directory structure within the repo folder
        let out_path = repo_dir.join(&file_info.rfilename);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).map_err(ModelError::IoError)?;
        }

        let resp = client.get(&download_url).send().await.map_err(|e| ModelError::ParseError {
            format: "hub".into(),
            reason: format!("Failed to download {}: {}", file_info.rfilename, e),
        })?;

        if !resp.status().is_success() {
            // Skip files that can't be fetched (e.g., LFS pointers without auth)
            continue;
        }

        let mut outfile = std::fs::File::create(&out_path).map_err(ModelError::IoError)?;
        let mut stream = resp.bytes_stream();
        let mut last_emit = std::time::Instant::now();

        while let Some(chunk) = stream.next().await {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                drop(outfile);
                let _ = std::fs::remove_file(&out_path);
                let _ = app.emit("hub:download-progress", DownloadProgress {
                    file_name: repo_dir_name.clone(),
                    bytes_downloaded: overall_downloaded,
                    bytes_total: total_size,
                    percent: 0.0,
                    status: "cancelled".into(),
                    files_done: Some(idx as u32),
                    files_total: Some(total_files),
                });
                return Err(ModelError::ParseError {
                    format: "hub".into(),
                    reason: "Download cancelled".into(),
                });
            }

            let bytes = chunk.map_err(|e| ModelError::ParseError {
                format: "hub".into(),
                reason: format!("Stream error for {}: {}", file_info.rfilename, e),
            })?;

            {
                use std::io::Write;
                outfile.write_all(&bytes).map_err(ModelError::IoError)?;
            }
            overall_downloaded += bytes.len() as u64;

            let now = std::time::Instant::now();
            if now.duration_since(last_emit).as_millis() >= 500 {
                let percent = if total_size > 0 {
                    (overall_downloaded as f64 / total_size as f64) * 100.0
                } else {
                    ((idx as f64 + 0.5) / total_files as f64) * 100.0
                };
                let _ = app.emit("hub:download-progress", DownloadProgress {
                    file_name: file_info.rfilename.clone(),
                    bytes_downloaded: overall_downloaded,
                    bytes_total: total_size,
                    percent,
                    status: "downloading".into(),
                    files_done: Some(idx as u32),
                    files_total: Some(total_files),
                });
                last_emit = now;
            }
        }

        // Per-file completion emit
        let percent = if total_size > 0 {
            (overall_downloaded as f64 / total_size as f64) * 100.0
        } else {
            ((idx + 1) as f64 / total_files as f64) * 100.0
        };
        let _ = app.emit("hub:download-progress", DownloadProgress {
            file_name: file_info.rfilename.clone(),
            bytes_downloaded: overall_downloaded,
            bytes_total: total_size,
            percent,
            status: "downloading".into(),
            files_done: Some((idx + 1) as u32),
            files_total: Some(total_files),
        });
    }

    // Calculate total size of downloaded directory
    let dir_size = calculate_dir_size(&repo_dir);

    let entry = LocalModelEntry {
        id: format!("{}-{}", repo_dir_name, chrono::Utc::now().timestamp()),
        file_name: repo_dir_name.clone(),
        file_path: repo_dir.to_string_lossy().to_string(),
        file_size: dir_size,
        file_size_display: crate::model::format_file_size(dir_size),
        format: "repo".into(),
        source_repo: Some(repo_id),
        downloaded_at: chrono::Utc::now().to_rfc3339(),
    };

    // Update manifest
    let mut manifest = read_manifest(&models_dir);
    manifest.push(entry.clone());
    write_manifest(&models_dir, &manifest)?;

    // Emit completion
    let _ = app.emit("hub:download-progress", DownloadProgress {
        file_name: repo_dir_name,
        bytes_downloaded: dir_size,
        bytes_total: dir_size,
        percent: 100.0,
        status: "complete".into(),
        files_done: Some(total_files),
        files_total: Some(total_files),
    });

    Ok(entry)
}

#[tauri::command]
pub async fn hub_list_local(app: tauri::AppHandle) -> Result<Vec<LocalModelEntry>, ModelError> {
    let models_dir = get_models_dir(&app)?;
    let mut manifest = read_manifest(&models_dir);

    // Filter out entries whose files/directories no longer exist
    manifest.retain(|e| std::path::Path::new(&e.file_path).exists());

    // Rewrite cleaned manifest
    write_manifest(&models_dir, &manifest)?;

    Ok(manifest)
}

#[tauri::command]
pub async fn hub_delete_model(model_id: String, app: tauri::AppHandle) -> Result<(), ModelError> {
    let models_dir = get_models_dir(&app)?;
    let mut manifest = read_manifest(&models_dir);

    let idx = manifest.iter().position(|e| e.id == model_id);
    if let Some(idx) = idx {
        let entry = manifest.remove(idx);
        let path = std::path::Path::new(&entry.file_path);
        if path.exists() {
            if path.is_dir() {
                std::fs::remove_dir_all(path).map_err(ModelError::IoError)?;
            } else {
                std::fs::remove_file(path).map_err(ModelError::IoError)?;
            }
        }
        write_manifest(&models_dir, &manifest)?;
    }

    Ok(())
}

#[tauri::command]
pub async fn hub_cancel_download(state: State<'_, AppState>) -> Result<(), ModelError> {
    state.download_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
pub async fn hub_import_local(path: String, app: tauri::AppHandle) -> Result<LocalModelEntry, ModelError> {
    let src = PathBuf::from(&path);
    if !src.exists() {
        return Err(ModelError::FileNotFound(path));
    }

    let models_dir = get_models_dir(&app)?;
    std::fs::create_dir_all(&models_dir).map_err(ModelError::IoError)?;

    let is_dir = src.is_dir();

    // Determine format and file info
    let (file_name, _file_size, format) = if is_dir {
        // Folder import: check for .safetensors files inside
        let st_count = std::fs::read_dir(&src)
            .map_err(ModelError::IoError)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.to_lowercase() == "safetensors")
                    .unwrap_or(false)
            })
            .count();

        if st_count == 0 {
            return Err(ModelError::ParseError {
                format: "hub".into(),
                reason: "No .safetensors files found in directory".into(),
            });
        }

        // Calculate total size of directory
        let total_size: u64 = walkdir(&src);

        let dir_name = src
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        (dir_name, total_size, "safetensors".to_string())
    } else {
        // Single file import
        let fname = src
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let fsize = std::fs::metadata(&src).map_err(ModelError::IoError)?.len();
        let fmt = detect_file_format(&fname).ok_or_else(|| ModelError::ParseError {
            format: "hub".into(),
            reason: "Unsupported file format. Use .gguf or .safetensors files".into(),
        })?;
        (fname, fsize, fmt)
    };

    // Copy to models directory
    let dest = models_dir.join(&file_name);

    if is_dir {
        // Copy directory recursively
        copy_dir_recursive(&src, &dest)?;
    } else {
        std::fs::copy(&src, &dest).map_err(ModelError::IoError)?;
    }

    let dest_size = if dest.is_dir() { walkdir(&dest) } else { std::fs::metadata(&dest).map_err(ModelError::IoError)?.len() };

    let entry = LocalModelEntry {
        id: format!("import-{}", chrono::Utc::now().timestamp_millis()),
        file_name: file_name.clone(),
        file_path: dest.to_string_lossy().to_string(),
        file_size: dest_size,
        file_size_display: crate::model::format_file_size(dest_size),
        format,
        source_repo: Some("local import".to_string()),
        downloaded_at: chrono::Utc::now().to_rfc3339(),
    };

    let mut manifest = read_manifest(&models_dir);
    manifest.push(entry.clone());
    write_manifest(&models_dir, &manifest)?;

    Ok(entry)
}

fn walkdir(dir: &std::path::Path) -> u64 {
    let mut total: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                total += walkdir(&path);
            } else if let Ok(meta) = std::fs::metadata(&path) {
                total += meta.len();
            }
        }
    }
    total
}

fn copy_dir_recursive(src: &std::path::Path, dest: &std::path::Path) -> Result<(), ModelError> {
    std::fs::create_dir_all(dest).map_err(ModelError::IoError)?;
    for entry in std::fs::read_dir(src).map_err(ModelError::IoError)? {
        let entry = entry.map_err(ModelError::IoError)?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dest_path)?;
        } else {
            std::fs::copy(&src_path, &dest_path).map_err(ModelError::IoError)?;
        }
    }
    Ok(())
}

// ── SafeTensors to GGUF Conversion ─────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertDepsStatus {
    pub python_found: bool,
    pub python_version: Option<String>,
    pub python_path: Option<String>,
    pub venv_ready: bool,
    pub script_ready: bool,
    pub packages_ready: bool,
    pub missing_packages: Vec<String>,
    pub ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertModelInfo {
    pub repo_path: String,
    pub architectures: Vec<String>,
    pub model_type: Option<String>,
    pub hidden_size: Option<u64>,
    pub num_layers: Option<u64>,
    pub vocab_size: Option<u64>,
    pub has_tokenizer: bool,
    pub has_tokenizer_model: bool,
    pub has_config: bool,
    pub safetensor_count: u32,
    pub total_size: u64,
    pub total_size_display: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertProgress {
    pub stage: String,
    pub message: String,
    pub percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertResult {
    pub output_path: String,
    pub output_size: u64,
    pub output_size_display: String,
}

fn get_convert_dir(app: &tauri::AppHandle) -> Result<PathBuf, ModelError> {
    let data_dir = app.path().app_data_dir().map_err(|e| ModelError::ParseError {
        format: "convert".into(),
        reason: format!("Cannot resolve app data dir: {}", e),
    })?;
    Ok(data_dir.join("tools").join("convert"))
}

pub(crate) fn find_python() -> Option<(String, String)> {
    let candidates = if cfg!(target_os = "windows") {
        vec!["python", "python3"]
    } else {
        vec!["python3", "python"]
    };

    for cmd in candidates {
        if let Ok(output) = std::process::Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let version = if version.is_empty() {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                } else {
                    version
                };
                if version.contains("3.") {
                    return Some((cmd.to_string(), version));
                }
            }
        }
    }
    None
}

fn get_venv_python(convert_dir: &std::path::Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        convert_dir.join("venv").join("Scripts").join("python.exe")
    } else {
        convert_dir.join("venv").join("bin").join("python3")
    }
}

fn get_script_path(convert_dir: &std::path::Path) -> PathBuf {
    convert_dir.join("convert_hf_to_gguf.py")
}

fn check_packages(venv_python: &std::path::Path) -> (bool, Vec<String>) {
    let required = ["gguf", "numpy", "sentencepiece", "transformers", "safetensors"];
    let mut missing = Vec::new();

    for pkg in &required {
        let output = std::process::Command::new(venv_python)
            .args(["-c", &format!("import {}", pkg)])
            .output();

        match output {
            Ok(o) if o.status.success() => {}
            _ => missing.push(pkg.to_string()),
        }
    }

    (missing.is_empty(), missing)
}

fn parse_convert_progress(line: &str) -> ConvertProgress {
    let line_lower = line.to_lowercase();

    // Try to extract percentage from patterns like "45%" or "100%|"
    if let Some(pct_pos) = line.find('%') {
        let before = &line[..pct_pos];
        let num_str: String = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        if let Ok(pct) = num_str.parse::<f64>() {
            let stage = if line_lower.contains("loading") || line_lower.contains("reading") {
                "loading"
            } else if line_lower.contains("writing") {
                "writing"
            } else {
                "converting"
            };
            return ConvertProgress {
                stage: stage.into(),
                message: line.trim().chars().take(200).collect(),
                percent: pct.min(100.0),
            };
        }
    }

    let stage = if line_lower.contains("loading") || line_lower.contains("reading") {
        "loading"
    } else if line_lower.contains("writing") || line_lower.contains("wrote") {
        "writing"
    } else if line_lower.contains("error") || line_lower.contains("exception") || line_lower.contains("traceback") {
        "error"
    } else if line_lower.contains("token") {
        "tokenizer"
    } else {
        "converting"
    };

    ConvertProgress {
        stage: stage.into(),
        message: line.trim().chars().take(200).collect(),
        percent: -1.0,
    }
}

#[tauri::command]
pub async fn convert_check_deps(app: tauri::AppHandle) -> Result<ConvertDepsStatus, ModelError> {
    let convert_dir = get_convert_dir(&app)?;

    let (python_found, python_version, python_path) = match find_python() {
        Some((path, ver)) => (true, Some(ver), Some(path)),
        None => (false, None, None),
    };

    let venv_python = get_venv_python(&convert_dir);
    let venv_ready = venv_python.exists();

    let script_path = get_script_path(&convert_dir);
    let script_ready = script_path.exists();

    let (packages_ready, missing_packages) = if venv_ready {
        check_packages(&venv_python)
    } else {
        (
            false,
            vec![
                "gguf".into(),
                "numpy".into(),
                "sentencepiece".into(),
                "transformers".into(),
                "safetensors".into(),
            ],
        )
    };

    let ready = python_found && venv_ready && script_ready && packages_ready;

    Ok(ConvertDepsStatus {
        python_found,
        python_version,
        python_path,
        venv_ready,
        script_ready,
        packages_ready,
        missing_packages,
        ready,
    })
}

#[tauri::command]
pub async fn convert_setup(app: tauri::AppHandle) -> Result<(), ModelError> {
    let convert_dir = get_convert_dir(&app)?;
    std::fs::create_dir_all(&convert_dir).map_err(ModelError::IoError)?;

    let (python_cmd, _) = find_python().ok_or_else(|| ModelError::ParseError {
        format: "convert".into(),
        reason: "Python 3 not found. Please install Python 3.10+.".into(),
    })?;

    // 1. Create venv
    let _ = app.emit(
        "convert:setup-progress",
        ConvertProgress {
            stage: "setup".into(),
            message: "Creating Python environment...".into(),
            percent: 10.0,
        },
    );

    let venv_dir = convert_dir.join("venv");
    if !venv_dir.exists() {
        let status = tokio::process::Command::new(&python_cmd)
            .args(["-m", "venv", &venv_dir.to_string_lossy()])
            .status()
            .await
            .map_err(|e| ModelError::ParseError {
                format: "convert".into(),
                reason: format!("Failed to create venv: {}", e),
            })?;

        if !status.success() {
            return Err(ModelError::ParseError {
                format: "convert".into(),
                reason: "Failed to create Python venv. Ensure python3-venv is installed.".into(),
            });
        }
    }

    let venv_python = get_venv_python(&convert_dir);

    // 2. Install PyTorch (GPU-accelerated if NVIDIA detected, CPU fallback)
    let gpu = detect_gpu();
    let (torch_msg, torch_args): (&str, Vec<&str>) = if gpu.has_nvidia {
        (
            "Installing PyTorch with CUDA (GPU accelerated, this may take several minutes)...",
            vec!["-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu124", "--quiet", "--disable-pip-version-check"],
        )
    } else {
        (
            "Installing PyTorch CPU (this may take several minutes)...",
            vec!["-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet", "--disable-pip-version-check"],
        )
    };

    let _ = app.emit(
        "convert:setup-progress",
        ConvertProgress {
            stage: "setup".into(),
            message: torch_msg.into(),
            percent: 25.0,
        },
    );

    let status = tokio::process::Command::new(&venv_python)
        .args(&torch_args)
        .status()
        .await
        .map_err(|e| ModelError::ParseError {
            format: "convert".into(),
            reason: format!("Failed to install PyTorch: {}", e),
        })?;

    if !status.success() {
        // CUDA install failed — fall back to CPU
        if gpu.has_nvidia {
            let _ = app.emit(
                "convert:setup-progress",
                ConvertProgress {
                    stage: "setup".into(),
                    message: "CUDA install failed, falling back to PyTorch CPU...".into(),
                    percent: 30.0,
                },
            );
            let fallback = tokio::process::Command::new(&venv_python)
                .args(["-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet", "--disable-pip-version-check"])
                .status()
                .await
                .map_err(|e| ModelError::ParseError {
                    format: "convert".into(),
                    reason: format!("Failed to install PyTorch: {}", e),
                })?;
            if !fallback.success() {
                return Err(ModelError::ParseError {
                    format: "convert".into(),
                    reason: "Failed to install PyTorch. Check your internet connection.".into(),
                });
            }
        } else {
            return Err(ModelError::ParseError {
                format: "convert".into(),
                reason: "Failed to install PyTorch CPU. Check your internet connection.".into(),
            });
        }
    }

    // 3. Install gguf from llama.cpp source (must match the convert script version)
    let _ = app.emit(
        "convert:setup-progress",
        ConvertProgress {
            stage: "setup".into(),
            message: "Removing old gguf package...".into(),
            percent: 45.0,
        },
    );

    // Uninstall any existing gguf from PyPI first
    let _ = tokio::process::Command::new(&venv_python)
        .args(["-m", "pip", "uninstall", "gguf", "-y", "--quiet", "--disable-pip-version-check"])
        .status()
        .await;

    let _ = app.emit(
        "convert:setup-progress",
        ConvertProgress {
            stage: "setup".into(),
            message: "Installing gguf package from llama.cpp source...".into(),
            percent: 50.0,
        },
    );

    let gguf_url = "https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz#subdirectory=gguf-py";
    let status = tokio::process::Command::new(&venv_python)
        .args(["-m", "pip", "install", gguf_url, "--no-cache-dir", "--quiet", "--disable-pip-version-check"])
        .status()
        .await
        .map_err(|e| ModelError::ParseError {
            format: "convert".into(),
            reason: format!("Failed to install gguf from source: {}", e),
        })?;

    if !status.success() {
        return Err(ModelError::ParseError {
            format: "convert".into(),
            reason: "Failed to install gguf package from llama.cpp source.".into(),
        });
    }

    // 4. Install other packages
    let _ = app.emit(
        "convert:setup-progress",
        ConvertProgress {
            stage: "setup".into(),
            message: "Installing conversion dependencies...".into(),
            percent: 70.0,
        },
    );

    let status = tokio::process::Command::new(&venv_python)
        .args([
            "-m", "pip", "install",
            "numpy", "sentencepiece", "transformers", "safetensors", "protobuf",
            "--quiet", "--disable-pip-version-check",
        ])
        .status()
        .await
        .map_err(|e| ModelError::ParseError {
            format: "convert".into(),
            reason: format!("Failed to install packages: {}", e),
        })?;

    if !status.success() {
        return Err(ModelError::ParseError {
            format: "convert".into(),
            reason: "Failed to install Python packages.".into(),
        });
    }

    // 5. Download convert script from llama.cpp
    let _ = app.emit(
        "convert:setup-progress",
        ConvertProgress {
            stage: "setup".into(),
            message: "Downloading conversion script...".into(),
            percent: 90.0,
        },
    );

    let script_url =
        "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py";
    let client = build_http_client()?;
    let resp = client
        .get(script_url)
        .send()
        .await
        .map_err(|e| ModelError::ParseError {
            format: "convert".into(),
            reason: format!("Failed to download convert script: {}", e),
        })?;

    if !resp.status().is_success() {
        return Err(ModelError::ParseError {
            format: "convert".into(),
            reason: format!(
                "Failed to download convert script (HTTP {})",
                resp.status()
            ),
        });
    }

    let script_bytes = resp.bytes().await.map_err(|e| ModelError::ParseError {
        format: "convert".into(),
        reason: format!("Download error: {}", e),
    })?;

    std::fs::write(get_script_path(&convert_dir), &script_bytes).map_err(ModelError::IoError)?;

    let _ = app.emit(
        "convert:setup-progress",
        ConvertProgress {
            stage: "done".into(),
            message: "Setup complete!".into(),
            percent: 100.0,
        },
    );

    Ok(())
}

#[tauri::command]
pub async fn convert_detect_model(repo_path: String) -> Result<ConvertModelInfo, ModelError> {
    let repo = PathBuf::from(&repo_path);

    if !repo.exists() || !repo.is_dir() {
        return Err(ModelError::ParseError {
            format: "convert".into(),
            reason: "Path does not exist or is not a directory".into(),
        });
    }

    // Read config.json
    let config_path = repo.join("config.json");
    let has_config = config_path.exists();

    let (architectures, model_type, hidden_size, num_layers, vocab_size) = if has_config {
        let config_str = std::fs::read_to_string(&config_path).map_err(ModelError::IoError)?;
        let config: serde_json::Value =
            serde_json::from_str(&config_str).map_err(|e| ModelError::ParseError {
                format: "convert".into(),
                reason: format!("Invalid config.json: {}", e),
            })?;

        let archs = config
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let mt = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(String::from);
        let hs = config.get("hidden_size").and_then(|v| v.as_u64());
        let nl = config.get("num_hidden_layers").and_then(|v| v.as_u64());
        let vs = config.get("vocab_size").and_then(|v| v.as_u64());

        (archs, mt, hs, nl, vs)
    } else {
        (vec![], None, None, None, None)
    };

    let has_tokenizer =
        repo.join("tokenizer.json").exists() || repo.join("tokenizer_config.json").exists();
    let has_tokenizer_model = repo.join("tokenizer.model").exists();

    // Count safetensors and total size
    let mut safetensor_count = 0u32;
    let mut total_size = 0u64;

    fn count_safetensors_recursive(dir: &std::path::Path, count: &mut u32, size: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    count_safetensors_recursive(&path, count, size);
                } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if ext == "safetensors" {
                        *count += 1;
                        if let Ok(meta) = path.metadata() {
                            *size += meta.len();
                        }
                    }
                }
            }
        }
    }

    count_safetensors_recursive(&repo, &mut safetensor_count, &mut total_size);

    Ok(ConvertModelInfo {
        repo_path,
        architectures,
        model_type,
        hidden_size,
        num_layers,
        vocab_size,
        has_tokenizer,
        has_tokenizer_model,
        has_config,
        safetensor_count,
        total_size,
        total_size_display: crate::model::format_file_size(total_size),
    })
}

#[tauri::command]
pub async fn convert_run(
    repo_path: String,
    outtype: String,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<ConvertResult, ModelError> {
    let convert_dir = get_convert_dir(&app)?;
    let models_dir = get_models_dir(&app)?;
    let venv_python = get_venv_python(&convert_dir);
    let script_path = get_script_path(&convert_dir);

    if !venv_python.exists() || !script_path.exists() {
        return Err(ModelError::ParseError {
            format: "convert".into(),
            reason: "Convert dependencies not set up. Run setup first.".into(),
        });
    }

    // Validate output type
    let valid_types = ["f32", "f16", "bf16", "q8_0", "auto"];
    if !valid_types.contains(&outtype.as_str()) {
        return Err(ModelError::ParseError {
            format: "convert".into(),
            reason: format!("Invalid output type: {}. Use: f32, f16, bf16, q8_0, auto", outtype),
        });
    }

    // Determine output file path
    let repo_dir_name = std::path::Path::new(&repo_path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let outfile = models_dir.join(format!("{}-{}.gguf", repo_dir_name, outtype));

    // Reset cancel flag
    let cancel = state.convert_cancel.clone();
    cancel.store(false, std::sync::atomic::Ordering::Relaxed);

    let _ = app.emit(
        "convert:progress",
        ConvertProgress {
            stage: "starting".into(),
            message: "Starting conversion...".into(),
            percent: 0.0,
        },
    );

    // Spawn conversion process
    let mut child = tokio::process::Command::new(&venv_python)
        .arg(&script_path)
        .arg(&repo_path)
        .arg("--outtype")
        .arg(&outtype)
        .arg("--outfile")
        .arg(&outfile)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| ModelError::ParseError {
            format: "convert".into(),
            reason: format!("Failed to start conversion: {}", e),
        })?;

    // Store PID for cancellation
    if let Some(pid) = child.id() {
        *state.convert_pid.lock().unwrap() = Some(pid);
    }

    // Read stdout for progress
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let app_out = app.clone();
    let out_handle = tokio::spawn(async move {
        if let Some(stdout) = stdout {
            let reader = tokio::io::BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if !line.trim().is_empty() {
                    let progress = parse_convert_progress(&line);
                    let _ = app_out.emit("convert:progress", progress);
                }
            }
        }
    });

    let app_err = app.clone();
    let err_handle = tokio::spawn(async move {
        let mut last_error = String::new();
        if let Some(stderr) = stderr {
            let reader = tokio::io::BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if !line.trim().is_empty() {
                    last_error = line.clone();
                    let progress = parse_convert_progress(&line);
                    let _ = app_err.emit("convert:progress", progress);
                }
            }
        }
        last_error
    });

    // Wait for process with cancel checking
    let status = loop {
        tokio::select! {
            result = child.wait() => {
                break result.map_err(|e| ModelError::ParseError {
                    format: "convert".into(),
                    reason: format!("Process error: {}", e),
                })?;
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    child.kill().await.ok();
                    *state.convert_pid.lock().unwrap() = None;
                    let _ = app.emit("convert:progress", ConvertProgress {
                        stage: "cancelled".into(),
                        message: "Conversion cancelled".into(),
                        percent: 0.0,
                    });
                    return Err(ModelError::ParseError {
                        format: "convert".into(),
                        reason: "Conversion cancelled".into(),
                    });
                }
            }
        }
    };

    // Wait for output readers to finish
    let _ = out_handle.await;
    let last_error = err_handle.await.unwrap_or_default();

    // Clear PID
    *state.convert_pid.lock().unwrap() = None;

    if !status.success() {
        let reason = if last_error.is_empty() {
            format!("Conversion failed with exit code: {:?}", status.code())
        } else {
            format!("Conversion failed: {}", last_error)
        };
        return Err(ModelError::ParseError {
            format: "convert".into(),
            reason,
        });
    }

    if !outfile.exists() {
        return Err(ModelError::ParseError {
            format: "convert".into(),
            reason: "Conversion produced no output file".into(),
        });
    }

    let output_size = std::fs::metadata(&outfile).map(|m| m.len()).unwrap_or(0);

    // Add converted model to library manifest
    let entry = LocalModelEntry {
        id: format!(
            "{}-{}",
            outfile
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy(),
            chrono::Utc::now().timestamp()
        ),
        file_name: outfile
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        file_path: outfile.to_string_lossy().to_string(),
        file_size: output_size,
        file_size_display: crate::model::format_file_size(output_size),
        format: "gguf".into(),
        source_repo: Some(repo_dir_name),
        downloaded_at: chrono::Utc::now().to_rfc3339(),
    };

    let mut manifest = read_manifest(&models_dir);
    manifest.push(entry);
    write_manifest(&models_dir, &manifest)?;

    let _ = app.emit(
        "convert:progress",
        ConvertProgress {
            stage: "done".into(),
            message: "Conversion complete!".into(),
            percent: 100.0,
        },
    );

    Ok(ConvertResult {
        output_path: outfile.to_string_lossy().to_string(),
        output_size,
        output_size_display: crate::model::format_file_size(output_size),
    })
}

#[tauri::command]
pub async fn convert_cancel(state: State<'_, AppState>) -> Result<(), ModelError> {
    state
        .convert_cancel
        .store(true, std::sync::atomic::Ordering::Relaxed);

    if let Some(pid) = *state.convert_pid.lock().unwrap() {
        #[cfg(unix)]
        {
            let _ = std::process::Command::new("kill")
                .arg(pid.to_string())
                .output();
        }
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .output();
        }
    }

    Ok(())
}

// ── Model Testing / Inference ──────────────────────────

fn llama_cli_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "llama-cli.exe"
    } else {
        "llama-cli"
    }
}

fn resolve_llama_cli(app: &tauri::AppHandle) -> PathBuf {
    let name = llama_cli_binary_name();
    if let Ok(tools_dir) = get_tools_dir(app) {
        if let Some(bundled) = find_binary_recursive(&tools_dir, name) {
            return bundled;
        }
    }
    PathBuf::from(name)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub text: String,
    pub tokens_generated: u32,
    pub time_ms: u64,
    pub device: String,
}

#[tauri::command]
pub async fn test_generate(
    model_path: String,
    prompt: String,
    max_tokens: u32,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<u32>,
    repeat_penalty: Option<f64>,
    gpu_layers: Option<i32>,
    system_prompt: Option<String>,
    context_size: Option<u32>,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<TestResult, ModelError> {
    let path = PathBuf::from(&model_path);
    let cancel = state.test_cancel.clone();
    cancel.store(false, std::sync::atomic::Ordering::Relaxed);

    // Detect format: directory → safetensors, .gguf file → gguf, .safetensors file → use parent dir
    let (format, inference_path) = if path.is_dir() {
        if !path.join("config.json").exists() {
            return Err(ModelError::ParseError {
                format: "test".into(),
                reason: "Directory must contain config.json for SafeTensors inference.".into(),
            });
        }
        ("safetensors", path)
    } else if path.extension().map_or(false, |e| e == "gguf") {
        if !path.exists() {
            return Err(ModelError::ParseError {
                format: "test".into(),
                reason: "GGUF file not found.".into(),
            });
        }
        ("gguf", path)
    } else if path.extension().map_or(false, |e| e == "safetensors") {
        let parent = path.parent().ok_or_else(|| ModelError::ParseError {
            format: "test".into(),
            reason: "Cannot determine model directory.".into(),
        })?.to_path_buf();
        if !parent.join("config.json").exists() {
            return Err(ModelError::ParseError {
                format: "test".into(),
                reason: "SafeTensors inference requires a model directory with config.json. Use the folder path instead.".into(),
            });
        }
        ("safetensors", parent)
    } else {
        return Err(ModelError::ParseError {
            format: "test".into(),
            reason: "Unsupported format. Use a .gguf file or SafeTensors directory.".into(),
        });
    };

    let start = std::time::Instant::now();

    let (full_output, device) = if format == "gguf" {
        // ── GGUF: use llama-cli ──
        let binary = resolve_llama_cli(&app);
        let temp_str = format!("{:.2}", temperature);
        let n_str = max_tokens.to_string();

        let gpu = detect_gpu();
        let has_gpu = gpu.has_nvidia || gpu.has_vulkan || gpu.has_metal;
        let ngl_val = match gpu_layers {
            Some(n) if n >= 0 => n.to_string(),
            _ => if has_gpu { "99".to_string() } else { "0".to_string() },
        };
        let gguf_device = if ngl_val == "0" {
            "CPU".to_string()
        } else if has_gpu {
            if gpu.has_nvidia { "CUDA".to_string() }
            else if gpu.has_metal { "METAL".to_string() }
            else { "VULKAN".to_string() }
        } else {
            "CPU".to_string()
        };

        // Build the full prompt with optional system prompt
        let full_prompt = if let Some(ref sys) = system_prompt {
            if !sys.trim().is_empty() {
                format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]", sys.trim(), prompt)
            } else {
                prompt.clone()
            }
        } else {
            prompt.clone()
        };

        let mut args = vec![
            "-m".to_string(), inference_path.to_string_lossy().to_string(),
            "-p".to_string(), full_prompt,
            "-n".to_string(), n_str.clone(),
            "--temp".to_string(), temp_str.clone(),
            "-ngl".to_string(), ngl_val,
            "--no-display-prompt".to_string(),
            "--log-disable".to_string(),
            "--simple-io".to_string(),
        ];

        if let Some(tp) = top_p {
            args.push("--top-p".to_string());
            args.push(format!("{:.2}", tp));
        }
        if let Some(tk) = top_k {
            args.push("--top-k".to_string());
            args.push(tk.to_string());
        }
        if let Some(rp) = repeat_penalty {
            args.push("--repeat-penalty".to_string());
            args.push(format!("{:.2}", rp));
        }
        if let Some(ctx) = context_size {
            args.push("-c".to_string());
            args.push(ctx.to_string());
        }

        let mut child = tokio::process::Command::new(&binary)
            .args(&args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| ModelError::ParseError {
                format: "test".into(),
                reason: if e.kind() == std::io::ErrorKind::NotFound {
                    "llama-cli not found. Install llama.cpp tools via Settings > Tools.".into()
                } else {
                    format!("Failed to start inference: {}", e)
                },
            })?;

        if let Some(pid) = child.id() {
            *state.test_pid.lock().unwrap() = Some(pid);
        }

        let stdout = child.stdout.take();
        let app_out = app.clone();
        let cancel_out = cancel.clone();

        let output_handle = tokio::spawn(async move {
            let mut output = String::new();
            if let Some(stdout) = stdout {
                let mut reader = tokio::io::BufReader::new(stdout);
                let mut buf = [0u8; 256];
                loop {
                    if cancel_out.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    match reader.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => {
                            let chunk = String::from_utf8_lossy(&buf[..n]).to_string();
                            output.push_str(&chunk);
                            let _ = app_out.emit("test:token", &chunk);
                        }
                        Err(_) => break,
                    }
                }
            }
            output
        });

        let stderr = child.stderr.take();
        let err_handle = tokio::spawn(async move {
            let mut last_err = String::new();
            if let Some(stderr) = stderr {
                let reader = tokio::io::BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if !line.trim().is_empty() {
                        last_err = line;
                    }
                }
            }
            last_err
        });

        let status = loop {
            tokio::select! {
                result = child.wait() => {
                    break result.map_err(|e| ModelError::ParseError {
                        format: "test".into(),
                        reason: format!("Process error: {}", e),
                    })?;
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        child.kill().await.ok();
                        *state.test_pid.lock().unwrap() = None;
                        return Err(ModelError::ParseError {
                            format: "test".into(),
                            reason: "Generation cancelled".into(),
                        });
                    }
                }
            }
        };

        let output = output_handle.await.unwrap_or_default();
        let last_error = err_handle.await.unwrap_or_default();
        *state.test_pid.lock().unwrap() = None;

        if !status.success() && output.is_empty() {
            return Err(ModelError::ParseError {
                format: "test".into(),
                reason: if last_error.is_empty() {
                    format!("Inference failed (exit code {:?})", status.code())
                } else {
                    format!("Inference failed: {}", last_error)
                },
            });
        }

        (output, gguf_device)
    } else {
        // ── SafeTensors: use Python transformers ──
        // Prefer training venv (has bitsandbytes, accelerate, peft for quantized models),
        // fall back to convert venv
        let training_dir = crate::training::venv::get_training_dir(&app)?;
        let training_python = crate::training::venv::get_venv_python(&training_dir);

        let convert_dir = get_convert_dir(&app)?;
        let convert_python = get_venv_python(&convert_dir);

        let venv_python = if training_python.exists() {
            training_python
        } else if convert_python.exists() {
            convert_python
        } else {
            return Err(ModelError::ParseError {
                format: "test".into(),
                reason: "Python environment not set up. Install dependencies via the TRAINING or CONVERT page first.".into(),
            });
        };

        let script = r#"
import sys, json, os, torch, warnings, gc
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

path, prompt = sys.argv[1], sys.argv[2]
max_tok, temp = int(sys.argv[3]), float(sys.argv[4])
opts = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}

force_cpu = opts.get("gpu_layers", -1) == 0
has_cuda = torch.cuda.is_available() and not force_cpu

# Check if model has quantization config (4-bit/8-bit fine-tuned)
import json as _json
config_path = os.path.join(path, "config.json")
is_quantized = False
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = _json.load(f)
    is_quantized = "quantization_config" in cfg

tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

device = "cpu"
model = None

if has_cuda:
    torch.cuda.empty_cache()
    gc.collect()
    try:
        if is_quantized:
            # Quantized models: use device_map="auto" (requires accelerate)
            model = AutoModelForCausalLM.from_pretrained(
                path, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
            )
            device = "cuda"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
            ).to("cuda")
            device = "cuda"
    except Exception:
        gc.collect()
        torch.cuda.empty_cache()
        device = "cpu"
        model = None

if model is None:
    if is_quantized:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
                quantization_config=BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=False) if not has_cuda else None,
            )
        except Exception:
            # Last resort: force no quantization
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
        )

sys.stderr.write(f"[device:{device}]\n")
sys.stderr.flush()
model.eval()

# Build prompt — try chat template first, fall back to manual
sys_prompt = opts.get("system_prompt", "")
try:
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception:
    if sys_prompt:
        full = f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
    else:
        full = prompt

ids = tok(full, return_tensors="pt")
if device == "cuda" and not is_quantized:
    ids = ids.to("cuda")
elif device == "cuda" and is_quantized:
    ids = ids.to(model.device)

streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
gen_kwargs = dict(**ids, max_new_tokens=max_tok, temperature=max(temp, 0.01), do_sample=temp > 0, streamer=streamer)

if "top_p" in opts and opts["top_p"] is not None:
    gen_kwargs["top_p"] = opts["top_p"]
if "top_k" in opts and opts["top_k"] is not None:
    gen_kwargs["top_k"] = opts["top_k"]
if "repeat_penalty" in opts and opts["repeat_penalty"] is not None:
    gen_kwargs["repetition_penalty"] = opts["repeat_penalty"]

thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()
for text in streamer:
    sys.stdout.write(text)
    sys.stdout.flush()
thread.join()
"#;

        let temp_str = format!("{:.2}", temperature);
        let n_str = max_tokens.to_string();

        // Build extra options JSON for Python
        let mut py_opts = serde_json::json!({});
        if let Some(tp) = top_p { py_opts["top_p"] = serde_json::json!(tp); }
        if let Some(tk) = top_k { py_opts["top_k"] = serde_json::json!(tk); }
        if let Some(rp) = repeat_penalty { py_opts["repeat_penalty"] = serde_json::json!(rp); }
        if let Some(gl) = gpu_layers { py_opts["gpu_layers"] = serde_json::json!(gl); }
        if let Some(ref sp) = system_prompt { py_opts["system_prompt"] = serde_json::json!(sp); }
        let py_opts_str = py_opts.to_string();

        let mut child = tokio::process::Command::new(&venv_python)
            .args(["-c", script, &inference_path.to_string_lossy(), &prompt, &n_str, &temp_str, &py_opts_str])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| ModelError::ParseError {
                format: "test".into(),
                reason: format!("Failed to start Python inference: {}", e),
            })?;

        if let Some(pid) = child.id() {
            *state.test_pid.lock().unwrap() = Some(pid);
        }

        let stdout = child.stdout.take();
        let app_out = app.clone();
        let cancel_out = cancel.clone();

        let output_handle = tokio::spawn(async move {
            let mut output = String::new();
            if let Some(stdout) = stdout {
                let mut reader = tokio::io::BufReader::new(stdout);
                let mut buf = [0u8; 256];
                loop {
                    if cancel_out.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    match reader.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => {
                            let chunk = String::from_utf8_lossy(&buf[..n]).to_string();
                            output.push_str(&chunk);
                            let _ = app_out.emit("test:token", &chunk);
                        }
                        Err(_) => break,
                    }
                }
            }
            output
        });

        let stderr = child.stderr.take();
        let err_handle = tokio::spawn(async move {
            let mut stderr_lines: Vec<String> = Vec::new();
            let mut detected_device = String::new();
            if let Some(stderr) = stderr {
                let reader = tokio::io::BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if line.starts_with("[device:") && line.ends_with(']') {
                        detected_device = line[8..line.len()-1].to_uppercase();
                    } else if !line.trim().is_empty() {
                        stderr_lines.push(line);
                    }
                }
            }
            // Extract meaningful error: find last traceback or last meaningful lines
            let error_msg = {
                let mut tb_start: Option<usize> = None;
                for (i, l) in stderr_lines.iter().enumerate() {
                    if l.starts_with("Traceback (most recent call last)") {
                        tb_start = Some(i);
                    }
                }
                if let Some(start) = tb_start {
                    stderr_lines[start..].iter().take(30).cloned().collect::<Vec<_>>().join("\n")
                } else {
                    stderr_lines.iter().rev()
                        .filter(|l| !l.starts_with("WARNING") && !l.contains("FutureWarning") && !l.contains("UserWarning"))
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .into_iter().rev().collect::<Vec<_>>().join("\n")
                }
            };
            (error_msg, detected_device)
        });

        let status = loop {
            tokio::select! {
                result = child.wait() => {
                    break result.map_err(|e| ModelError::ParseError {
                        format: "test".into(),
                        reason: format!("Process error: {}", e),
                    })?;
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        child.kill().await.ok();
                        *state.test_pid.lock().unwrap() = None;
                        return Err(ModelError::ParseError {
                            format: "test".into(),
                            reason: "Generation cancelled".into(),
                        });
                    }
                }
            }
        };

        let output = output_handle.await.unwrap_or_default();
        let (last_error, py_device) = err_handle.await.unwrap_or_default();
        *state.test_pid.lock().unwrap() = None;

        if !status.success() && output.is_empty() {
            return Err(ModelError::ParseError {
                format: "test".into(),
                reason: if last_error.is_empty() {
                    format!("Inference failed (exit code {:?})", status.code())
                } else {
                    format!("Inference failed: {}", last_error)
                },
            });
        }

        let st_device = if py_device.is_empty() { "CPU".to_string() } else { py_device };
        (output, st_device)
    };

    let elapsed = start.elapsed().as_millis() as u64;
    let token_count = full_output.split_whitespace().count() as u32;

    let _ = app.emit("test:done", &full_output);

    Ok(TestResult {
        text: full_output,
        tokens_generated: token_count,
        time_ms: elapsed,
        device,
    })
}

#[tauri::command]
pub async fn test_cancel(state: State<'_, AppState>) -> Result<(), ModelError> {
    state.test_cancel.store(true, std::sync::atomic::Ordering::Relaxed);

    if let Some(pid) = *state.test_pid.lock().unwrap() {
        #[cfg(unix)]
        {
            let _ = std::process::Command::new("kill")
                .arg(pid.to_string())
                .output();
        }
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .output();
        }
    }

    Ok(())
}

// ── Quantize ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeResult {
    pub success: bool,
    pub output_path: String,
    pub output_size: u64,
    pub output_size_display: String,
}

#[tauri::command]
pub async fn quantize_model(
    target_type: String,
    output_path: String,
    state: State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<QuantizeResult, ModelError> {
    // Extract file path and validate format
    let input_path = {
        let loaded = state.loaded_model.lock().unwrap();
        let info = loaded.as_ref().ok_or_else(|| ModelError::ParseError {
            format: "quantize".into(),
            reason: "No model loaded".into(),
        })?;

        match info.format {
            ModelFormat::Gguf => {}
            _ => {
                return Err(ModelError::ParseError {
                    format: "quantize".into(),
                    reason: "Only GGUF models can be quantized with llama-quantize".into(),
                })
            }
        }

        info.file_path.clone()
    };

    // Validate target quantization type
    let valid_types = [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S",
        "Q5_K_M", "Q6_K", "Q8_0", "F16",
    ];
    if !valid_types.contains(&target_type.as_str()) {
        return Err(ModelError::ParseError {
            format: "quantize".into(),
            reason: format!("Unknown quantization type: {}", target_type),
        });
    }

    // Resolve binary: bundled takes priority, then PATH
    let binary = resolve_quantize_binary(&app);
    let target = target_type.clone();
    let output = output_path.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let child = std::process::Command::new(&binary)
            .arg(&input_path)
            .arg(&output)
            .arg(&target)
            .output();

        match child {
            Ok(proc_output) => {
                if proc_output.status.success() {
                    Ok(())
                } else {
                    // Combine stdout + stderr, take only the last meaningful lines
                    let stdout = String::from_utf8_lossy(&proc_output.stdout);
                    let stderr = String::from_utf8_lossy(&proc_output.stderr);
                    let combined = format!("{}{}", stdout, stderr);
                    let last_lines: String = combined
                        .lines()
                        .rev()
                        .take(5)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join("\n");
                    Err(ModelError::ParseError {
                        format: "quantize".into(),
                        reason: format!("llama-quantize failed: {}", last_lines),
                    })
                }
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Err(ModelError::ParseError {
                        format: "quantize".into(),
                        reason: "llama-quantize not found. Install it via Settings > Tools or add it to your PATH.".into(),
                    })
                } else {
                    Err(ModelError::IoError(e))
                }
            }
        }
    })
    .await
    .map_err(|e| ModelError::ParseError {
        format: "quantize".into(),
        reason: format!("Task failed: {}", e),
    })??;

    let output_meta = std::fs::metadata(&output_path).map_err(ModelError::IoError)?;

    Ok(QuantizeResult {
        success: true,
        output_path,
        output_size: output_meta.len(),
        output_size_display: crate::model::format_file_size(output_meta.len()),
    })
}

// ── System Info ───────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub used_ram_mb: u64,
    pub cpu_name: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
}

#[tauri::command]
pub fn get_system_info() -> SystemInfo {
    use sysinfo::System;

    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();

    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let available_ram_mb = sys.available_memory() / (1024 * 1024);
    let used_ram_mb = sys.used_memory() / (1024 * 1024);

    let cpu_name = sys.cpus().first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown".to_string());
    let cpu_cores = sys.physical_core_count().unwrap_or(0);
    let cpu_threads = sys.cpus().len();

    SystemInfo {
        total_ram_mb,
        available_ram_mb,
        used_ram_mb,
        cpu_name,
        cpu_cores,
        cpu_threads,
    }
}

// ── App Settings Persistence ──────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppSettings {
    pub memory_limit_mb: Option<u64>,
}

#[tauri::command]
pub fn load_settings(app: tauri::AppHandle) -> AppSettings {
    let dir = app.path().app_data_dir().expect("No app data dir");
    let path = dir.join("settings.json");
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

#[tauri::command]
pub fn save_settings(app: tauri::AppHandle, settings: AppSettings) -> Result<(), ModelError> {
    let dir = app.path().app_data_dir().expect("No app data dir");
    std::fs::create_dir_all(&dir).map_err(ModelError::IoError)?;
    let path = dir.join("settings.json");
    let json = serde_json::to_string_pretty(&settings)
        .map_err(|e| ModelError::MergeError(e.to_string()))?;
    std::fs::write(&path, json).map_err(ModelError::IoError)?;
    Ok(())
}

// ── Convert Environment Commands ──────────────────────────

#[tauri::command]
pub async fn convert_clean_env(app: tauri::AppHandle) -> Result<(), ModelError> {
    let convert_dir = get_convert_dir(&app)?;
    if convert_dir.exists() {
        std::fs::remove_dir_all(&convert_dir).map_err(ModelError::IoError)?;
    }
    Ok(())
}

// ── Dataset Hub Commands ──────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfDatasetFileInfo {
    pub rfilename: String,
    pub size: Option<u64>,
    pub size_display: String,
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfDatasetRepoInfo {
    pub id: String,
    pub files: Vec<HfDatasetFileInfo>,
}

fn detect_dataset_format(filename: &str) -> Option<String> {
    let lower = filename.to_lowercase();
    if lower.ends_with(".parquet") {
        Some("parquet".into())
    } else if lower.ends_with(".json") {
        Some("json".into())
    } else if lower.ends_with(".jsonl") || lower.ends_with(".ndjson") {
        Some("jsonl".into())
    } else if lower.ends_with(".csv") {
        Some("csv".into())
    } else {
        None
    }
}

fn get_datasets_dir(app: &tauri::AppHandle) -> Result<PathBuf, ModelError> {
    let data_dir = app.path().app_data_dir().map_err(|e| ModelError::ParseError {
        format: "datastudio".into(),
        reason: format!("Cannot resolve app data dir: {}", e),
    })?;
    Ok(data_dir.join("datasets"))
}

#[tauri::command]
pub async fn hf_fetch_dataset_repo(repo_id: String) -> Result<HfDatasetRepoInfo, ModelError> {
    let client = build_http_client()?;

    let url = format!("https://huggingface.co/api/datasets/{}", repo_id);
    let resp = client.get(&url).send().await.map_err(|e| ModelError::ParseError {
        format: "datastudio".into(),
        reason: format!("Failed to fetch dataset repo: {}", e),
    })?;

    if !resp.status().is_success() {
        return Err(ModelError::ParseError {
            format: "datastudio".into(),
            reason: format!("Dataset not found or inaccessible (HTTP {})", resp.status()),
        });
    }

    #[derive(Deserialize)]
    struct DsApiResponse {
        id: Option<String>,
        siblings: Option<Vec<HfApiSibling>>,
    }

    let api_resp: DsApiResponse = resp.json().await.map_err(|e| ModelError::ParseError {
        format: "datastudio".into(),
        reason: format!("Failed to parse response: {}", e),
    })?;

    let repo_name = api_resp.id.unwrap_or_else(|| repo_id.to_string());
    let siblings = api_resp.siblings.unwrap_or_default();

    let mut files: Vec<HfDatasetFileInfo> = siblings
        .into_iter()
        .filter_map(|s| {
            let format = detect_dataset_format(&s.rfilename);
            if format.is_some() {
                let size_display = s.size.map(|sz| crate::model::format_file_size(sz)).unwrap_or_else(|| "---".into());
                Some(HfDatasetFileInfo {
                    rfilename: s.rfilename,
                    size: s.size,
                    size_display,
                    format,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort: parquet first, then by size descending
    files.sort_by(|a, b| {
        let a_parquet = a.format.as_deref() == Some("parquet");
        let b_parquet = b.format.as_deref() == Some("parquet");
        b_parquet.cmp(&a_parquet)
            .then_with(|| b.size.unwrap_or(0).cmp(&a.size.unwrap_or(0)))
    });

    Ok(HfDatasetRepoInfo {
        id: repo_name,
        files,
    })
}

#[tauri::command]
pub async fn hf_download_dataset_file(
    repo_id: String,
    filename: String,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<String, ModelError> {
    let datasets_dir = get_datasets_dir(&app)?;
    let repo_folder = datasets_dir.join(repo_id.replace('/', "--"));
    std::fs::create_dir_all(&repo_folder).map_err(ModelError::IoError)?;

    // Reset cancel flag
    let cancel = state.download_cancel.clone();
    cancel.store(false, std::sync::atomic::Ordering::Relaxed);

    let download_url = format!(
        "https://huggingface.co/datasets/{}/resolve/main/{}",
        repo_id, filename
    );

    let client = build_http_client()?;

    let resp = client.get(&download_url).send().await.map_err(|e| ModelError::ParseError {
        format: "datastudio".into(),
        reason: format!("Download request failed: {}", e),
    })?;

    if !resp.status().is_success() {
        return Err(ModelError::ParseError {
            format: "datastudio".into(),
            reason: format!("Download failed (HTTP {})", resp.status()),
        });
    }

    let total_size = resp.content_length().unwrap_or(0);

    // Preserve directory structure within the repo folder
    let out_path = repo_folder.join(&filename);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(ModelError::IoError)?;
    }
    let partial_path = repo_folder.join(format!("{}.partial", filename.replace('/', "_")));

    let mut file = std::fs::File::create(&partial_path).map_err(ModelError::IoError)?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    let safe_name = std::path::Path::new(&filename)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&filename)
        .to_string();

    while let Some(chunk) = stream.next().await {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(&partial_path);
            let _ = app.emit("datastudio:download-progress", DownloadProgress {
                file_name: safe_name.clone(),
                bytes_downloaded: downloaded,
                bytes_total: total_size,
                percent: 0.0,
                status: "cancelled".into(),
                files_done: None,
                files_total: None,
            });
            return Err(ModelError::ParseError {
                format: "datastudio".into(),
                reason: "Download cancelled".into(),
            });
        }

        let bytes = chunk.map_err(|e| ModelError::ParseError {
            format: "datastudio".into(),
            reason: format!("Download stream error: {}", e),
        })?;

        {
            use std::io::Write;
            file.write_all(&bytes).map_err(ModelError::IoError)?;
        }
        downloaded += bytes.len() as u64;

        let now = std::time::Instant::now();
        if now.duration_since(last_emit).as_millis() >= 500 || (total_size > 0 && downloaded >= total_size) {
            let percent = if total_size > 0 {
                (downloaded as f64 / total_size as f64) * 100.0
            } else {
                0.0
            };
            let _ = app.emit("datastudio:download-progress", DownloadProgress {
                file_name: safe_name.clone(),
                bytes_downloaded: downloaded,
                bytes_total: total_size,
                percent,
                status: "downloading".into(),
                files_done: None,
                files_total: None,
            });
            last_emit = now;
        }
    }

    drop(file);

    // Rename partial to final
    std::fs::rename(&partial_path, &out_path).map_err(ModelError::IoError)?;

    // Emit completion
    let _ = app.emit("datastudio:download-progress", DownloadProgress {
        file_name: safe_name,
        bytes_downloaded: downloaded,
        bytes_total: downloaded,
        percent: 100.0,
        status: "complete".into(),
        files_done: None,
        files_total: None,
    });

    Ok(out_path.to_string_lossy().to_string())
}
