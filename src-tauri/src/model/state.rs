use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use super::ModelInfo;
use crate::merge::registry::ParentRegistry;

pub struct AppState {
    pub loaded_model: Mutex<Option<ModelInfo>>,
    pub download_cancel: Arc<AtomicBool>,
    pub convert_cancel: Arc<AtomicBool>,
    pub convert_pid: Mutex<Option<u32>>,
    pub test_cancel: Arc<AtomicBool>,
    pub test_pid: Mutex<Option<u32>>,
    pub merge_parents: Mutex<ParentRegistry>,
    pub merge_cancel: Arc<AtomicBool>,
    pub merge_active: Arc<AtomicBool>,
    pub profiler_cancel: Arc<AtomicBool>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            loaded_model: Mutex::new(None),
            download_cancel: Arc::new(AtomicBool::new(false)),
            convert_cancel: Arc::new(AtomicBool::new(false)),
            convert_pid: Mutex::new(None),
            test_cancel: Arc::new(AtomicBool::new(false)),
            test_pid: Mutex::new(None),
            merge_parents: Mutex::new(ParentRegistry::default()),
            merge_cancel: Arc::new(AtomicBool::new(false)),
            merge_active: Arc::new(AtomicBool::new(false)),
            profiler_cancel: Arc::new(AtomicBool::new(false)),
        }
    }
}
