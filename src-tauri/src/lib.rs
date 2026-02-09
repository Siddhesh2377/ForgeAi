mod commands;
mod merge;
mod merge_commands;
mod model;

use model::state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::load_model,
            commands::load_model_dir,
            commands::get_loaded_model,
            commands::unload_model,
            commands::inspect_model,
            commands::compute_fingerprint,
            commands::quantize_model,
            commands::detect_gpu,
            commands::get_tools_status,
            commands::download_llama_cpp,
            commands::remove_tools,
            commands::hf_fetch_repo,
            commands::hf_download_file,
            commands::hf_download_repo,
            commands::hub_list_local,
            commands::hub_delete_model,
            commands::hub_cancel_download,
            commands::hub_import_local,
            commands::convert_check_deps,
            commands::convert_setup,
            commands::convert_detect_model,
            commands::convert_run,
            commands::convert_cancel,
            commands::test_generate,
            commands::test_cancel,
            // Merge commands
            merge_commands::merge_load_parent,
            merge_commands::merge_load_parent_dir,
            merge_commands::merge_remove_parent,
            merge_commands::merge_get_parents,
            merge_commands::merge_clear_parents,
            merge_commands::merge_check_compatibility,
            merge_commands::merge_validate_config,
            merge_commands::merge_execute,
            merge_commands::merge_cancel,
            merge_commands::merge_profile_layers,
            merge_commands::merge_profile_cancel,
            merge_commands::merge_preview,
            merge_commands::merge_get_methods,
            merge_commands::merge_compare_tensors,
            merge_commands::merge_analyze_layers,
            merge_commands::merge_get_categories,
            merge_commands::merge_get_layer_components,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
