import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

// ── Types ───────────────────────────────────────────

export interface ColumnAnalysis {
  name: string;
  dtype: string;
  non_null_count: number;
  null_count: number;
  sample_values: string[];
  avg_length: number | null;
}

export interface DatasetFullInfo {
  path: string;
  format: string;
  rows: number;
  columns: string[];
  column_analysis: ColumnAnalysis[];
  preview: any[];
  size: number;
  size_display: string;
  detected_template: string | null;
}

export interface HfDatasetFileInfo {
  rfilename: string;
  size: number | null;
  size_display: string;
  format: string | null;
}

export interface HfDatasetRepoInfo {
  id: string;
  files: HfDatasetFileInfo[];
}

export interface DownloadProgress {
  file_name: string;
  bytes_downloaded: number;
  bytes_total: number;
  percent: number;
  status: string;
}

// ── Store ───────────────────────────────────────────

class DataStudioStore {
  dataset = $state<DatasetFullInfo | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);
  pendingPath = $state<string | null>(null);

  // Source toggle
  source = $state<"local" | "huggingface">("local");

  // HuggingFace state
  hfQuery = $state("");
  hfRepo = $state<HfDatasetRepoInfo | null>(null);
  hfLoading = $state(false);
  hfError = $state<string | null>(null);
  hfDownloading = $state(false);
  hfDownloadProgress = $state<DownloadProgress | null>(null);

  private unlisten: UnlistenFn | null = null;

  async loadDataset(path: string) {
    this.loading = true;
    this.error = null;
    this.dataset = null;
    try {
      this.dataset = await invoke<DatasetFullInfo>("training_detect_dataset_full", { path });
    } catch (e) {
      this.error = String(e);
    } finally {
      this.loading = false;
    }
  }

  async browseDataset() {
    const result = await open({
      filters: [
        { name: "Dataset", extensions: ["json", "jsonl", "csv", "parquet"] },
      ],
      title: "Select Dataset File",
    });
    if (result) {
      await this.loadDataset(result as string);
    }
  }

  async fetchHfDataset(repoId: string) {
    this.hfLoading = true;
    this.hfError = null;
    this.hfRepo = null;
    try {
      this.hfRepo = await invoke<HfDatasetRepoInfo>("hf_fetch_dataset_repo", { repoId });
    } catch (e) {
      this.hfError = String(e);
    } finally {
      this.hfLoading = false;
    }
  }

  async downloadHfFile(repoId: string, filename: string) {
    this.hfDownloading = true;
    this.hfError = null;
    this.hfDownloadProgress = null;

    // Listen for progress events
    this.unlisten = await listen<DownloadProgress>("datastudio:download-progress", (event) => {
      this.hfDownloadProgress = event.payload;
    });

    try {
      const localPath = await invoke<string>("hf_download_dataset_file", { repoId, filename });
      // Auto-load the downloaded dataset
      this.source = "local";
      await this.loadDataset(localPath);
    } catch (e) {
      this.hfError = String(e);
    } finally {
      this.hfDownloading = false;
      this.hfDownloadProgress = null;
      if (this.unlisten) {
        this.unlisten();
        this.unlisten = null;
      }
    }
  }

  reset() {
    this.dataset = null;
    this.loading = false;
    this.error = null;
    this.pendingPath = null;
  }

  resetHf() {
    this.hfRepo = null;
    this.hfLoading = false;
    this.hfError = null;
    this.hfDownloading = false;
    this.hfDownloadProgress = null;
    this.hfQuery = "";
  }
}

export const datastudio = new DataStudioStore();
