import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export interface HfFileInfo {
  rfilename: string;
  size: number | null;
  size_display: string;
  format: string | null;
}

export interface HfRepoInfo {
  id: string;
  files: HfFileInfo[];
}

export interface LocalModelEntry {
  id: string;
  file_name: string;
  file_path: string;
  file_size: number;
  file_size_display: string;
  format: string;
  source_repo: string | null;
  downloaded_at: string;
}

export interface DownloadProgress {
  file_name: string;
  bytes_downloaded: number;
  bytes_total: number;
  percent: number;
  status: string;
  files_done: number | null;
  files_total: number | null;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + " GB";
  if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + " MB";
  if (bytes >= 1024) return (bytes / 1024).toFixed(0) + " KB";
  return bytes + " B";
}

class HubStore {
  repoInfo = $state<HfRepoInfo | null>(null);
  repoLoading = $state(false);
  repoError = $state<string | null>(null);

  downloadProgress = $state<DownloadProgress | null>(null);
  downloading = $state(false);
  downloadError = $state<string | null>(null);

  localModels = $state<LocalModelEntry[]>([]);
  libraryLoading = $state(false);

  private unlisten: UnlistenFn | null = null;

  async setupListener() {
    if (this.unlisten) return;
    this.unlisten = await listen<DownloadProgress>("hub:download-progress", (e) => {
      this.downloadProgress = e.payload;
      if (e.payload.status === "complete" || e.payload.status === "cancelled") {
        this.downloading = false;
      }
    });
  }

  async fetchRepo(repoId: string) {
    this.repoLoading = true;
    this.repoError = null;
    this.repoInfo = null;
    try {
      this.repoInfo = await invoke<HfRepoInfo>("hf_fetch_repo", { repoId });
    } catch (e) {
      this.repoError = String(e);
    } finally {
      this.repoLoading = false;
    }
  }

  async downloadFile(repoId: string, filename: string) {
    await this.setupListener();
    this.downloading = true;
    this.downloadError = null;
    this.downloadProgress = {
      file_name: filename,
      bytes_downloaded: 0,
      bytes_total: 0,
      percent: 0,
      status: "downloading",
      files_done: null,
      files_total: null,
    };
    try {
      await invoke<LocalModelEntry>("hf_download_file", { repoId, filename });
      await this.loadLibrary();
    } catch (e) {
      const msg = String(e);
      if (!msg.includes("cancelled")) {
        this.downloadError = msg;
      }
    } finally {
      this.downloading = false;
    }
  }

  async downloadRepo(repoId: string) {
    await this.setupListener();
    this.downloading = true;
    this.downloadError = null;
    this.downloadProgress = {
      file_name: repoId,
      bytes_downloaded: 0,
      bytes_total: 0,
      percent: 0,
      status: "downloading",
      files_done: 0,
      files_total: this.repoInfo?.files.length ?? 0,
    };
    try {
      await invoke<LocalModelEntry>("hf_download_repo", { repoId });
      await this.loadLibrary();
    } catch (e) {
      const msg = String(e);
      if (!msg.includes("cancelled")) {
        this.downloadError = msg;
      }
    } finally {
      this.downloading = false;
    }
  }

  async cancelDownload() {
    try {
      await invoke("hub_cancel_download");
    } catch {
      // ignore
    }
  }

  async loadLibrary() {
    this.libraryLoading = true;
    try {
      this.localModels = await invoke<LocalModelEntry[]>("hub_list_local");
    } catch (e) {
      console.error("Failed to load library:", e);
    } finally {
      this.libraryLoading = false;
    }
  }

  async importLocal(path: string) {
    try {
      await invoke<LocalModelEntry>("hub_import_local", { path });
      await this.loadLibrary();
    } catch (e) {
      throw e;
    }
  }

  async deleteModel(id: string) {
    try {
      await invoke("hub_delete_model", { modelId: id });
      await this.loadLibrary();
    } catch (e) {
      console.error("Failed to delete model:", e);
    }
  }

  get totalStorageBytes(): number {
    return this.localModels.reduce((sum, m) => sum + m.file_size, 0);
  }

  get totalStorageDisplay(): string {
    return formatBytes(this.totalStorageBytes);
  }
}

export const hub = new HubStore();
