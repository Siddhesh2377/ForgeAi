import { invoke } from "@tauri-apps/api/core";

export interface TensorInfo {
  name: string;
  dtype: string;
  shape: number[];
}

export interface ModelInfo {
  file_name: string;
  file_path: string;
  file_size: number;
  file_size_display: string;
  format: "safe_tensors" | "gguf";
  tensor_count: number;
  parameter_count: number;
  parameter_count_display: string;
  layer_count: number | null;
  quantization: string | null;
  architecture: string | null;
  context_length: number | null;
  embedding_size: number | null;
  metadata: Record<string, string>;
  tensor_preview: TensorInfo[];
  shard_count: number | null;
  has_tokenizer: boolean | null;
  has_config: boolean | null;
  model_type: string | null;
  vocab_size: number | null;
}

export type LoadStatus = "idle" | "loading" | "loaded" | "error";

class ModelStore {
  info = $state<ModelInfo | null>(null);
  status = $state<LoadStatus>("idle");
  error = $state<string | null>(null);

  get isLoaded(): boolean {
    return this.info !== null;
  }

  get formatDisplay(): string {
    if (!this.info) return "---";
    const map: Record<string, string> = {
      safe_tensors: "SAFETENSORS",
      gguf: "GGUF",
    };
    return map[this.info.format] ?? this.info.format.toUpperCase();
  }

  async load(filePath: string): Promise<void> {
    this.status = "loading";
    this.error = null;
    try {
      const result = await invoke<ModelInfo>("load_model", { path: filePath });
      this.info = result;
      this.status = "loaded";
    } catch (e) {
      this.error = String(e);
      this.status = "error";
      this.info = null;
    }
  }

  async loadDir(dirPath: string): Promise<void> {
    this.status = "loading";
    this.error = null;
    try {
      const result = await invoke<ModelInfo>("load_model_dir", { path: dirPath });
      this.info = result;
      this.status = "loaded";
    } catch (e) {
      this.error = String(e);
      this.status = "error";
      this.info = null;
    }
  }

  get isFolder(): boolean {
    return (this.info?.shard_count ?? 0) > 0;
  }

  async unload(): Promise<void> {
    try {
      await invoke("unload_model");
    } catch {
      // ignore
    }
    this.info = null;
    this.status = "idle";
    this.error = null;
  }
}

export const model = new ModelStore();
