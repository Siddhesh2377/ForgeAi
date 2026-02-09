import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export interface ConvertDepsStatus {
  python_found: boolean;
  python_version: string | null;
  python_path: string | null;
  venv_ready: boolean;
  script_ready: boolean;
  packages_ready: boolean;
  missing_packages: string[];
  ready: boolean;
}

export interface ConvertModelInfo {
  repo_path: string;
  architectures: string[];
  model_type: string | null;
  hidden_size: number | null;
  num_layers: number | null;
  vocab_size: number | null;
  has_tokenizer: boolean;
  has_tokenizer_model: boolean;
  has_config: boolean;
  safetensor_count: number;
  total_size: number;
  total_size_display: string;
}

export interface ConvertProgress {
  stage: string;
  message: string;
  percent: number;
}

export interface ConvertResult {
  output_path: string;
  output_size: number;
  output_size_display: string;
}

class ConvertStore {
  deps = $state<ConvertDepsStatus | null>(null);
  depsLoading = $state(false);

  setupRunning = $state(false);
  setupProgress = $state<ConvertProgress | null>(null);
  setupError = $state<string | null>(null);

  modelInfo = $state<ConvertModelInfo | null>(null);
  modelLoading = $state(false);
  modelError = $state<string | null>(null);

  converting = $state(false);
  convertProgress = $state<ConvertProgress | null>(null);
  convertError = $state<string | null>(null);
  convertResult = $state<ConvertResult | null>(null);

  private setupUnlisten: UnlistenFn | null = null;
  private convertUnlisten: UnlistenFn | null = null;

  async checkDeps() {
    this.depsLoading = true;
    try {
      this.deps = await invoke<ConvertDepsStatus>("convert_check_deps");
    } catch (e) {
      console.error("Failed to check deps:", e);
    } finally {
      this.depsLoading = false;
    }
  }

  async setup() {
    // Listen for setup progress events
    if (!this.setupUnlisten) {
      this.setupUnlisten = await listen<ConvertProgress>(
        "convert:setup-progress",
        (e) => {
          this.setupProgress = e.payload;
        },
      );
    }

    this.setupRunning = true;
    this.setupError = null;
    this.setupProgress = {
      stage: "setup",
      message: "Starting setup...",
      percent: 0,
    };

    try {
      await invoke("convert_setup");
      await this.checkDeps();
    } catch (e) {
      this.setupError = String(e);
    } finally {
      this.setupRunning = false;
    }
  }

  async detectModel(repoPath: string) {
    this.modelLoading = true;
    this.modelError = null;
    this.modelInfo = null;
    try {
      this.modelInfo = await invoke<ConvertModelInfo>("convert_detect_model", {
        repoPath,
      });
    } catch (e) {
      this.modelError = String(e);
    } finally {
      this.modelLoading = false;
    }
  }

  async run(repoPath: string, outtype: string) {
    // Listen for convert progress events
    if (!this.convertUnlisten) {
      this.convertUnlisten = await listen<ConvertProgress>(
        "convert:progress",
        (e) => {
          this.convertProgress = e.payload;
        },
      );
    }

    this.converting = true;
    this.convertError = null;
    this.convertResult = null;
    this.convertProgress = {
      stage: "starting",
      message: "Starting conversion...",
      percent: 0,
    };

    try {
      this.convertResult = await invoke<ConvertResult>("convert_run", {
        repoPath,
        outtype,
      });
    } catch (e) {
      const msg = String(e);
      if (!msg.includes("cancelled")) {
        this.convertError = msg;
      }
    } finally {
      this.converting = false;
    }
  }

  async cancel() {
    try {
      await invoke("convert_cancel");
    } catch {
      // ignore
    }
  }

  reset() {
    this.modelInfo = null;
    this.modelError = null;
    this.convertProgress = null;
    this.convertError = null;
    this.convertResult = null;
  }
}

export const convert = new ConvertStore();
