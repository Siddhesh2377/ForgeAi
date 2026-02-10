import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

// ── Types ───────────────────────────────────────────

export interface TrainingDepsStatus {
  python_found: boolean;
  python_version: string | null;
  venv_ready: boolean;
  packages_ready: boolean;
  missing_packages: string[];
  cuda_available: boolean;
  cuda_version: string | null;
  torch_version: string | null;
  ready: boolean;
}

export interface DatasetInfo {
  path: string;
  format: string;
  rows: number;
  columns: string[];
  preview: any[];
  size: number;
  size_display: string;
  detected_template: string | null;
}

export interface TrainingProgress {
  stage: string;
  message: string;
  percent: number;
  epoch: number | null;
  step: number | null;
  total_steps: number | null;
  loss: number | null;
  learning_rate: number | null;
  eta_seconds: number | null;
  gpu_memory_used_mb: number | null;
}

export interface TrainingResult {
  output_path: string;
  output_size: number;
  output_size_display: string;
  method: string;
  epochs_completed: number;
  final_loss: number | null;
  adapter_merged: boolean;
}

export interface SurgeryResult {
  output_path: string;
  output_size: number;
  output_size_display: string;
  original_layers: number;
  final_layers: number;
  tensors_written: number;
}

export interface TargetModuleGroup {
  name: string;
  modules: string[];
}

export interface LayerCapabilityMapping {
  capability: string;
  name: string;
  layers: number[];
}

export interface TrainingLayerDetail {
  index: number;
  total_bytes: number;
  display: string;
  attention_count: number;
  mlp_count: number;
  norm_count: number;
  other_count: number;
  attention_bytes: number;
  mlp_bytes: number;
  norm_bytes: number;
  other_bytes: number;
  capabilities: string[];
  tensors: LayerTensorInfo[];
}

export interface LayerTensorInfo {
  name: string;
  dtype: string;
  shape: number[];
  memory_display: string;
  component: string;
}

export type TrainingMode = "finetune" | "surgery";
export type TrainingMethod = "sft" | "lora" | "qlora" | "dpo" | "full_finetune";
export type PresetId = "low_vram" | "balanced" | "quality" | "max_quality" | "custom";

export interface TrainingPreset {
  id: PresetId;
  name: string;
  desc: string;
  vram: string;
  method: TrainingMethod;
  learningRate: number;
  epochs: number;
  batchSize: number;
  gradientAccumulationSteps: number;
  maxSeqLength: number;
  warmupSteps: number;
  weightDecay: number;
  saveSteps: number;
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  quantizationBits: number;
}

export const TRAINING_PRESETS: TrainingPreset[] = [
  {
    id: "low_vram", name: "LOW VRAM", desc: "4-bit quantized, minimal footprint", vram: "~4 GB",
    method: "qlora", learningRate: 2e-4, epochs: 3, batchSize: 1, gradientAccumulationSteps: 16,
    maxSeqLength: 256, warmupSteps: 50, weightDecay: 0.01, saveSteps: 500,
    loraRank: 8, loraAlpha: 16, loraDropout: 0.05, quantizationBits: 4,
  },
  {
    id: "balanced", name: "BALANCED", desc: "Good quality/speed trade-off", vram: "~6 GB",
    method: "qlora", learningRate: 2e-4, epochs: 3, batchSize: 1, gradientAccumulationSteps: 8,
    maxSeqLength: 512, warmupSteps: 100, weightDecay: 0.01, saveSteps: 500,
    loraRank: 16, loraAlpha: 32, loraDropout: 0.05, quantizationBits: 4,
  },
  {
    id: "quality", name: "QUALITY", desc: "Higher rank, longer context", vram: "~12 GB",
    method: "lora", learningRate: 1e-4, epochs: 3, batchSize: 2, gradientAccumulationSteps: 4,
    maxSeqLength: 1024, warmupSteps: 100, weightDecay: 0.01, saveSteps: 500,
    loraRank: 32, loraAlpha: 64, loraDropout: 0.05, quantizationBits: 4,
  },
  {
    id: "max_quality", name: "MAX QUALITY", desc: "Full LoRA rank, large context", vram: "~24 GB",
    method: "lora", learningRate: 5e-5, epochs: 5, batchSize: 4, gradientAccumulationSteps: 4,
    maxSeqLength: 2048, warmupSteps: 200, weightDecay: 0.01, saveSteps: 250,
    loraRank: 64, loraAlpha: 128, loraDropout: 0.1, quantizationBits: 4,
  },
];

// ── Store ───────────────────────────────────────────

class TrainingStore {
  // Dependencies
  deps = $state<TrainingDepsStatus | null>(null);
  depsLoading = $state(false);

  // Setup
  setupRunning = $state(false);
  setupProgress = $state<TrainingProgress | null>(null);
  setupError = $state<string | null>(null);
  setupLogs = $state<string[]>([]);

  // Mode & status
  mode = $state<TrainingMode>("finetune");
  error = $state<string | null>(null);

  // Model selection
  modelPath = $state<string | null>(null);
  modelName = $state("");
  modelLayers = $state(0);
  modelArch = $state("");
  modelParams = $state("");

  // Dataset
  dataset = $state<DatasetInfo | null>(null);
  datasetLoading = $state(false);
  datasetError = $state<string | null>(null);

  // Training config
  method = $state<TrainingMethod>("lora");
  learningRate = $state(2e-4);
  epochs = $state(3);
  batchSize = $state(1);
  gradientAccumulationSteps = $state(8);
  maxSeqLength = $state(512);
  warmupSteps = $state(100);
  weightDecay = $state(0.01);
  saveSteps = $state(500);

  // LoRA config
  loraRank = $state(16);
  loraAlpha = $state(32);
  loraDropout = $state(0.05);
  targetModules = $state<string[]>(["q_proj", "v_proj"]);
  selectedLayers = $state<number[]>([]);

  // QLoRA
  quantizationBits = $state(4);

  // DPO
  dpoBeta = $state(0.1);

  // Output
  outputPath = $state("");
  mergeAdapter = $state(true);

  // Available modules/capabilities
  availableModules = $state<TargetModuleGroup[]>([]);
  layerCapabilities = $state<LayerCapabilityMapping[]>([]);
  capabilityToggles = $state<Record<string, boolean>>({});

  // Training progress
  training = $state(false);
  progress = $state<TrainingProgress | null>(null);
  result = $state<TrainingResult | null>(null);
  lossHistory = $state<{ step: number; loss: number }[]>([]);

  // Surgery state
  surgeryRunning = $state(false);
  surgeryResult = $state<SurgeryResult | null>(null);
  layersToRemove = $state<number[]>([]);
  layersToDuplicate = $state<{ source: number; insertAt: number }[]>([]);

  // Layer details for surgery view
  layerDetails = $state<TrainingLayerDetail[] | null>(null);
  layerDetailsLoading = $state(false);
  expandedSurgeryLayer = $state<number | null>(null);

  // Preset
  activePreset = $state<PresetId>("balanced");

  // Advanced panel visibility
  showAdvanced = $state(false);

  // Event listeners
  private setupUnlisten: UnlistenFn | null = null;
  private setupLogUnlisten: UnlistenFn | null = null;
  private progressUnlisten: UnlistenFn | null = null;
  private surgeryUnlisten: UnlistenFn | null = null;

  // ── Derived ─────────────────────────────────────

  get isLoraMethod(): boolean {
    return this.method === "lora" || this.method === "qlora";
  }

  get canTrain(): boolean {
    return (
      this.deps?.ready === true &&
      this.modelPath !== null &&
      this.dataset !== null &&
      this.outputPath !== "" &&
      !this.training
    );
  }

  get selectedLayersFromCapabilities(): number[] {
    const layers = new Set<number>();
    for (const [capId, enabled] of Object.entries(this.capabilityToggles)) {
      if (enabled) {
        const mapping = this.layerCapabilities.find((c) => c.capability === capId);
        if (mapping) {
          for (const l of mapping.layers) layers.add(l);
        }
      }
    }
    return Array.from(layers).sort((a, b) => a - b);
  }

  get surgeryPreview(): { final: number; removed: number; added: number } {
    const removed = this.layersToRemove.length;
    const added = this.layersToDuplicate.length;
    return {
      final: this.modelLayers - removed + added,
      removed,
      added,
    };
  }

  // ── Actions ─────────────────────────────────────

  async checkDeps() {
    this.depsLoading = true;
    try {
      this.deps = await invoke<TrainingDepsStatus>("training_check_deps");
    } catch (e) {
      console.error("Failed to check training deps:", e);
    } finally {
      this.depsLoading = false;
    }
  }

  async setup() {
    if (!this.setupUnlisten) {
      this.setupUnlisten = await listen<TrainingProgress>(
        "training:setup-progress",
        (e) => { this.setupProgress = e.payload; },
      );
    }

    if (!this.setupLogUnlisten) {
      this.setupLogUnlisten = await listen<string>(
        "training:setup-log",
        (e) => { this.setupLogs = [...this.setupLogs, e.payload]; },
      );
    }

    this.setupRunning = true;
    this.setupError = null;
    this.setupLogs = [];
    this.setupProgress = { stage: "setup", message: "Starting setup...", percent: 0, epoch: null, step: null, total_steps: null, loss: null, learning_rate: null, eta_seconds: null, gpu_memory_used_mb: null };

    try {
      await invoke("training_setup");
      await this.checkDeps();
    } catch (e) {
      this.setupError = String(e);
    } finally {
      this.setupRunning = false;
    }
  }

  async browseModel() {
    const result = await open({
      directory: true,
      title: "Select Model Directory (SafeTensors)",
    });
    if (result) {
      this.modelPath = result as string;
      this.modelName = (result as string).split("/").pop() ?? "";
      await this.loadModelCapabilities();
    }
  }

  async browseModelFile() {
    const result = await open({
      filters: [{ name: "GGUF Model", extensions: ["gguf"] }],
      title: "Select GGUF Model",
    });
    if (result) {
      this.modelPath = result as string;
      this.modelName = (result as string).split("/").pop() ?? "";
      await this.loadModelCapabilities();
    }
  }

  async browseDataset() {
    const result = await open({
      filters: [
        { name: "Dataset", extensions: ["json", "jsonl", "csv", "parquet"] },
      ],
      title: "Select Training Dataset",
    });
    if (result) {
      await this.detectDataset(result as string);
    }
  }

  async detectDataset(path: string) {
    this.datasetLoading = true;
    this.datasetError = null;
    this.dataset = null;
    try {
      this.dataset = await invoke<DatasetInfo>("training_detect_dataset", { path });
    } catch (e) {
      this.datasetError = String(e);
    } finally {
      this.datasetLoading = false;
    }
  }

  async loadModelCapabilities() {
    if (!this.modelPath) return;

    try {
      this.availableModules = await invoke<TargetModuleGroup[]>("training_get_target_modules", { modelPath: this.modelPath });
      // Auto-select first group's modules as default targets
      if (this.availableModules.length > 0) {
        this.targetModules = this.availableModules[0].modules.slice();
      }
    } catch {
      this.availableModules = [];
    }

    try {
      this.layerCapabilities = await invoke<LayerCapabilityMapping[]>("training_get_layer_capabilities", { modelPath: this.modelPath });
      // Derive layer count from capabilities
      if (this.layerCapabilities.length > 0) {
        const allLayers = this.layerCapabilities.flatMap(c => c.layers);
        if (allLayers.length > 0) {
          this.modelLayers = Math.max(...allLayers) + 1;
        }
      }
    } catch {
      this.layerCapabilities = [];
    }

    await this.loadLayerDetails();
  }

  async loadLayerDetails() {
    if (!this.modelPath) return;
    this.layerDetailsLoading = true;
    try {
      this.layerDetails = await invoke<TrainingLayerDetail[]>("training_get_layer_details", { modelPath: this.modelPath });
    } catch {
      this.layerDetails = null;
    } finally {
      this.layerDetailsLoading = false;
    }
  }

  async selectOutputPath() {
    const result = await open({
      directory: true,
      title: "Select Output Directory",
    });
    if (result) {
      this.outputPath = result as string;
    }
  }

  toggleCapability(capId: string) {
    this.capabilityToggles = {
      ...this.capabilityToggles,
      [capId]: !this.capabilityToggles[capId],
    };
    // Update selected layers from capability toggles
    this.selectedLayers = this.selectedLayersFromCapabilities;
  }

  toggleModule(moduleName: string) {
    if (this.targetModules.includes(moduleName)) {
      this.targetModules = this.targetModules.filter((m) => m !== moduleName);
    } else {
      this.targetModules = [...this.targetModules, moduleName];
    }
  }

  applyPreset(preset: TrainingPreset) {
    this.activePreset = preset.id;
    this.method = preset.method;
    this.learningRate = preset.learningRate;
    this.epochs = preset.epochs;
    this.batchSize = preset.batchSize;
    this.gradientAccumulationSteps = preset.gradientAccumulationSteps;
    this.maxSeqLength = preset.maxSeqLength;
    this.warmupSteps = preset.warmupSteps;
    this.weightDecay = preset.weightDecay;
    this.saveSteps = preset.saveSteps;
    this.loraRank = preset.loraRank;
    this.loraAlpha = preset.loraAlpha;
    this.loraDropout = preset.loraDropout;
    this.quantizationBits = preset.quantizationBits;
  }

  async run() {
    if (!this.progressUnlisten) {
      this.progressUnlisten = await listen<TrainingProgress>(
        "training:progress",
        (e) => {
          this.progress = e.payload;
          if (e.payload.loss !== null && e.payload.step !== null) {
            this.lossHistory = [
              ...this.lossHistory,
              { step: e.payload.step, loss: e.payload.loss },
            ];
          }
        },
      );
    }

    this.training = true;
    this.error = null;
    this.result = null;
    this.lossHistory = [];
    this.progress = {
      stage: "starting",
      message: "Preparing training...",
      percent: 0,
      epoch: null, step: null, total_steps: null,
      loss: null, learning_rate: null,
      eta_seconds: null, gpu_memory_used_mb: null,
    };

    const config: any = {
      model_path: this.modelPath,
      dataset_path: this.dataset?.path,
      dataset_format: this.dataset?.format ?? "jsonl",
      method: this.method,
      output_path: this.outputPath,
      merge_adapter: this.mergeAdapter,
      learning_rate: this.learningRate,
      epochs: this.epochs,
      batch_size: this.batchSize,
      gradient_accumulation_steps: this.gradientAccumulationSteps,
      max_seq_length: this.maxSeqLength,
      warmup_steps: this.warmupSteps,
      weight_decay: this.weightDecay,
      save_steps: this.saveSteps,
    };

    if (this.isLoraMethod) {
      config.lora_rank = this.loraRank;
      config.lora_alpha = this.loraAlpha;
      config.lora_dropout = this.loraDropout;
      config.target_modules = this.targetModules;
      if (this.selectedLayers.length > 0) {
        config.layers_to_transform = this.selectedLayers;
      }
    }

    if (this.method === "qlora") {
      config.quantization_bits = this.quantizationBits;
    }

    if (this.method === "dpo") {
      config.dpo_beta = this.dpoBeta;
    }

    try {
      this.result = await invoke<TrainingResult>("training_run", { config });
    } catch (e) {
      const msg = String(e);
      if (!msg.includes("cancelled")) {
        this.error = msg;
      }
    } finally {
      this.training = false;
    }
  }

  async cancel() {
    try {
      await invoke("training_cancel");
    } catch {
      // ignore
    }
  }

  // Surgery actions

  toggleLayerRemove(index: number) {
    if (this.layersToRemove.includes(index)) {
      this.layersToRemove = this.layersToRemove.filter((i) => i !== index);
    } else {
      this.layersToRemove = [...this.layersToRemove, index];
    }
  }

  addDuplicate(sourceIndex: number) {
    this.layersToDuplicate = [
      ...this.layersToDuplicate,
      { source: sourceIndex, insertAt: sourceIndex + 1 },
    ];
  }

  removeDuplicate(index: number) {
    this.layersToDuplicate = this.layersToDuplicate.filter((_, i) => i !== index);
  }

  async runSurgery() {
    if (!this.surgeryUnlisten) {
      this.surgeryUnlisten = await listen<any>(
        "training:surgery-progress",
        (e) => {
          this.progress = { ...e.payload, epoch: null, step: null, total_steps: null, loss: null, learning_rate: null, eta_seconds: null, gpu_memory_used_mb: null };
        },
      );
    }

    this.surgeryRunning = true;
    this.error = null;
    this.surgeryResult = null;

    const operations: any[] = [];
    for (const idx of this.layersToRemove.sort((a, b) => a - b)) {
      operations.push({ remove_layer: { index: idx } });
    }
    for (const dup of this.layersToDuplicate) {
      operations.push({ duplicate_layer: { source_index: dup.source, insert_at: dup.insertAt } });
    }

    try {
      this.surgeryResult = await invoke<SurgeryResult>("training_surgery_run", {
        config: {
          model_path: this.modelPath,
          output_path: this.outputPath,
          operations,
        },
      });
    } catch (e) {
      const msg = String(e);
      if (!msg.includes("cancelled")) {
        this.error = msg;
      }
    } finally {
      this.surgeryRunning = false;
    }
  }

  async cancelSurgery() {
    try {
      await invoke("training_surgery_cancel");
    } catch {
      // ignore
    }
  }

  reset() {
    this.error = null;
    this.progress = null;
    this.result = null;
    this.surgeryResult = null;
    this.lossHistory = [];
    this.layersToRemove = [];
    this.layersToDuplicate = [];
  }

  destroy() {
    this.setupUnlisten?.();
    this.setupLogUnlisten?.();
    this.progressUnlisten?.();
    this.surgeryUnlisten?.();
    this.setupUnlisten = null;
    this.setupLogUnlisten = null;
    this.progressUnlisten = null;
    this.surgeryUnlisten = null;
  }
}

export const training = new TrainingStore();
