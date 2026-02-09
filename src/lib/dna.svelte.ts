import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

// ── Types ────────────────────────────────────────────────

export interface ParentModelInfo {
  id: string;
  slot: number;
  name: string;
  file_path: string;
  format: string;
  file_size: number;
  file_size_display: string;
  parameter_count: number;
  parameter_count_display: string;
  layer_count: number | null;
  architecture: string | null;
  quantization: string | null;
  color: string;
  tensor_count: number;
}

export interface MergeMethodInfo {
  id: string;
  name: string;
  description: string;
  requires_base: boolean;
  min_parents: number;
  difficulty: string;
}

export interface CompatReport {
  compatible: boolean;
  warnings: string[];
  errors: string[];
  shared_tensor_count: number;
  total_tensor_count: number;
  architecture_match: boolean;
  dimension_match: boolean;
  layer_count_match: boolean;
}

export interface LayerProfile {
  layer_index: number;
  top_predictions: { token: string; probability: number; rank: number }[];
  entropy: number;
  specialization: string;
  confidence: number;
}

export interface MergeProgress {
  stage: string;
  percent: number;
  message: string;
  current_tensor: string | null;
  tensors_done: number;
  tensors_total: number;
}

export interface MergeResult {
  output_path: string;
  output_size: number;
  output_size_display: string;
  tensors_written: number;
  method: string;
  copied_files: string[];
}

export interface MergePreview {
  total_operations: number;
  copy_operations: number;
  merge_operations: number;
  synthesize_operations: number;
  estimated_output_bytes: number;
  estimated_output_display: string;
}

export interface LayerAssignment {
  layerIndex: number;
  sourceParentId: string;
}

export interface LayerComponentInfo {
  layer_index: number;
  attention_tensors: string[];
  mlp_tensors: string[];
  norm_tensors: string[];
  other_tensors: string[];
  total_params: number;
}

export interface HoveredLayer {
  parentId: string | null;
  layerIndex: number;
}

export interface LayerAnalysis {
  layer_index: number;
  category: string;
  label: string;
  color: string;
  description: string;
  confidence: number;
  attn_tensors: number;
  mlp_tensors: number;
  norm_tensors: number;
  norm_l2: number;
  norm_variance: number;
  mlp_dominance: number;
}

export interface AnalysisResult {
  parent_id: string;
  parent_name: string;
  layers: LayerAnalysis[];
  total_layers: number;
  categories: LayerCategoryInfo[];
}

export interface LayerCategoryInfo {
  id: string;
  label: string;
  color: string;
  description: string;
}

export interface AnalysisProgress {
  parent_id: string;
  layer_index: number;
  total_layers: number;
  percent: number;
  stage: string;
}

export type DnaStatus = "idle" | "loading" | "ready" | "merging" | "profiling" | "analyzing" | "error" | "complete";
export type DnaMode = "easy" | "intermediate" | "advanced";
export type DnaTab = "files" | "layers" | "settings";

// ── Store ────────────────────────────────────────────────

class DnaStore {
  // Core state
  mode = $state<DnaMode>("easy");
  status = $state<DnaStatus>("idle");
  error = $state<string | null>(null);

  // Parent models (2-5)
  parents = $state<ParentModelInfo[]>([]);

  // Layer assignments
  layerAssignments = $state<LayerAssignment[]>([]);

  // Merge config
  selectedMethod = $state("slerp");
  methodParams = $state<Record<string, any>>({});
  baseParentId = $state<string | null>(null);
  outputFormat = $state<"safe_tensors" | "gguf">("safe_tensors");
  outputPath = $state("");
  modelName = $state("merged-model");

  // Available methods
  methods = $state<MergeMethodInfo[]>([]);

  // Compatibility
  compatReport = $state<CompatReport | null>(null);

  // Profiling
  profiles = $state<LayerProfile[]>([]);
  profiling = $state(false);

  // Merge execution
  merging = $state(false);
  mergeProgress = $state<MergeProgress | null>(null);
  mergeResult = $state<MergeResult | null>(null);

  // Preview
  preview = $state<MergePreview | null>(null);

  // Per-parent layer component data
  layerComponents = $state<Record<string, LayerComponentInfo[]>>({});

  // Layer analysis (computed via COMPUTE LAYERS)
  layerAnalysis = $state<Record<string, LayerAnalysis[]>>({});
  analyzing = $state(false);
  analysisProgress = $state<AnalysisProgress | null>(null);
  categories = $state<LayerCategoryInfo[]>([]);

  // UI state
  activeTab = $state<DnaTab>("files");
  hoveredLayer = $state<HoveredLayer | null>(null);
  selectedParentForProfile = $state<string | null>(null);

  // Event listeners
  private progressUnlisten: UnlistenFn | null = null;
  private phaseUnlisten: UnlistenFn | null = null;
  private profileProgressUnlisten: UnlistenFn | null = null;
  private profileLayerUnlisten: UnlistenFn | null = null;
  private analysisProgressUnlisten: UnlistenFn | null = null;
  private layerAnalyzedUnlisten: UnlistenFn | null = null;

  // ── Derived ──────────────────────────────────────────

  get loadedCount(): number {
    return this.parents.length;
  }

  get maxLayers(): number {
    return Math.max(
      ...this.parents.map((p) => p.layer_count ?? 0),
      0
    );
  }

  get canMerge(): boolean {
    const method = this.methods.find((m) => m.id === this.selectedMethod);
    const minParents = method?.min_parents ?? 2;
    if (this.parents.length < minParents || this.outputPath === "") return false;
    // Frankenmerge requires layer assignments
    if (this.selectedMethod === "frankenmerge" && this.layerAssignments.length === 0) return false;
    return true;
  }

  get canProfile(): boolean {
    return this.parents.length >= 1 && !this.profiling;
  }

  get canAnalyze(): boolean {
    return this.parents.length >= 1 && !this.analyzing;
  }

  get isAnalyzed(): boolean {
    return this.parents.length > 0 && this.parents.every(p => this.layerAnalysis[p.id]?.length > 0);
  }

  get hasGaps(): boolean {
    if (this.layerAssignments.length === 0) return false;
    const max = this.maxLayers;
    const assigned = new Set(this.layerAssignments.map((a) => a.layerIndex));
    for (let i = 0; i < max; i++) {
      if (!assigned.has(i)) return true;
    }
    return false;
  }

  get compositionStats(): { parentId: string; percentage: number; color: string; name: string }[] {
    const total = this.layerAssignments.length || 1;
    const counts: Record<string, number> = {};
    for (const a of this.layerAssignments) {
      counts[a.sourceParentId] = (counts[a.sourceParentId] || 0) + 1;
    }
    return this.parents.map((p) => ({
      parentId: p.id,
      percentage: ((counts[p.id] || 0) / total) * 100,
      color: p.color,
      name: p.name,
    }));
  }

  get parentWeights(): { parent_id: string; weight: number }[] {
    const n = this.parents.length;
    if (n === 0) return [];
    const equalWeight = 1.0 / n;
    return this.parents.map((p) => ({
      parent_id: p.id,
      weight: equalWeight,
    }));
  }

  // ── Actions ──────────────────────────────────────────

  async fetchLayerComponents(parentId: string) {
    try {
      const components = await invoke<LayerComponentInfo[]>(
        "merge_get_layer_components",
        { parentId }
      );
      this.layerComponents = { ...this.layerComponents, [parentId]: components };
    } catch (e) {
      console.error(`Failed to fetch layer components for ${parentId}:`, e);
    }
  }

  getLayerTooltipData(parentId: string, layerIndex: number) {
    const components = this.layerComponents[parentId];
    if (!components) return null;
    const layer = components.find(c => c.layer_index === layerIndex);
    if (!layer) return null;

    const profile = this.profiles.find(p => p.layer_index === layerIndex);
    return {
      attn: layer.attention_tensors,
      mlp: layer.mlp_tensors,
      norm: layer.norm_tensors,
      other: layer.other_tensors,
      totalTensors: layer.attention_tensors.length + layer.mlp_tensors.length + layer.norm_tensors.length + layer.other_tensors.length,
      specialization: profile?.specialization ?? null,
      entropy: profile?.entropy ?? null,
      confidence: profile?.confidence ?? null,
    };
  }

  getLayerAnalysis(parentId: string, layerIndex: number): LayerAnalysis | null {
    const analyses = this.layerAnalysis[parentId];
    if (!analyses) return null;
    return analyses.find(a => a.layer_index === layerIndex) ?? null;
  }

  async init() {
    try {
      this.methods = await invoke<MergeMethodInfo[]>("merge_get_methods");
      const existing = await invoke<ParentModelInfo[]>("merge_get_parents");
      this.parents = existing;
      for (const p of existing) {
        this.fetchLayerComponents(p.id);
      }
      if (existing.length >= 2) {
        this.status = "ready";
        await this.checkCompatibility();
      }
    } catch (e) {
      console.error("Failed to init DNA store:", e);
    }
  }

  async addParent() {
    this.error = null;
    try {
      const result = await open({
        multiple: false,
        directory: false,
        filters: [
          { name: "Model Files", extensions: ["safetensors", "gguf"] },
        ],
      });

      if (!result) return;
      const filePath = typeof result === "string" ? result : result.path;
      if (!filePath) return;

      this.status = "loading";
      const slot = this.parents.length;

      const parent = await invoke<ParentModelInfo>("merge_load_parent", {
        path: filePath,
        slot,
      });

      this.parents = [...this.parents, parent];
      this.fetchLayerComponents(parent.id);

      if (this.parents.length >= 2) {
        this.status = "ready";
        await this.checkCompatibility();
      } else {
        this.status = "idle";
      }
    } catch (e) {
      this.error = String(e);
      this.status = "error";
    }
  }

  async addParentDir() {
    this.error = null;
    try {
      const result = await open({
        multiple: false,
        directory: true,
      });

      if (!result) return;
      const dirPath = typeof result === "string" ? result : result.path;
      if (!dirPath) return;

      this.status = "loading";
      const slot = this.parents.length;

      const parent = await invoke<ParentModelInfo>("merge_load_parent_dir", {
        path: dirPath,
        slot,
      });

      this.parents = [...this.parents, parent];
      this.fetchLayerComponents(parent.id);

      if (this.parents.length >= 2) {
        this.status = "ready";
        await this.checkCompatibility();
      } else {
        this.status = "idle";
      }
    } catch (e) {
      this.error = String(e);
      this.status = "error";
    }
  }

  async removeParent(id: string) {
    try {
      await invoke("merge_remove_parent", { parentId: id });
      this.parents = this.parents.filter((p) => p.id !== id);
      this.layerAssignments = this.layerAssignments.filter(
        (a) => a.sourceParentId !== id
      );
      const { [id]: _, ...rest } = this.layerComponents;
      this.layerComponents = rest;
      const { [id]: _a, ...restAnalysis } = this.layerAnalysis;
      this.layerAnalysis = restAnalysis;
      if (this.baseParentId === id) this.baseParentId = null;
      if (this.parents.length >= 2) {
        await this.checkCompatibility();
      } else {
        this.compatReport = null;
      }
    } catch (e) {
      this.error = String(e);
    }
  }

  async checkCompatibility() {
    try {
      this.compatReport = await invoke<CompatReport>("merge_check_compatibility");
    } catch (e) {
      console.error("Compatibility check failed:", e);
    }
  }

  assignLayer(layerIndex: number, parentId: string) {
    const existing = this.layerAssignments.findIndex(
      (a) => a.layerIndex === layerIndex
    );
    if (existing >= 0) {
      this.layerAssignments[existing].sourceParentId = parentId;
    } else {
      this.layerAssignments = [
        ...this.layerAssignments,
        { layerIndex, sourceParentId: parentId },
      ];
    }
  }

  autoAssign(strategy: "interleave" | "split" | "first") {
    const max = this.maxLayers;
    if (max === 0 || this.parents.length === 0) return;

    const assignments: LayerAssignment[] = [];

    if (strategy === "interleave") {
      for (let i = 0; i < max; i++) {
        const parent = this.parents[i % this.parents.length];
        assignments.push({ layerIndex: i, sourceParentId: parent.id });
      }
    } else if (strategy === "split") {
      const perParent = Math.ceil(max / this.parents.length);
      for (let i = 0; i < max; i++) {
        const parentIdx = Math.min(
          Math.floor(i / perParent),
          this.parents.length - 1
        );
        assignments.push({
          layerIndex: i,
          sourceParentId: this.parents[parentIdx].id,
        });
      }
    } else {
      // "first" — all from first parent
      for (let i = 0; i < max; i++) {
        assignments.push({
          layerIndex: i,
          sourceParentId: this.parents[0].id,
        });
      }
    }

    this.layerAssignments = assignments;
  }

  async runProfiling() {
    const parentId = this.selectedParentForProfile || this.parents[0]?.id;
    if (!parentId) return;

    this.error = null;
    this.profiling = true;
    this.profiles = [];
    this.status = "profiling";

    if (!this.profileLayerUnlisten) {
      this.profileLayerUnlisten = await listen<LayerProfile>(
        "merge:profile-layer-done",
        (e) => {
          this.profiles = [...this.profiles, e.payload];
        }
      );
    }

    try {
      await invoke("merge_profile_layers", { parentId });
    } catch (e) {
      const msg = String(e);
      if (!msg.includes("cancelled")) {
        this.error = msg;
      }
    } finally {
      this.profiling = false;
      this.status = this.parents.length >= 2 ? "ready" : "idle";
    }
  }

  async merge() {
    if (!this.canMerge) return;

    this.error = null;
    this.merging = true;
    this.mergeProgress = null;
    this.mergeResult = null;
    this.status = "merging";

    if (!this.progressUnlisten) {
      this.progressUnlisten = await listen<MergeProgress>(
        "merge:progress",
        (e) => {
          this.mergeProgress = e.payload;
        }
      );
    }

    const config = {
      parents: this.parentWeights,
      method: this.selectedMethod,
      params: this.methodParams,
      base_parent_id: this.baseParentId,
      layer_assignments: this.layerAssignments.map((a) => ({
        layer_index: a.layerIndex,
        source_parent_id: a.sourceParentId,
      })),
      component_overrides: [],
      tensor_overrides: [],
      output: {
        format: this.outputFormat,
        path: this.outputPath,
        model_name: this.modelName,
      },
    };

    // Fire-and-forget: don't block the UI on the merge result.
    // This prevents freeze when navigating away and back during merge.
    invoke<MergeResult>("merge_execute", { config })
      .then((result) => {
        this.mergeResult = result;
        this.status = "complete";
        this.merging = false;
      })
      .catch((e) => {
        const msg = String(e);
        if (msg.includes("cancelled")) {
          this.status = this.parents.length >= 2 ? "ready" : "idle";
        } else {
          this.error = msg;
          this.status = "error";
        }
        this.merging = false;
      });
  }

  async cancelMerge() {
    try {
      await invoke("merge_cancel");
    } catch {
      // ignore
    }
  }

  async cancelProfile() {
    try {
      await invoke("merge_profile_cancel");
    } catch {
      // ignore
    }
  }

  async analyzeLayers() {
    if (!this.canAnalyze) return;

    this.error = null;
    this.analyzing = true;
    this.analysisProgress = null;
    this.status = "analyzing";

    // Listen for progress events
    if (!this.analysisProgressUnlisten) {
      this.analysisProgressUnlisten = await listen<AnalysisProgress>(
        "merge:analysis-progress",
        (e) => {
          this.analysisProgress = e.payload;
        }
      );
    }

    if (!this.layerAnalyzedUnlisten) {
      this.layerAnalyzedUnlisten = await listen<LayerAnalysis>(
        "merge:layer-analyzed",
        (_e) => {
          // Individual layer updates come via the full result
        }
      );
    }

    // Fire-and-forget: run analysis without blocking UI
    const parentsCopy = [...this.parents];
    const analyzeSequentially = async () => {
      try {
        for (const parent of parentsCopy) {
          const result = await invoke<AnalysisResult>("merge_analyze_layers", {
            parentId: parent.id,
          });
          this.layerAnalysis = { ...this.layerAnalysis, [parent.id]: result.layers };
          if (result.categories.length > 0) {
            this.categories = result.categories;
          }
        }
      } catch (e) {
        const msg = String(e);
        if (!msg.includes("cancelled")) {
          this.error = msg;
        }
      } finally {
        this.analyzing = false;
        this.analysisProgress = null;
        this.status = this.parents.length >= 2 ? "ready" : "idle";
      }
    };
    analyzeSequentially();
  }

  cancelAnalysis() {
    // Reuse profiler cancel since they share the same cancel flag
    try {
      invoke("merge_profile_cancel");
    } catch {
      // ignore
    }
  }

  async getPreview() {
    if (this.parents.length < 2) return;

    const config = {
      parents: this.parentWeights,
      method: this.selectedMethod,
      params: this.methodParams,
      base_parent_id: this.baseParentId,
      layer_assignments: this.layerAssignments.map((a) => ({
        layer_index: a.layerIndex,
        source_parent_id: a.sourceParentId,
      })),
      component_overrides: [],
      tensor_overrides: [],
      output: {
        format: this.outputFormat,
        path: this.outputPath || "/tmp/preview",
        model_name: this.modelName,
      },
    };

    try {
      this.preview = await invoke<MergePreview>("merge_preview", { config });
    } catch (e) {
      console.error("Preview failed:", e);
    }
  }

  async selectOutputPath() {
    try {
      const result = await open({
        multiple: false,
        directory: true,
      });

      if (!result) return;
      const dir = typeof result === "string" ? result : result.path;
      if (!dir) return;

      if (this.outputFormat === "safe_tensors") {
        // SafeTensors outputs a HF-compatible directory (model.safetensors + config/tokenizer inside)
        this.outputPath = `${dir}/${this.modelName}`;
      } else {
        this.outputPath = `${dir}/${this.modelName}.gguf`;
      }
    } catch (e) {
      console.error("Failed to select output path:", e);
    }
  }

  reset() {
    this.status = "idle";
    this.error = null;
    this.layerAssignments = [];
    this.layerComponents = {};
    this.layerAnalysis = {};
    this.categories = [];
    this.profiles = [];
    this.mergeProgress = null;
    this.mergeResult = null;
    this.preview = null;
    this.hoveredLayer = null;
    this.analysisProgress = null;
  }

  destroy() {
    this.progressUnlisten?.();
    this.phaseUnlisten?.();
    this.profileProgressUnlisten?.();
    this.profileLayerUnlisten?.();
    this.analysisProgressUnlisten?.();
    this.layerAnalyzedUnlisten?.();
  }
}

export const dna = new DnaStore();
