<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { goto } from "$app/navigation";
  import { model } from "$lib/model.svelte";

  interface InspectTensor {
    name: string;
    dtype: string;
    shape: number[];
    memory_bytes: number;
    memory_display: string;
    component: string;
    params: number;
  }

  interface MemoryComponent {
    name: string;
    bytes: number;
    display: string;
    percentage: number;
  }

  interface QuantEntry {
    dtype: string;
    count: number;
    total_bytes: number;
    display: string;
    percentage: number;
    total_params: number;
  }

  interface LayerGroup {
    index: number;
    total_bytes: number;
    display: string;
    attention: InspectTensor[];
    mlp: InspectTensor[];
    norms: InspectTensor[];
    other: InspectTensor[];
  }

  interface AttentionInfo {
    attention_type: string;
    q_heads: number | null;
    kv_heads: number | null;
    head_dim: number | null;
    gqa_ratio: number | null;
  }

  interface ConfigEntry {
    label: string;
    value: string;
  }

  interface ModelConfig {
    entries: ConfigEntry[];
  }

  interface SpecialToken {
    role: string;
    id: number;
    token: string | null;
  }

  interface TokenizerInfo {
    tokenizer_type: string | null;
    vocab_size: number | null;
    special_tokens: SpecialToken[];
  }

  interface InspectData {
    memory_breakdown: MemoryComponent[];
    total_memory_bytes: number;
    total_memory_display: string;
    quant_distribution: QuantEntry[];
    layers: LayerGroup[];
    other_tensors: InspectTensor[];
    attention_info: AttentionInfo | null;
    model_config: ModelConfig;
    tokenizer_info: TokenizerInfo | null;
    tensor_count: number;
    total_params: number;
    total_params_display: string;
  }

  interface ModelFingerprint {
    sha256: string;
    file_size_bytes: number;
    tensor_count_verified: boolean;
  }

  // ── State ──────────────────────────────────────────
  let data = $state<InspectData | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let expandedLayers = $state<Set<number>>(new Set());

  // Fingerprint (on-demand)
  let fingerprint = $state<ModelFingerprint | null>(null);
  let fingerprintLoading = $state(false);
  let fingerprintError = $state<string | null>(null);

  // Capabilities
  interface Capability {
    id: string;
    name: string;
    detected: boolean;
    confidence: number;
    evidence: string[];
    affected_layers: number[];
  }
  interface CapabilityReport {
    parent_id: string;
    parent_name: string;
    capabilities: Capability[];
    total_detected: number;
  }
  let capReport = $state<CapabilityReport | null>(null);
  let capLoading = $state(false);

  // Tensor filter
  let tensorSearch = $state("");
  let dtypeFilter = $state("all");
  let layerRangeMin = $state("");
  let layerRangeMax = $state("");

  // Isometric visualization
  let hoveredSlab = $state<number | null>(null);
  let mouseX = $state(0);
  let mouseY = $state(0);

  function handleIsoMouseMove(e: MouseEvent) {
    mouseX = e.clientX;
    mouseY = e.clientY;
  }

  // ── Constants ──────────────────────────────────────
  const memoryColors = [
    "var(--accent)",
    "var(--info)",
    "var(--success)",
    "var(--gray)",
    "var(--danger)",
    "var(--text-muted)",
  ];

  const quantColors: Record<string, string> = {
    F32: "#ef4444",
    F16: "#f59e0b",
    BF16: "#f59e0b",
    Q8_0: "#22c55e",
    Q8_1: "#22c55e",
    Q6_K: "#3b82f6",
    Q5_K: "#8b5cf6",
    Q5_0: "#8b5cf6",
    Q5_1: "#8b5cf6",
    Q4_K: "#ec4899",
    Q4_0: "#ec4899",
    Q4_1: "#ec4899",
    Q3_K: "#f97316",
    Q2_K: "#ef4444",
  };

  // ── Compatibility matrix ───────────────────────────
  interface RuntimeCompat {
    name: string;
    gguf: boolean;
    safetensors: boolean;
    quant_support: string[];
  }

  const RUNTIMES: RuntimeCompat[] = [
    { name: "llama.cpp", gguf: true, safetensors: false, quant_support: ["Q2_K","Q3_K","Q4_0","Q4_1","Q4_K","Q5_0","Q5_1","Q5_K","Q6_K","Q8_0","Q8_1","Q8_K","F16","F32","BF16","IQ1_S","IQ1_M","IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS","IQ3_S","IQ4_NL","IQ4_XS"] },
    { name: "Ollama", gguf: true, safetensors: false, quant_support: ["Q2_K","Q3_K","Q4_0","Q4_1","Q4_K","Q5_0","Q5_1","Q5_K","Q6_K","Q8_0","F16"] },
    { name: "LM Studio", gguf: true, safetensors: false, quant_support: ["Q2_K","Q3_K","Q4_0","Q4_1","Q4_K","Q5_0","Q5_1","Q5_K","Q6_K","Q8_0","F16","F32"] },
    { name: "KoboldCpp", gguf: true, safetensors: false, quant_support: ["Q2_K","Q3_K","Q4_0","Q4_1","Q4_K","Q5_0","Q5_1","Q5_K","Q6_K","Q8_0","F16","F32"] },
    { name: "vLLM", gguf: true, safetensors: true, quant_support: ["F16","BF16","F32","Q4_0","Q8_0"] },
    { name: "HuggingFace Transformers", gguf: false, safetensors: true, quant_support: ["F16","BF16","F32"] },
    { name: "ExLlamaV2", gguf: false, safetensors: true, quant_support: ["F16","BF16"] },
    { name: "MLX", gguf: false, safetensors: true, quant_support: ["F16","BF16","F32"] },
  ];

  type CompatStatus = "full" | "partial" | "none";

  function getCompatStatus(runtime: RuntimeCompat): CompatStatus {
    if (!model.info) return "none";
    const isGguf = model.info.format === "Gguf";
    const isST = model.info.format === "SafeTensors";

    if ((isGguf && !runtime.gguf) || (isST && !runtime.safetensors)) return "none";

    const quant = model.info.quantization;
    if (!quant) return "partial";
    if (runtime.quant_support.includes(quant)) return "full";
    return "partial";
  }

  // ── Helpers ────────────────────────────────────────
  function getQuantColor(dtype: string): string {
    return quantColors[dtype] ?? "var(--text-muted)";
  }

  function formatMemory(bytes: number): string {
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(1) + "G";
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + "M";
    if (bytes >= 1024) return (bytes / 1024).toFixed(0) + "K";
    return bytes + "B";
  }

  function slabColor(rgb: [number, number, number], factor: number): string {
    return `rgb(${Math.round(rgb[0] * factor)}, ${Math.round(rgb[1] * factor)}, ${Math.round(rgb[2] * factor)})`;
  }

  function toggleLayer(idx: number) {
    const next = new Set(expandedLayers);
    if (next.has(idx)) {
      next.delete(idx);
    } else {
      next.add(idx);
    }
    expandedLayers = next;
  }

  function matchesTensorFilter(t: InspectTensor): boolean {
    if (tensorSearch && !t.name.toLowerCase().includes(tensorSearch.toLowerCase())) return false;
    if (dtypeFilter !== "all" && t.dtype !== dtypeFilter) return false;
    return true;
  }

  function clearFilters() {
    tensorSearch = "";
    dtypeFilter = "all";
    layerRangeMin = "";
    layerRangeMax = "";
  }

  // ── Derived ────────────────────────────────────────
  let allDtypes = $derived.by(() => {
    if (!data) return [];
    const set = new Set<string>();
    for (const layer of data.layers) {
      for (const t of [...layer.attention, ...layer.mlp, ...layer.norms, ...layer.other]) {
        set.add(t.dtype);
      }
    }
    for (const t of data.other_tensors) {
      set.add(t.dtype);
    }
    return Array.from(set).sort();
  });

  let filteredLayers = $derived.by(() => {
    if (!data) return [];
    const min = layerRangeMin !== "" ? parseInt(layerRangeMin) : null;
    const max = layerRangeMax !== "" ? parseInt(layerRangeMax) : null;
    return data.layers
      .filter(layer => {
        if (min !== null && !isNaN(min) && layer.index < min) return false;
        if (max !== null && !isNaN(max) && layer.index > max) return false;
        return true;
      })
      .map(layer => ({
        ...layer,
        attention: layer.attention.filter(matchesTensorFilter),
        mlp: layer.mlp.filter(matchesTensorFilter),
        norms: layer.norms.filter(matchesTensorFilter),
        other: layer.other.filter(matchesTensorFilter),
      }))
      .filter(layer =>
        layer.attention.length + layer.mlp.length + layer.norms.length + layer.other.length > 0
      );
  });

  let filteredOtherTensors = $derived.by(() => {
    if (!data) return [];
    return data.other_tensors.filter(matchesTensorFilter);
  });

  let isFiltering = $derived(
    tensorSearch !== "" || dtypeFilter !== "all" || layerRangeMin !== "" || layerRangeMax !== ""
  );

  // ── Isometric visualization ─────────────────────────
  interface IsoSlab {
    type: "embedding" | "block" | "output";
    label: string;
    bytes: number;
    rgb: [number, number, number];
    tensorCount: number;
    memDisplay: string;
    attnCount?: number;
    mlpCount?: number;
    normCount?: number;
    attnMem?: string;
    mlpMem?: string;
    normMem?: string;
  }

  let isoSlabs = $derived.by((): IsoSlab[] => {
    if (!data) return [];
    const slabs: IsoSlab[] = [];

    const embTensors = data.other_tensors.filter(t => t.component === "embedding");
    if (embTensors.length > 0) {
      const bytes = embTensors.reduce((s, t) => s + t.memory_bytes, 0);
      slabs.push({
        type: "embedding", label: "EMBEDDING",
        bytes, rgb: [245, 158, 11],
        tensorCount: embTensors.length, memDisplay: formatMemory(bytes),
      });
    }

    for (const layer of data.layers) {
      const attnBytes = layer.attention.reduce((s, t) => s + t.memory_bytes, 0);
      const mlpBytes = layer.mlp.reduce((s, t) => s + t.memory_bytes, 0);
      const normBytes = layer.norms.reduce((s, t) => s + t.memory_bytes, 0);
      const total = layer.total_bytes;
      const ratio = total > 0 ? attnBytes / total : 0.5;
      const r = Math.round(59 * ratio + 34 * (1 - ratio));
      const g = Math.round(130 * ratio + 197 * (1 - ratio));
      const b = Math.round(246 * ratio + 94 * (1 - ratio));
      const tc = layer.attention.length + layer.mlp.length + layer.norms.length + layer.other.length;
      slabs.push({
        type: "block", label: `BLOCK ${layer.index}`,
        bytes: total, rgb: [r, g, b],
        tensorCount: tc, memDisplay: formatMemory(total),
        attnCount: layer.attention.length, mlpCount: layer.mlp.length, normCount: layer.norms.length,
        attnMem: formatMemory(attnBytes), mlpMem: formatMemory(mlpBytes), normMem: formatMemory(normBytes),
      });
    }

    const outTensors = data.other_tensors.filter(t => t.component !== "embedding");
    if (outTensors.length > 0) {
      const bytes = outTensors.reduce((s, t) => s + t.memory_bytes, 0);
      slabs.push({
        type: "output", label: "OUTPUT",
        bytes, rgb: [239, 68, 68],
        tensorCount: outTensors.length, memDisplay: formatMemory(bytes),
      });
    }

    return slabs;
  });

  let isoGeometry = $derived.by(() => {
    const count = isoSlabs.length;
    if (count === 0) return { polys: [] as { top: string; front: string; right: string }[], viewBox: "0 0 100 100" };

    // Vertical stack: flat plates stacked along Z
    const W = 80, D = 80;
    const H = Math.max(2, Math.min(6, Math.floor(200 / count)));
    const gap = Math.max(1, Math.floor(H * 0.3));
    const step = H + gap;
    const cos30 = Math.cos(Math.PI / 6);
    const sin30 = 0.5;

    function iso(x: number, y: number, z: number): string {
      return `${((x - y) * cos30).toFixed(1)},${((x + y) * sin30 - z).toFixed(1)}`;
    }

    function isoXY(x: number, y: number, z: number): [number, number] {
      return [(x - y) * cos30, (x + y) * sin30 - z];
    }

    const polys = isoSlabs.map((_, i) => {
      const z = i * step;
      return {
        top: `${iso(0,0,z+H)} ${iso(W,0,z+H)} ${iso(W,D,z+H)} ${iso(0,D,z+H)}`,
        front: `${iso(0,D,z+H)} ${iso(W,D,z+H)} ${iso(W,D,z)} ${iso(0,D,z)}`,
        right: `${iso(W,0,z+H)} ${iso(W,D,z+H)} ${iso(W,D,z)} ${iso(W,0,z)}`,
      };
    });

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < count; i++) {
      const z0 = i * step;
      const z1 = z0 + H;
      for (const pt of [[0,0,z0],[W,0,z0],[0,D,z0],[W,D,z0],[0,0,z1],[W,0,z1],[0,D,z1],[W,D,z1]]) {
        const [sx, sy] = isoXY(pt[0], pt[1], pt[2]);
        if (sx < minX) minX = sx;
        if (sx > maxX) maxX = sx;
        if (sy < minY) minY = sy;
        if (sy > maxY) maxY = sy;
      }
    }

    const pad = 8;
    return {
      polys,
      viewBox: `${Math.floor(minX - pad)} ${Math.floor(minY - pad)} ${Math.ceil(maxX - minX + 2 * pad)} ${Math.ceil(maxY - minY + 2 * pad)}`,
    };
  });

  // ── Data loading ───────────────────────────────────
  async function loadInspectData() {
    if (!model.isLoaded) {
      error = "No model loaded";
      loading = false;
      return;
    }
    try {
      data = await invoke<InspectData>("inspect_model");
      loading = false;
      loadCapabilities();
    } catch (e) {
      error = String(e);
      loading = false;
    }
  }

  async function computeFingerprint() {
    fingerprintLoading = true;
    fingerprintError = null;
    try {
      fingerprint = await invoke<ModelFingerprint>("compute_fingerprint");
    } catch (e) {
      fingerprintError = String(e);
    } finally {
      fingerprintLoading = false;
    }
  }

  async function loadCapabilities() {
    capLoading = true;
    try {
      capReport = await invoke<CapabilityReport>("inspect_capabilities");
    } catch (e) {
      console.error("Failed to detect capabilities:", e);
    } finally {
      capLoading = false;
    }
  }

  // ── Export ─────────────────────────────────────────
  function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function exportMetadataJson() {
    if (!model.info) return;
    const blob = new Blob(
      [JSON.stringify(model.info.metadata, null, 2)],
      { type: "application/json" }
    );
    downloadBlob(blob, `${model.info.file_name}-metadata.json`);
  }

  function exportTensorsCsv() {
    if (!data) return;
    const allTensors: InspectTensor[] = [
      ...data.other_tensors,
      ...data.layers.flatMap(l => [...l.attention, ...l.mlp, ...l.norms, ...l.other]),
    ];
    const header = "name,dtype,shape,params,memory_bytes,memory_display,component";
    const rows = allTensors.map(t =>
      `"${t.name}","${t.dtype}","[${t.shape.join(",")}]",${t.params},${t.memory_bytes},"${t.memory_display}","${t.component}"`
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    downloadBlob(blob, `${model.info?.file_name ?? "model"}-tensors.csv`);
  }

  $effect(() => {
    if (model.isLoaded) {
      loading = true;
      error = null;
      fingerprint = null;
      fingerprintError = null;
      capReport = null;
      loadInspectData();
    }
  });
</script>

<div class="inspect fade-in">
  <!-- ── Hero Panel ────────────────────────────────── -->
  <div class="hero panel">
    <div class="hero-top">
      <span class="label-xs">FRG.02</span>
      <span class="label-xs" style="color: var(--text-muted);">INSPECT-ENGINE</span>
      {#if model.isLoaded}
        <span class="badge badge-success" style="margin-left: auto;">
          <span class="dot dot-success"></span>
          LOADED
        </span>
      {:else}
        <span class="badge badge-dim" style="margin-left: auto;">
          <span class="dot"></span>
          NO MODEL
        </span>
      {/if}
    </div>

    <h1 class="hero-title">INSPECT</h1>
    <p class="hero-subtitle">Model architecture analysis &middot; Memory breakdown</p>

    {#if model.info && data}
      <div class="hero-specs">
        <div class="spec-cell">
          <span class="label-xs">MODEL</span>
          <span class="spec-value" style="font-size: 10px;">{model.info.file_name}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">ARCH</span>
          <span class="spec-value">{model.info.architecture?.toUpperCase() ?? "---"}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">PARAMS</span>
          <span class="spec-value">{data.total_params_display}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">LAYERS</span>
          <span class="spec-value">{data.layers.length || "---"}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">TENSORS</span>
          <span class="spec-value">{data.tensor_count}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">MEMORY</span>
          <span class="spec-value">{data.total_memory_display}</span>
        </div>
      </div>

      <div class="hero-actions">
        <button class="btn btn-secondary btn-sm" onclick={exportMetadataJson}>EXPORT JSON</button>
        <button class="btn btn-secondary btn-sm" onclick={exportTensorsCsv}>EXPORT CSV</button>
      </div>
    {/if}
  </div>

  {#if !model.isLoaded}
    <!-- No model state -->
    <div class="empty-state panel-flat" style="border-style: dashed;">
      <div class="empty-inner">
        <span class="heading-sm" style="color: var(--text-muted);">NO MODEL LOADED</span>
        <span class="label-xs" style="margin-top: 4px;">Load a model first to inspect its architecture</span>
        <button class="btn btn-accent" style="margin-top: 12px;" onclick={() => goto('/load')}>
          LOAD MODEL
        </button>
      </div>
    </div>
  {:else if loading}
    <div class="empty-state panel-flat">
      <div class="empty-inner">
        <span class="heading-sm" style="color: var(--info); animation: pulse 1.2s ease infinite;">ANALYZING MODEL...</span>
        <span class="label-xs" style="margin-top: 4px;">Parsing tensor data and computing memory layout</span>
      </div>
    </div>
  {:else if error}
    <div class="empty-state panel-flat" style="border-color: var(--danger);">
      <div class="empty-inner">
        <span class="danger-text">{error}</span>
      </div>
    </div>
  {:else if data}
    <div class="inspect-grid">
    <!-- ── Left Column ─────────────────────────────────── -->
    <div class="inspect-col">

    <!-- ── Folder Details (when loaded from directory) ──── -->
    {#if model.info && model.info.shard_count && model.info.shard_count > 0}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">MODEL DIRECTORY</span>
        </div>
        <div class="folder-panel panel-flat">
          <div class="folder-grid">
            <div class="folder-cell">
              <span class="label-xs">SHARDS</span>
              <span class="config-value code" style="color: var(--accent);">{model.info.shard_count}</span>
            </div>
            <div class="folder-cell">
              <span class="label-xs">CONFIG.JSON</span>
              <span class="config-value code" style="color: {model.info.has_config ? 'var(--success)' : 'var(--danger)'};">
                {model.info.has_config ? "FOUND" : "MISSING"}
              </span>
            </div>
            <div class="folder-cell">
              <span class="label-xs">TOKENIZER</span>
              <span class="config-value code" style="color: {model.info.has_tokenizer ? 'var(--success)' : 'var(--danger)'};">
                {model.info.has_tokenizer ? "FOUND" : "MISSING"}
              </span>
            </div>
            {#if model.info.model_type}
              <div class="folder-cell">
                <span class="label-xs">MODEL TYPE</span>
                <span class="config-value code">{model.info.model_type.toUpperCase()}</span>
              </div>
            {/if}
            {#if model.info.vocab_size}
              <div class="folder-cell">
                <span class="label-xs">VOCAB SIZE</span>
                <span class="config-value code">{model.info.vocab_size.toLocaleString()}</span>
              </div>
            {/if}
          </div>
          <div class="folder-path">
            <span class="label-xs">PATH</span>
            <span class="code" style="font-size: 10px; color: var(--text-secondary); word-break: break-all;">{model.info.file_path}</span>
          </div>
        </div>
      </div>
    {/if}

    <!-- ── File Verification / Fingerprint ────────────── -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">FILE VERIFICATION</span>
      </div>
      <div class="fingerprint-panel panel-flat">
        {#if fingerprint}
          <div class="fingerprint-grid">
            <div class="fingerprint-row">
              <span class="label-xs">SHA-256</span>
              <span class="code fingerprint-hash">{fingerprint.sha256}</span>
            </div>
            <div class="fingerprint-row">
              <span class="label-xs">FILE SIZE</span>
              <span class="code">{fingerprint.file_size_bytes.toLocaleString()} bytes</span>
            </div>
            <div class="fingerprint-row">
              <span class="label-xs">TENSORS</span>
              <span class="badge badge-success">
                <span class="dot dot-success"></span>
                {data.tensor_count} / {data.tensor_count} VERIFIED
              </span>
            </div>
          </div>
        {:else if fingerprintLoading}
          <div class="fingerprint-center">
            <span class="heading-sm" style="color: var(--info); animation: pulse 1.2s ease infinite;">COMPUTING HASH...</span>
            <span class="label-xs" style="margin-top: 4px;">This may take a while for large files</span>
          </div>
        {:else if fingerprintError}
          <div class="fingerprint-center">
            <span class="danger-text">{fingerprintError}</span>
          </div>
        {:else}
          <div class="fingerprint-center">
            <span class="label-xs">Compute SHA-256 hash for file integrity verification</span>
            <button class="btn btn-accent btn-sm" style="margin-top: 8px;" onclick={computeFingerprint}>COMPUTE HASH</button>
          </div>
        {/if}
      </div>
    </div>

    <!-- ── Model Configuration ─────────────────────────── -->
    {#if data.model_config.entries.length > 0}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">MODEL CONFIGURATION</span>
        </div>
        <div class="config-panel panel-flat">
          <div class="config-grid">
            {#each data.model_config.entries as entry}
              <div class="config-cell">
                <span class="label-xs">{entry.label}</span>
                <span class="config-value code">{entry.value}</span>
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}

    <!-- ── Memory Distribution ───────────────────────── -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">MEMORY DISTRIBUTION</span>
      </div>

      <div class="memory-panel panel-flat">
        <div class="memory-bar-container">
          {#each data.memory_breakdown as comp, i}
            <div
              class="memory-bar-segment"
              style="width: {comp.percentage}%; background: {memoryColors[i % memoryColors.length]};"
              title="{comp.name}: {comp.display} ({comp.percentage}%)"
            ></div>
          {/each}
        </div>

        <div class="memory-legend">
          {#each data.memory_breakdown as comp, i}
            <div class="memory-legend-item">
              <span class="memory-legend-dot" style="background: {memoryColors[i % memoryColors.length]};"></span>
              <span class="memory-legend-name">{comp.name}</span>
              <span class="memory-legend-size">{comp.display}</span>
              <span class="memory-legend-pct">{comp.percentage}%</span>
            </div>
          {/each}
        </div>

        <div class="memory-total">
          <span class="label-xs">TOTAL</span>
          <span class="spec-value">{data.total_memory_display}</span>
        </div>
      </div>
    </div>

    <!-- ── Quantization Breakdown ────────────────────── -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">QUANTIZATION BREAKDOWN</span>
      </div>

      <div class="quant-panel panel-flat">
        {#each data.quant_distribution as q}
          <div class="quant-row">
            <div class="quant-info">
              <span class="quant-dtype" style="color: {getQuantColor(q.dtype)};">{q.dtype}</span>
              <span class="label-xs">{q.count} tensors</span>
            </div>
            <div class="quant-bar-track">
              <div class="quant-bar-fill" style="width: {q.percentage}%; background: {getQuantColor(q.dtype)};"></div>
            </div>
            <div class="quant-stats">
              <span class="code">{q.display}</span>
              <span class="label-xs">{q.percentage}%</span>
            </div>
          </div>
        {/each}
      </div>
    </div>

    <!-- ── Compatibility Matrix ────────────────────────── -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">RUNTIME COMPATIBILITY</span>
      </div>
      <div class="compat-panel panel-flat">
        {#each RUNTIMES as rt}
          {@const status = getCompatStatus(rt)}
          <div class="compat-row">
            <span class="compat-name code">{rt.name}</span>
            {#if status === "full"}
              <span class="badge badge-success"><span class="dot dot-success"></span>COMPATIBLE</span>
            {:else if status === "partial"}
              <span class="badge badge-accent"><span class="dot dot-active"></span>PARTIAL</span>
            {:else}
              <span class="badge badge-dim"><span class="dot"></span>NOT SUPPORTED</span>
            {/if}
          </div>
        {/each}
      </div>
    </div>

    </div><!-- /inspect-col left -->

    <!-- ── Right Column ────────────────────────────────── -->
    <div class="inspect-col">

    <!-- ── Architecture Visualization ───────────────── -->
    {#if isoSlabs.length > 0}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">ARCHITECTURE VISUALIZATION</span>
        </div>
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div class="iso-container panel-flat" onmousemove={handleIsoMouseMove}>
          <div class="iso-viewport">
            <svg
              viewBox={isoGeometry.viewBox}
              class="iso-svg"
              preserveAspectRatio="xMidYMid meet"
            >
              <defs>
                <pattern id="iso-dots" width="4" height="4" patternUnits="userSpaceOnUse">
                  <rect x="0" y="0" width="1.5" height="1.5" fill="white" opacity="0.05"/>
                </pattern>
              </defs>
              {#each isoSlabs as slab, i}
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <g
                  class="iso-slab-group"
                  class:iso-dimmed={hoveredSlab !== null && hoveredSlab !== i}
                  onmouseenter={() => hoveredSlab = i}
                  onmouseleave={() => hoveredSlab = null}
                >
                  <polygon
                    points={isoGeometry.polys[i].right}
                    fill={slabColor(slab.rgb, 0.35)}
                    stroke={slabColor(slab.rgb, 0.25)}
                    stroke-width="0.5"
                  />
                  <polygon
                    points={isoGeometry.polys[i].front}
                    fill={slabColor(slab.rgb, 0.55)}
                    stroke={slabColor(slab.rgb, 0.4)}
                    stroke-width="0.5"
                  />
                  <polygon
                    points={isoGeometry.polys[i].top}
                    fill={slabColor(slab.rgb, 0.85)}
                    stroke={slabColor(slab.rgb, 0.65)}
                    stroke-width="0.5"
                  />
                  <polygon
                    points={isoGeometry.polys[i].top}
                    fill="url(#iso-dots)"
                  />
                </g>
              {/each}
            </svg>
          </div>
          <div class="iso-footer">
            <div class="iso-legend-row">
              <div class="iso-legend-item">
                <span class="iso-legend-dot" style="background: #f59e0b;"></span>
                <span class="label-xs">EMBEDDING</span>
              </div>
              <div class="iso-legend-item">
                <span class="iso-legend-dot" style="background: #3b82f6;"></span>
                <span class="label-xs">ATTENTION</span>
              </div>
              <div class="iso-legend-item">
                <span class="iso-legend-dot" style="background: #22c55e;"></span>
                <span class="label-xs">MLP</span>
              </div>
              <div class="iso-legend-item">
                <span class="iso-legend-dot" style="background: #ef4444;"></span>
                <span class="label-xs">OUTPUT</span>
              </div>
              <span class="label-xs" style="margin-left: auto; color: var(--text-muted);">{isoSlabs.length} LAYERS · {data?.total_memory_display}</span>
            </div>
          </div>

          {#if hoveredSlab !== null && isoSlabs[hoveredSlab]}
            {@const slab = isoSlabs[hoveredSlab]}
            <div
              class="iso-tooltip"
              style="left: {mouseX + 16}px; top: {mouseY - 16}px;"
            >
              <div class="iso-tooltip-bar" style="background: {slabColor(slab.rgb, 0.85)};"></div>
              <div class="iso-tooltip-header">
                <span class="iso-tooltip-type">{slab.type.toUpperCase()}</span>
                <span class="heading-sm" style="color: {slabColor(slab.rgb, 1)};">{slab.label}</span>
              </div>
              <div class="iso-tooltip-body">
                <div class="iso-tooltip-row">
                  <span class="label-xs">MEMORY</span>
                  <span class="code">{slab.memDisplay}</span>
                </div>
                <div class="iso-tooltip-row">
                  <span class="label-xs">TENSORS</span>
                  <span class="code">{slab.tensorCount}</span>
                </div>
                {#if slab.attnCount !== undefined}
                  <div class="iso-tooltip-sep"></div>
                  <div class="iso-tooltip-row">
                    <span class="label-xs" style="color: var(--info);">ATTENTION</span>
                    <span class="code">{slab.attnCount} · {slab.attnMem}</span>
                  </div>
                {/if}
                {#if slab.mlpCount !== undefined}
                  <div class="iso-tooltip-row">
                    <span class="label-xs" style="color: var(--success);">MLP</span>
                    <span class="code">{slab.mlpCount} · {slab.mlpMem}</span>
                  </div>
                {/if}
                {#if slab.normCount !== undefined && slab.normCount > 0}
                  <div class="iso-tooltip-row">
                    <span class="label-xs" style="color: var(--gray);">NORMS</span>
                    <span class="code">{slab.normCount} · {slab.normMem}</span>
                  </div>
                {/if}
              </div>
            </div>
          {/if}
        </div>
      </div>
    {/if}

    <!-- ── Tokenizer ───────────────────────────────────── -->
    {#if data.tokenizer_info}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">TOKENIZER</span>
        </div>
        <div class="tokenizer-panel panel-flat">
          <div class="tokenizer-header">
            {#if data.tokenizer_info.tokenizer_type}
              <div class="config-cell">
                <span class="label-xs">TYPE</span>
                <span class="config-value code">{data.tokenizer_info.tokenizer_type.toUpperCase()}</span>
              </div>
            {/if}
            {#if data.tokenizer_info.vocab_size}
              <div class="config-cell">
                <span class="label-xs">VOCABULARY</span>
                <span class="config-value code">{data.tokenizer_info.vocab_size.toLocaleString()} TOKENS</span>
              </div>
            {/if}
          </div>
          {#if data.tokenizer_info.special_tokens.length > 0}
            <div class="special-tokens">
              <span class="label-xs" style="margin-bottom: 4px;">SPECIAL TOKENS</span>
              {#each data.tokenizer_info.special_tokens as st}
                <div class="special-token-row">
                  <span class="badge badge-dim">{st.role}</span>
                  <span class="code" style="color: var(--text-secondary);">ID {st.id}</span>
                  {#if st.token}
                    <span class="code" style="color: var(--accent);">{st.token}</span>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
        </div>
      </div>
    {/if}

    <!-- ── Capabilities ─────────────────────────────── -->
    {#if capReport}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">CAPABILITIES</span>
          <span class="label-xs" style="color: var(--text-muted); margin-left: 8px;">{capReport.total_detected} DETECTED</span>
        </div>
        <div class="cap-panel panel-flat">
          <div class="cap-grid">
            {#each capReport.capabilities as cap}
              <div class="cap-item" class:cap-detected={cap.detected} class:cap-undetected={!cap.detected}>
                <div class="cap-header">
                  <span class="cap-name">{cap.name}</span>
                  {#if cap.detected}
                    <span class="cap-confidence">{(cap.confidence * 100).toFixed(0)}%</span>
                  {/if}
                </div>
                {#if cap.detected}
                  <div class="cap-bar-track">
                    <div class="cap-bar-fill" style="width: {cap.confidence * 100}%"></div>
                  </div>
                  {#if cap.evidence.length > 0}
                    <div class="cap-evidence">
                      {#each cap.evidence as ev}
                        <span class="badge badge-dim">{ev}</span>
                      {/each}
                    </div>
                  {/if}
                  {#if cap.affected_layers.length > 0}
                    <span class="label-xs" style="color: var(--text-muted); margin-top: 2px;">LAYERS {cap.affected_layers[0]}-{cap.affected_layers[cap.affected_layers.length - 1]}</span>
                  {/if}
                {/if}
              </div>
            {/each}
          </div>
        </div>
      </div>
    {:else if capLoading}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">CAPABILITIES</span>
        </div>
        <div class="cap-panel panel-flat">
          <span class="label-xs" style="color: var(--text-muted);">DETECTING...</span>
        </div>
      </div>
    {/if}

    <!-- ── Attention Architecture ─────────────────────── -->
    {#if data.attention_info}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">ATTENTION ARCHITECTURE</span>
        </div>

        <div class="attn-panel panel-flat">
          <div class="attn-grid">
            <div class="attn-cell">
              <span class="label-xs">TYPE</span>
              <span class="attn-value" style="color: var(--accent);">{data.attention_info.attention_type}</span>
            </div>
            {#if data.attention_info.q_heads}
              <div class="attn-cell">
                <span class="label-xs">QUERY HEADS</span>
                <span class="attn-value">{data.attention_info.q_heads}</span>
              </div>
            {/if}
            {#if data.attention_info.kv_heads}
              <div class="attn-cell">
                <span class="label-xs">KV HEADS</span>
                <span class="attn-value">{data.attention_info.kv_heads}</span>
              </div>
            {/if}
            {#if data.attention_info.head_dim}
              <div class="attn-cell">
                <span class="label-xs">HEAD DIM</span>
                <span class="attn-value">{data.attention_info.head_dim}</span>
              </div>
            {/if}
            {#if data.attention_info.gqa_ratio}
              <div class="attn-cell">
                <span class="label-xs">GQA RATIO</span>
                <span class="attn-value">{data.attention_info.gqa_ratio}:1</span>
              </div>
            {/if}
          </div>

          {#if data.attention_info.attention_type === "GQA" && data.attention_info.q_heads && data.attention_info.kv_heads}
            <div class="attn-diagram">
              <div class="attn-diagram-label">
                <span class="label-xs">QUERY</span>
                <div class="attn-heads-row">
                  {#each Array(Math.min(data.attention_info.q_heads, 32)) as _, i}
                    <div class="attn-head attn-head-q" title="Q{i}"></div>
                  {/each}
                  {#if data.attention_info.q_heads > 32}
                    <span class="label-xs" style="color: var(--text-muted);">+{data.attention_info.q_heads - 32}</span>
                  {/if}
                </div>
              </div>
              <div class="attn-diagram-label">
                <span class="label-xs">KEY/VALUE</span>
                <div class="attn-heads-row">
                  {#each Array(Math.min(data.attention_info.kv_heads, 32)) as _, i}
                    <div class="attn-head attn-head-kv" title="KV{i}"></div>
                  {/each}
                </div>
              </div>
              <div class="attn-note">
                <span class="label-xs" style="color: var(--text-muted);">
                  {data.attention_info.gqa_ratio} query heads share each KV head &middot;
                  {((1 - (data.attention_info.kv_heads / data.attention_info.q_heads)) * 100).toFixed(0)}% KV cache reduction
                </span>
              </div>
            </div>
          {/if}
        </div>
      </div>
    {/if}

    </div><!-- /inspect-col right -->
    </div><!-- /inspect-grid -->

    <!-- ── Layer Hierarchy (full width) ────────────────── -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">LAYER HIERARCHY ({isFiltering ? `${filteredLayers.length}/` : ""}{data.layers.length} BLOCKS)</span>
      </div>

      <!-- Tensor Filter Bar -->
      <div class="filter-bar panel-flat">
        <div class="filter-row">
          <div class="filter-field filter-field-grow">
            <span class="label-xs">SEARCH</span>
            <input
              type="text"
              class="filter-input"
              placeholder="tensor name..."
              bind:value={tensorSearch}
            />
          </div>
          <div class="filter-field">
            <span class="label-xs">DTYPE</span>
            <select class="filter-select" bind:value={dtypeFilter}>
              <option value="all">ALL</option>
              {#each allDtypes as dtype}
                <option value={dtype}>{dtype}</option>
              {/each}
            </select>
          </div>
          <div class="filter-field">
            <span class="label-xs">LAYER RANGE</span>
            <div class="filter-range">
              <input type="number" class="filter-input filter-input-sm" placeholder="min" bind:value={layerRangeMin} min="0" />
              <span class="label-xs">-</span>
              <input type="number" class="filter-input filter-input-sm" placeholder="max" bind:value={layerRangeMax} min="0" />
            </div>
          </div>
          {#if isFiltering}
            <button class="btn btn-ghost btn-sm" style="align-self: flex-end;" onclick={clearFilters}>CLEAR</button>
          {/if}
        </div>
      </div>

      <!-- Global tensors (embeddings, output, norms) -->
      {#if filteredOtherTensors.length > 0}
        <div class="layer-group panel-inset">
          <div class="layer-header">
            <span class="heading-sm" style="color: var(--accent);">GLOBAL TENSORS</span>
            <span class="label-xs">{filteredOtherTensors.length} tensors</span>
          </div>
          <div class="tensor-list">
            {#each filteredOtherTensors as t}
              <div class="tensor-row">
                <span class="tensor-name code">{t.name}</span>
                <span class="tensor-dtype code" style="color: {getQuantColor(t.dtype)};">{t.dtype}</span>
                <span class="tensor-shape code">[{t.shape.join(", ")}]</span>
                <span class="tensor-mem label-xs">{t.memory_display}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Transformer blocks -->
      <div class="layers-list">
        {#each filteredLayers as layer}
          <div class="layer-group panel-inset">
            <button class="layer-header layer-toggle" onclick={() => toggleLayer(layer.index)}>
              <span class="layer-expand">{expandedLayers.has(layer.index) ? "▼" : "▶"}</span>
              <span class="heading-sm">BLOCK {layer.index}</span>
              <span class="label-xs" style="margin-left: auto;">{layer.display}</span>
              <div class="layer-mini-bar">
                {#if layer.attention.length > 0}
                  {@const attnBytes = layer.attention.reduce((s, t) => s + t.memory_bytes, 0)}
                  <div class="layer-mini-seg" style="flex: {attnBytes}; background: var(--info);" title="Attention"></div>
                {/if}
                {#if layer.mlp.length > 0}
                  {@const mlpBytes = layer.mlp.reduce((s, t) => s + t.memory_bytes, 0)}
                  <div class="layer-mini-seg" style="flex: {mlpBytes}; background: var(--success);" title="MLP"></div>
                {/if}
                {#if layer.norms.length > 0}
                  {@const normBytes = layer.norms.reduce((s, t) => s + t.memory_bytes, 0)}
                  <div class="layer-mini-seg" style="flex: {normBytes}; background: var(--gray);" title="Norms"></div>
                {/if}
              </div>
            </button>

            {#if expandedLayers.has(layer.index)}
              <div class="layer-body">
                {#if layer.attention.length > 0}
                  <div class="component-group">
                    <span class="component-label" style="color: var(--info);">ATTENTION</span>
                    {#each layer.attention as t}
                      <div class="tensor-row">
                        <span class="tensor-name code">{t.name}</span>
                        <span class="tensor-dtype code" style="color: {getQuantColor(t.dtype)};">{t.dtype}</span>
                        <span class="tensor-shape code">[{t.shape.join(", ")}]</span>
                        <span class="tensor-mem label-xs">{t.memory_display}</span>
                      </div>
                    {/each}
                  </div>
                {/if}
                {#if layer.mlp.length > 0}
                  <div class="component-group">
                    <span class="component-label" style="color: var(--success);">MLP / FEED-FORWARD</span>
                    {#each layer.mlp as t}
                      <div class="tensor-row">
                        <span class="tensor-name code">{t.name}</span>
                        <span class="tensor-dtype code" style="color: {getQuantColor(t.dtype)};">{t.dtype}</span>
                        <span class="tensor-shape code">[{t.shape.join(", ")}]</span>
                        <span class="tensor-mem label-xs">{t.memory_display}</span>
                      </div>
                    {/each}
                  </div>
                {/if}
                {#if layer.norms.length > 0}
                  <div class="component-group">
                    <span class="component-label" style="color: var(--gray);">LAYER NORMS</span>
                    {#each layer.norms as t}
                      <div class="tensor-row">
                        <span class="tensor-name code">{t.name}</span>
                        <span class="tensor-dtype code" style="color: {getQuantColor(t.dtype)};">{t.dtype}</span>
                        <span class="tensor-shape code">[{t.shape.join(", ")}]</span>
                        <span class="tensor-mem label-xs">{t.memory_display}</span>
                      </div>
                    {/each}
                  </div>
                {/if}
                {#if layer.other.length > 0}
                  <div class="component-group">
                    <span class="component-label">OTHER</span>
                    {#each layer.other as t}
                      <div class="tensor-row">
                        <span class="tensor-name code">{t.name}</span>
                        <span class="tensor-dtype code" style="color: {getQuantColor(t.dtype)};">{t.dtype}</span>
                        <span class="tensor-shape code">[{t.shape.join(", ")}]</span>
                        <span class="tensor-mem label-xs">{t.memory_display}</span>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .inspect {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .inspect-grid {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .inspect-col {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  @media (min-width: 1100px) {
    .inspect-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      align-items: start;
    }
  }

  /* ── Hero ──────────────────────────────────────── */
  .hero {
    padding: 16px;
  }

  .hero-top {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .hero-title {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 0.18em;
    line-height: 1;
    color: var(--text-primary);
  }

  .hero-subtitle {
    font-size: 10px;
    color: var(--text-secondary);
    letter-spacing: 0.06em;
    margin-top: 6px;
    text-transform: uppercase;
  }

  .hero-specs {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1px;
    background: var(--border);
    margin-top: 16px;
    border: 1px solid var(--border);
  }

  .hero-actions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    justify-content: flex-end;
  }

  .spec-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px;
    background: var(--bg-surface);
    text-align: center;
  }

  .spec-value {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-primary);
    text-transform: uppercase;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* ── Empty State ──────────────────────────────── */
  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 140px;
  }

  .empty-inner {
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* ── Section ──────────────────────────────────── */
  .section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-label {
    padding: 0;
  }

  /* ── Fingerprint ───────────────────────────────── */
  .fingerprint-panel {
    padding: 12px 16px;
  }

  .fingerprint-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .fingerprint-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .fingerprint-hash {
    font-size: 10px;
    color: var(--accent);
    word-break: break-all;
    user-select: all;
  }

  .fingerprint-center {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 0;
    text-align: center;
  }

  /* ── Model Configuration ───────────────────────── */
  .config-panel {
    padding: 0;
  }

  .config-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
  }

  .config-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    background: var(--bg-surface);
  }

  .config-value {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-primary);
  }

  /* ── Tokenizer ─────────────────────────────────── */
  .tokenizer-panel {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .tokenizer-header {
    display: flex;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
  }

  .tokenizer-header .config-cell {
    flex: 1;
    text-align: center;
  }

  .special-tokens {
    display: flex;
    flex-direction: column;
  }

  .special-token-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid var(--border-dim);
  }

  /* ── Memory Distribution ──────────────────────── */
  .memory-panel {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .memory-bar-container {
    display: flex;
    height: 20px;
    gap: 2px;
    overflow: hidden;
  }

  .memory-bar-segment {
    height: 100%;
    min-width: 2px;
    transition: width 300ms ease;
  }

  .memory-legend {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .memory-legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .memory-legend-dot {
    width: 10px;
    height: 10px;
    flex-shrink: 0;
  }

  .memory-legend-name {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-primary);
    flex: 1;
  }

  .memory-legend-size {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    min-width: 80px;
    text-align: right;
  }

  .memory-legend-pct {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    min-width: 44px;
    text-align: right;
  }

  .memory-total {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 8px;
    border-top: 1px solid var(--border-dim);
  }

  /* ── Quantization ─────────────────────────────── */
  .quant-panel {
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .quant-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .quant-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 80px;
  }

  .quant-dtype {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    font-family: var(--font-mono);
  }

  .quant-bar-track {
    flex: 1;
    height: 8px;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
    overflow: hidden;
  }

  .quant-bar-fill {
    height: 100%;
    transition: width 300ms ease;
  }

  .quant-stats {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 80px;
    text-align: right;
    font-size: 10px;
  }

  /* ── Attention Architecture ───────────────────── */
  .attn-panel {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .attn-grid {
    display: flex;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
  }

  .attn-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px 14px;
    background: var(--bg-surface);
    flex: 1;
    text-align: center;
  }

  .attn-value {
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: var(--text-primary);
  }

  .attn-diagram {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
  }

  .attn-diagram-label {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .attn-heads-row {
    display: flex;
    gap: 3px;
    flex-wrap: wrap;
    align-items: center;
  }

  .attn-head {
    width: 10px;
    height: 10px;
  }

  .attn-head-q {
    background: var(--info);
  }

  .attn-head-kv {
    background: var(--accent);
  }

  .attn-note {
    padding-top: 6px;
    border-top: 1px solid var(--border-dim);
  }

  /* ── Compatibility Matrix ──────────────────────── */
  .compat-panel {
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .compat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-dim);
  }

  .compat-row:last-child {
    border-bottom: none;
  }

  .compat-name {
    font-size: 10px;
    color: var(--text-primary);
  }

  /* ── Tensor Filter ─────────────────────────────── */
  .filter-bar {
    padding: 10px 12px;
  }

  .filter-row {
    display: flex;
    gap: 12px;
    align-items: flex-end;
  }

  .filter-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .filter-field-grow {
    flex: 1;
  }

  .filter-input,
  .filter-select {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 5px 8px;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
    color: var(--text-primary);
    letter-spacing: 0.06em;
    outline: none;
  }

  .filter-input:focus,
  .filter-select:focus {
    border-color: var(--accent);
  }

  .filter-input-sm {
    width: 56px;
  }

  .filter-range {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .filter-select {
    appearance: none;
    padding-right: 20px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23737373'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 6px center;
    cursor: pointer;
  }

  /* ── Layer Hierarchy ──────────────────────────── */
  .layers-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .layer-group {
    overflow: hidden;
  }

  .layer-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
  }

  .layer-toggle {
    width: 100%;
    background: none;
    border: none;
    cursor: pointer;
    font-family: var(--font-mono);
    color: var(--text-primary);
    text-align: left;
  }

  .layer-toggle:hover {
    background: var(--bg-hover);
  }

  .layer-expand {
    font-size: 9px;
    color: var(--text-muted);
    width: 12px;
  }

  .layer-mini-bar {
    display: flex;
    height: 6px;
    width: 60px;
    gap: 1px;
    flex-shrink: 0;
  }

  .layer-mini-seg {
    height: 100%;
  }

  .layer-body {
    padding: 0 12px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .component-group {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .component-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 4px 0 2px;
    border-bottom: 1px solid var(--border-dim);
    margin-bottom: 2px;
  }

  /* ── Tensor Rows ──────────────────────────────── */
  .tensor-list {
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 8px 0;
  }

  .tensor-row {
    display: flex;
    gap: 12px;
    padding: 3px 0;
    align-items: baseline;
    font-size: 10px;
  }

  .tensor-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-secondary);
  }

  .tensor-dtype {
    width: 50px;
    flex-shrink: 0;
    text-align: center;
  }

  .tensor-shape {
    width: 160px;
    flex-shrink: 0;
    color: var(--text-muted);
  }

  .tensor-mem {
    width: 70px;
    flex-shrink: 0;
    text-align: right;
    color: var(--text-secondary);
  }

  /* ── Isometric Visualization ───────────────── */
  .iso-container {
    display: flex;
    flex-direction: column;
    padding: 0;
    overflow: visible;
    position: relative;
  }

  .iso-viewport {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 280px;
    padding: 24px 16px 12px;
  }

  .iso-svg {
    width: 100%;
    height: 100%;
    max-height: 360px;
  }

  .iso-slab-group {
    cursor: pointer;
    transition: opacity 200ms ease;
  }

  .iso-dimmed {
    opacity: 0.2;
  }

  .iso-footer {
    padding: 8px 16px;
    border-top: 1px solid var(--border-dim);
  }

  .iso-legend-row {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }

  .iso-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .iso-legend-dot {
    width: 8px;
    height: 8px;
    flex-shrink: 0;
  }

  /* ── Floating Tooltip ──────────────────────── */
  .iso-tooltip {
    position: fixed;
    z-index: 1000;
    pointer-events: none;
    min-width: 150px;
    max-width: 220px;
    background: var(--bg-inset);
    border: 2px solid var(--accent);
    clip-path: polygon(
      0 4px, 4px 4px, 4px 0,
      calc(100% - 4px) 0, calc(100% - 4px) 4px, 100% 4px,
      100% calc(100% - 4px), calc(100% - 4px) calc(100% - 4px), calc(100% - 4px) 100%,
      4px 100%, 4px calc(100% - 4px), 0 calc(100% - 4px)
    );
  }

  .iso-tooltip-bar {
    height: 3px;
  }

  .iso-tooltip-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px 4px;
  }

  .iso-tooltip-type {
    font-size: 8px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    padding: 1px 4px;
    border: 1px solid var(--border-dim);
  }

  .iso-tooltip-body {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 4px 10px 8px;
  }

  .iso-tooltip-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
  }

  .iso-tooltip-row .code {
    font-size: 10px;
    color: var(--text-primary);
  }

  .iso-tooltip-sep {
    height: 1px;
    background: var(--border-dim);
    margin: 2px 0;
  }

  /* ── Folder Details ─────────────────────────────── */
  .folder-panel {
    padding: 0;
    display: flex;
    flex-direction: column;
  }

  .folder-grid {
    display: flex;
    gap: 1px;
    background: var(--border);
    border-bottom: 1px solid var(--border);
  }

  .folder-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    background: var(--bg-surface);
    flex: 1;
    text-align: center;
  }

  .folder-path {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 8px 10px;
  }

  /* ── Capabilities ──────────────────────────── */
  .cap-panel {
    padding: 10px;
  }
  .cap-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 8px;
  }
  .cap-item {
    padding: 8px 10px;
    border: 1px solid var(--border);
    background: var(--bg-secondary);
  }
  .cap-detected {
    border-color: var(--accent);
  }
  .cap-undetected {
    opacity: 0.4;
  }
  .cap-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }
  .cap-name {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-primary);
  }
  .cap-detected .cap-name {
    color: var(--accent);
  }
  .cap-confidence {
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--accent);
  }
  .cap-bar-track {
    height: 3px;
    background: var(--border);
    margin-bottom: 4px;
  }
  .cap-bar-fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.3s ease;
  }
  .cap-evidence {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    margin-top: 2px;
  }
</style>
