<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { save } from "@tauri-apps/plugin-dialog";
  import { goto } from "$app/navigation";
  import { model } from "$lib/model.svelte";

  // ── Interfaces ───────────────────────────────────
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

  interface InspectData {
    memory_breakdown: MemoryComponent[];
    total_memory_bytes: number;
    total_memory_display: string;
    quant_distribution: QuantEntry[];
    tensor_count: number;
    total_params: number;
    total_params_display: string;
    layers: any[];
  }

  interface QuantizeResult {
    success: boolean;
    output_path: string;
    output_size: number;
    output_size_display: string;
  }

  interface QuantLevel {
    id: string;
    name: string;
    label: string;
    bpw: number;
    quality: number;
    speedGain: number;
    targetType: string;
    description: string;
  }

  // ── Constants ────────────────────────────────────
  const QUANT_LEVELS: QuantLevel[] = [
    { id: "extreme",  name: "EXTREME",  label: "Q2_K",   bpw: 2.56, quality: 58, speedGain: 3.2, targetType: "Q2_K",   description: "Maximum compression, significant quality loss" },
    { id: "tiny",     name: "TINY",     label: "Q3_K_S", bpw: 3.44, quality: 68, speedGain: 2.6, targetType: "Q3_K_S", description: "Very small, noticeable quality reduction" },
    { id: "small",    name: "SMALL",    label: "Q3_K_M", bpw: 3.69, quality: 75, speedGain: 2.4, targetType: "Q3_K_M", description: "Small with better quality retention" },
    { id: "compact",  name: "COMPACT",  label: "Q4_K_M", bpw: 4.85, quality: 82, speedGain: 2.0, targetType: "Q4_K_M", description: "Good balance of size and quality" },
    { id: "balanced", name: "BALANCED", label: "Q5_K_M", bpw: 5.69, quality: 90, speedGain: 1.6, targetType: "Q5_K_M", description: "Near-original quality, moderate size" },
    { id: "high",     name: "HIGH",     label: "Q6_K",   bpw: 6.56, quality: 95, speedGain: 1.3, targetType: "Q6_K",   description: "Minimal quality loss" },
    { id: "ultra",    name: "ULTRA",    label: "Q8_0",   bpw: 8.5,  quality: 99, speedGain: 1.1, targetType: "Q8_0",   description: "Near-lossless quantization" },
  ];

  const PRESETS = [
    { name: "MOBILE",   levelIndex: 1, useCase: "Edge devices, phones, low-RAM systems", icon: "M" },
    { name: "BALANCED", levelIndex: 3, useCase: "General purpose, best size/quality ratio", icon: "B" },
    { name: "QUALITY",  levelIndex: 5, useCase: "Production servers, quality-critical tasks", icon: "Q" },
  ];

  // ── State ────────────────────────────────────────
  let data = $state<InspectData | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);

  let selectedLevelIndex = $state(3); // default: COMPACT (Q4_K_M)

  let quantizing = $state(false);
  let quantizeError = $state<string | null>(null);
  let quantizeResult = $state<QuantizeResult | null>(null);

  // ── Helpers ──────────────────────────────────────
  function formatMemory(bytes: number): string {
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + " GB";
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + " MB";
    if (bytes >= 1024) return (bytes / 1024).toFixed(0) + " KB";
    return bytes + " B";
  }

  // ── Derived Estimation ───────────────────────────
  let selectedLevel = $derived(QUANT_LEVELS[selectedLevelIndex]);

  // Warn if model is already quantized at or below the target
  let isAlreadyQuantized = $derived(model.info?.quantization != null && model.info?.quantization !== "F16" && model.info?.quantization !== "F32");
  let requantWarning = $derived.by(() => {
    if (!isAlreadyQuantized || !model.info?.quantization) return null;
    const current = model.info.quantization.toUpperCase();
    // Find BPW of current quant
    const currentLevel = QUANT_LEVELS.find(l => l.targetType === current);
    if (currentLevel && selectedLevel.bpw >= currentLevel.bpw) {
      return `Model is already ${current} (${currentLevel.bpw} BPW). Requantizing to ${selectedLevel.targetType} (${selectedLevel.bpw} BPW) may fail or produce no benefit.`;
    }
    return `Model is already quantized (${current}). Requantizing may reduce quality further. For best results, quantize from an F16 or F32 source.`;
  });

  let currentBpw = $derived.by(() => {
    if (!data || data.total_params === 0) return 16;
    return (data.total_memory_bytes * 8) / data.total_params;
  });

  let estimatedSizeBytes = $derived.by(() => {
    if (!data || data.total_params === 0) return 0;

    const normBytes = data.memory_breakdown
      .filter(c => c.name === "Layer Norms" || c.name === "Other")
      .reduce((sum, c) => sum + c.bytes, 0);

    const embedBytes = data.memory_breakdown
      .filter(c => c.name === "Token Embeddings" || c.name === "Output Head")
      .reduce((sum, c) => sum + c.bytes, 0);

    const bulkBytes = data.memory_breakdown
      .filter(c => c.name === "Attention Weights" || c.name === "MLP / Feed-Forward")
      .reduce((sum, c) => sum + c.bytes, 0);

    const bulkParams = currentBpw > 0 ? (bulkBytes * 8) / currentBpw : 0;
    const newBulkBytes = (bulkParams * selectedLevel.bpw) / 8;

    const metadataOverhead = 512 * 1024;
    return Math.ceil(newBulkBytes + normBytes + embedBytes + metadataOverhead);
  });

  let estimatedSizeDisplay = $derived(formatMemory(estimatedSizeBytes));

  let sizeReduction = $derived.by(() => {
    if (!data || data.total_memory_bytes === 0) return 0;
    const r = ((data.total_memory_bytes - estimatedSizeBytes) / data.total_memory_bytes) * 100;
    return Math.max(0, Math.round(r * 10) / 10);
  });

  let estimatedBreakdown = $derived.by(() => {
    if (!data) return [];
    return data.memory_breakdown.map(comp => {
      const isBulk = comp.name === "Attention Weights" || comp.name === "MLP / Feed-Forward";
      if (isBulk && currentBpw > 0) {
        const params = (comp.bytes * 8) / currentBpw;
        const newBytes = Math.ceil((params * selectedLevel.bpw) / 8);
        return {
          name: comp.name,
          currentBytes: comp.bytes,
          currentDisplay: comp.display,
          estimatedBytes: newBytes,
          estimatedDisplay: formatMemory(newBytes),
        };
      }
      return {
        name: comp.name,
        currentBytes: comp.bytes,
        currentDisplay: comp.display,
        estimatedBytes: comp.bytes,
        estimatedDisplay: comp.display,
      };
    });
  });

  // ── Data Loading ─────────────────────────────────
  async function loadInspectData() {
    try {
      data = await invoke<InspectData>("inspect_model");
      loading = false;
    } catch (e) {
      error = String(e);
      loading = false;
    }
  }

  $effect(() => {
    if (model.isLoaded) {
      loading = true;
      error = null;
      quantizeResult = null;
      quantizeError = null;
      loadInspectData();
    }
  });

  // ── Quantize Action ──────────────────────────────
  async function handleQuantize() {
    if (!model.info) return;

    const outputPath = await save({
      defaultPath: model.info.file_name.replace(/\.gguf$/i, `-${selectedLevel.targetType}.gguf`),
      filters: [{ name: "GGUF Model", extensions: ["gguf"] }],
    });
    if (!outputPath) return;

    quantizing = true;
    quantizeError = null;
    quantizeResult = null;

    try {
      quantizeResult = await invoke<QuantizeResult>("quantize_model", {
        targetType: selectedLevel.targetType,
        outputPath,
      });
    } catch (e) {
      quantizeError = String(e);
    } finally {
      quantizing = false;
    }
  }
</script>

<div class="optimize fade-in">
  <!-- ── Hero Panel ────────────────────────────────── -->
  <div class="hero panel">
    <div class="hero-top">
      <span class="label-xs">FRG.03</span>
      <span class="label-xs" style="color: var(--text-muted);">OPTIMIZE-ENGINE</span>
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

    <h1 class="hero-title">OPTIMIZE</h1>
    <p class="hero-subtitle">Quantization &amp; compression engine</p>

    {#if model.info && data}
      <div class="hero-specs">
        <div class="spec-cell spec-cell-wide">
          <span class="label-xs">MODEL</span>
          <span class="spec-value" style="font-size: 10px;">{model.info.file_name}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">FORMAT</span>
          <span class="spec-value">{model.formatDisplay.toUpperCase()}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">SIZE</span>
          <span class="spec-value">{data.total_memory_display}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">PARAMS</span>
          <span class="spec-value">{data.total_params_display}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">QUANT</span>
          <span class="spec-value">{model.info.quantization?.toUpperCase() ?? "---"}</span>
        </div>
        <div class="spec-cell">
          <span class="label-xs">LAYERS</span>
          <span class="spec-value">{model.info.layer_count ?? "---"}</span>
        </div>
      </div>
    {/if}
  </div>

  {#if !model.isLoaded}
    <div class="empty-state panel-flat" style="border-style: dashed;">
      <div class="empty-inner">
        <span class="heading-sm" style="color: var(--text-muted);">NO MODEL LOADED</span>
        <span class="label-xs" style="margin-top: 4px;">Load a model first to configure optimization</span>
        <button class="btn btn-accent" style="margin-top: 12px;" onclick={() => goto('/load')}>
          LOAD MODEL
        </button>
      </div>
    </div>
  {:else if loading}
    <div class="empty-state panel-flat">
      <div class="empty-inner">
        <span class="heading-sm" style="color: var(--info); animation: pulse 1.2s ease infinite;">ANALYZING MODEL...</span>
        <span class="label-xs" style="margin-top: 4px;">Computing parameter distribution and memory layout</span>
      </div>
    </div>
  {:else if error}
    <div class="empty-state panel-flat" style="border-color: var(--danger);">
      <div class="empty-inner">
        <span class="danger-text">{error}</span>
      </div>
    </div>
  {:else if model.info && model.info.format !== "gguf"}
    <div class="empty-state panel-flat" style="border-color: var(--accent);">
      <div class="empty-inner">
        <span class="heading-sm" style="color: var(--accent);">GGUF MODELS ONLY</span>
        <span class="label-xs" style="margin-top: 4px; max-width: 400px; text-align: center; line-height: 1.6;">
          Quantization via llama-quantize is only available for GGUF format models.
          SafeTensors models must be converted to GGUF first.
        </span>
        <button class="btn btn-secondary" style="margin-top: 12px;" onclick={() => goto('/load')}>
          LOAD GGUF MODEL
        </button>
      </div>
    </div>
  {:else if data}

    {#if requantWarning}
      <div class="requant-warning panel-flat">
        <span class="dot dot-active"></span>
        <span class="requant-text">{requantWarning}</span>
      </div>
    {/if}

    <!-- ── Target Configuration ──────────────────────── -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">TARGET CONFIGURATION</span>
      </div>

      <div class="quant-selector panel-flat">
        <!-- Level Strip -->
        <div class="quant-strip">
          {#each QUANT_LEVELS as level, i}
            <button
              class="quant-btn"
              class:quant-btn-active={i === selectedLevelIndex}
              onclick={() => selectedLevelIndex = i}
            >
              <span class="quant-btn-type">{level.label}</span>
              <span class="quant-btn-bpw">{level.bpw} BPW</span>
            </button>
          {/each}
        </div>

        <!-- Selected Level Detail -->
        <div class="quant-detail">
          <div class="quant-detail-header">
            <span class="quant-detail-name">{selectedLevel.name}</span>
            <span class="quant-detail-type">{selectedLevel.targetType}</span>
          </div>
          <div class="quant-detail-grid">
            <div class="quant-detail-cell">
              <span class="label-xs">BITS/WEIGHT</span>
              <span class="spec-value">{selectedLevel.bpw}</span>
            </div>
            <div class="quant-detail-cell">
              <span class="label-xs">QUALITY</span>
              <span class="spec-value" style="color: {selectedLevel.quality >= 90 ? 'var(--success)' : selectedLevel.quality >= 70 ? 'var(--accent)' : 'var(--danger)'};">{selectedLevel.quality}%</span>
            </div>
            <div class="quant-detail-cell">
              <span class="label-xs">EST. SIZE</span>
              <span class="spec-value">{estimatedSizeDisplay}</span>
            </div>
            <div class="quant-detail-cell">
              <span class="label-xs">REDUCTION</span>
              <span class="spec-value" style="color: var(--success);">-{sizeReduction}%</span>
            </div>
            <div class="quant-detail-cell">
              <span class="label-xs">SPEED</span>
              <span class="spec-value">{selectedLevel.speedGain}x</span>
            </div>
          </div>
          <p class="quant-detail-desc">{selectedLevel.description}</p>
        </div>
      </div>
    </div>

    <!-- ── Presets ───────────────────────────────────── -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">PRESETS</span>
      </div>

      <div class="preset-grid">
        {#each PRESETS as preset}
          <button
            class="preset-card panel-flat"
            class:preset-active={selectedLevelIndex === preset.levelIndex}
            onclick={() => selectedLevelIndex = preset.levelIndex}
          >
            <div class="preset-header">
              <span class="preset-icon">{preset.icon}</span>
              <span class="preset-name">{preset.name}</span>
              {#if selectedLevelIndex === preset.levelIndex}
                <span class="dot dot-active" style="margin-left: auto;"></span>
              {/if}
            </div>
            <div class="preset-specs">
              <div class="preset-spec">
                <span class="label-xs">TYPE</span>
                <span class="preset-val">{QUANT_LEVELS[preset.levelIndex].label}</span>
              </div>
              <div class="preset-spec">
                <span class="label-xs">QUALITY</span>
                <span class="preset-val">{QUANT_LEVELS[preset.levelIndex].quality}%</span>
              </div>
              <div class="preset-spec">
                <span class="label-xs">BPW</span>
                <span class="preset-val">{QUANT_LEVELS[preset.levelIndex].bpw}</span>
              </div>
            </div>
            <p class="preset-usecase">{preset.useCase}</p>
          </button>
        {/each}
      </div>
    </div>

    <!-- ── Size / Quality Preview ────────────────────── -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">SIZE / QUALITY PREVIEW</span>
      </div>

      <div class="preview-panel panel-flat">
        <!-- Before / After Header -->
        <div class="ba-header">
          <div class="ba-col">
            <span class="label-xs">CURRENT</span>
            <span class="ba-value">{data.total_memory_display}</span>
            <span class="label-xs" style="color: var(--text-muted);">{model.info?.quantization ?? "ORIGINAL"}</span>
          </div>
          <div class="ba-arrow">&#x2192;</div>
          <div class="ba-col">
            <span class="label-xs">ESTIMATED</span>
            <span class="ba-value" style="color: var(--accent);">{estimatedSizeDisplay}</span>
            <span class="label-xs" style="color: var(--accent);">{selectedLevel.label}</span>
          </div>
          <div class="ba-col ba-col-result">
            <span class="label-xs">REDUCTION</span>
            <span class="ba-reduction">-{sizeReduction}%</span>
          </div>
        </div>

        <!-- Visual Bars -->
        <div class="ba-bars">
          <div class="ba-bar-row">
            <span class="label-xs ba-bar-label">BEFORE</span>
            <div class="ba-bar-track">
              <div class="ba-bar-fill" style="width: 100%; background: var(--text-muted);"></div>
            </div>
            <span class="code ba-bar-size">{data.total_memory_display}</span>
          </div>
          <div class="ba-bar-row">
            <span class="label-xs ba-bar-label">AFTER</span>
            <div class="ba-bar-track">
              <div class="ba-bar-fill" style="width: {Math.min(100, (estimatedSizeBytes / data.total_memory_bytes) * 100)}%; background: var(--accent);"></div>
            </div>
            <span class="code ba-bar-size">{estimatedSizeDisplay}</span>
          </div>
        </div>

        <!-- Per-Component Breakdown -->
        <div class="ba-breakdown">
          <span class="label-xs" style="margin-bottom: 6px; display: block;">COMPONENT BREAKDOWN</span>
          {#each estimatedBreakdown as comp}
            <div class="ba-breakdown-row">
              <span class="ba-comp-name">{comp.name}</span>
              <span class="code ba-comp-val">{comp.currentDisplay}</span>
              <span class="ba-comp-arrow">&#x2192;</span>
              <span class="code ba-comp-val" style="color: {comp.estimatedBytes < comp.currentBytes ? 'var(--accent)' : 'var(--text-secondary)'};">{comp.estimatedDisplay}</span>
            </div>
          {/each}
        </div>

        <!-- Quality & Speed -->
        <div class="ba-estimates">
          <div class="ba-estimate">
            <span class="label-xs">QUALITY ESTIMATE</span>
            <div class="ba-quality-row">
              <div class="ba-bar-track">
                <div class="ba-bar-fill" style="width: {selectedLevel.quality}%; background: {selectedLevel.quality >= 90 ? 'var(--success)' : selectedLevel.quality >= 70 ? 'var(--accent)' : 'var(--danger)'};"></div>
              </div>
              <span class="code" style="color: {selectedLevel.quality >= 90 ? 'var(--success)' : selectedLevel.quality >= 70 ? 'var(--accent)' : 'var(--danger)'};">{selectedLevel.quality}%</span>
            </div>
          </div>
          <div class="ba-estimate">
            <span class="label-xs">SPEED IMPROVEMENT</span>
            <span class="code" style="color: var(--success);">{selectedLevel.speedGain}x FASTER</span>
          </div>
        </div>
      </div>
    </div>

    <!-- ── Quantize Action ───────────────────────────── -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">QUANTIZE</span>
      </div>

      <div class="action-panel panel">
        <div class="action-info-grid">
          <div class="action-info-cell" style="grid-column: span 2;">
            <span class="label-xs">INPUT</span>
            <span class="code" style="font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{model.info?.file_name}</span>
          </div>
          <div class="action-info-cell">
            <span class="label-xs">TARGET</span>
            <span class="code" style="color: var(--accent);">{selectedLevel.targetType}</span>
          </div>
          <div class="action-info-cell">
            <span class="label-xs">EST. OUTPUT</span>
            <span class="code">{estimatedSizeDisplay}</span>
          </div>
          <div class="action-info-cell">
            <span class="label-xs">ENGINE</span>
            <span class="code">LLAMA-QUANTIZE</span>
          </div>
        </div>

        <div class="action-buttons">
          {#if quantizing}
            <button class="btn btn-info" disabled>
              <span style="animation: pulse 1.2s ease infinite;">QUANTIZING...</span>
            </button>
            <span class="badge badge-info">
              <span class="dot dot-working" style="animation: pulse 1.2s ease infinite;"></span>
              PROCESSING
            </span>
          {:else}
            <button class="btn btn-accent" onclick={handleQuantize}>QUANTIZE MODEL</button>
          {/if}
        </div>

        {#if quantizeResult}
          <div class="result-banner panel-flat" style="border-color: var(--success);">
            <span class="dot dot-success"></span>
            <div class="result-text">
              <span class="heading-sm" style="color: var(--success);">QUANTIZATION COMPLETE</span>
              <span class="label-xs" style="color: var(--text-secondary); word-break: break-all;">Output: {quantizeResult.output_path}</span>
              <span class="label-xs" style="color: var(--text-secondary);">Size: {quantizeResult.output_size_display}</span>
            </div>
          </div>
        {/if}

        {#if quantizeError}
          <div class="result-banner panel-flat" style="border-color: var(--danger);">
            <span class="dot dot-danger"></span>
            <span class="danger-text" style="flex: 1;">{quantizeError}</span>
          </div>
        {/if}
      </div>
    </div>

  {/if}
</div>

<style>
  /* ── Page ──────────────────────────────────────── */
  .optimize {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* ── Hero Panel ────────────────────────────────── */
  .hero-top {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }

  .hero-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-primary);
    margin: 0;
  }

  .hero-subtitle {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin: 4px 0 0;
    text-transform: uppercase;
  }

  .hero-specs {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    gap: 1px;
    background: var(--border);
    margin-top: 12px;
  }

  .spec-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    background: var(--bg-surface);
  }

  .spec-cell-wide {
    grid-column: span 2;
  }

  .spec-value {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
  }

  /* ── Requant Warning ─────────────────────────────── */
  .requant-warning {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 10px 12px;
    border-color: var(--accent);
  }

  .requant-text {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--accent);
    line-height: 1.6;
  }

  /* ── Empty States ──────────────────────────────── */
  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 120px;
  }

  .empty-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  /* ── Sections ──────────────────────────────────── */
  .section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-label {
    display: flex;
    align-items: center;
  }

  .divider-label {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .divider-label::before,
  .divider-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border-dim);
  }

  /* ── Quantization Strip ─────────────────────────── */
  .quant-selector {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .quant-strip {
    display: flex;
    gap: 1px;
    background: var(--border-dim);
  }

  .quant-btn {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 2px;
    padding: 10px 4px;
    background: var(--bg-surface);
    border: none;
    cursor: pointer;
    font-family: var(--font-mono);
    transition: all 120ms ease;
    position: relative;
  }

  .quant-btn:hover {
    background: var(--bg-hover);
  }

  .quant-btn-active {
    background: var(--accent-bg) !important;
  }

  .quant-btn-active::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--accent);
  }

  .quant-btn-type {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    transition: color 120ms ease;
  }

  .quant-btn-active .quant-btn-type {
    color: var(--accent);
  }

  .quant-btn-bpw {
    font-size: 8px;
    font-weight: 500;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  /* ── Quant Detail ───────────────────────────────── */
  .quant-detail {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-dim);
  }

  .quant-detail-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
  }

  .quant-detail-name {
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.1em;
    color: var(--text-primary);
  }

  .quant-detail-type {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--accent);
  }

  .quant-detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    gap: 1px;
    background: var(--border-dim);
  }

  .quant-detail-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    background: var(--bg-surface);
  }

  .quant-detail-desc {
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0;
  }

  /* ── Preset Cards ──────────────────────────────── */
  .preset-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .preset-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    cursor: pointer;
    transition: all 120ms ease;
    text-align: left;
    font-family: var(--font-mono);
  }

  .preset-card:hover {
    border-color: var(--border-strong);
    background: var(--bg-hover);
  }

  .preset-active {
    border-color: var(--accent) !important;
    background: var(--accent-bg) !important;
  }

  .preset-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .preset-icon {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-inset);
    border: 1px solid var(--border);
    font-size: 11px;
    font-weight: 800;
    color: var(--accent);
  }

  .preset-name {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--text-primary);
  }

  .preset-specs {
    display: flex;
    gap: 16px;
  }

  .preset-spec {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .preset-val {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
  }

  .preset-usecase {
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding-top: 6px;
    border-top: 1px solid var(--border-dim);
    margin: 0;
    line-height: 1.5;
  }

  /* ── Before / After Preview ────────────────────── */
  .preview-panel {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .ba-header {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .ba-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    flex: 1;
  }

  .ba-col-result {
    flex: 0.8;
  }

  .ba-value {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--text-primary);
  }

  .ba-arrow {
    font-size: 16px;
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .ba-reduction {
    font-size: 18px;
    font-weight: 700;
    color: var(--success);
    letter-spacing: 0.04em;
  }

  /* ── Bar Comparison ────────────────────────────── */
  .ba-bars {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .ba-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .ba-bar-label {
    width: 50px;
    flex-shrink: 0;
    text-align: right;
  }

  .ba-bar-track {
    flex: 1;
    height: 14px;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
    overflow: hidden;
  }

  .ba-bar-fill {
    height: 100%;
    transition: width 300ms ease;
  }

  .ba-bar-size {
    width: 70px;
    flex-shrink: 0;
    text-align: right;
    font-size: 10px;
  }

  /* ── Component Breakdown ───────────────────────── */
  .ba-breakdown {
    padding-top: 12px;
    border-top: 1px solid var(--border-dim);
  }

  .ba-breakdown-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 3px 0;
    font-size: 10px;
  }

  .ba-comp-name {
    flex: 1;
    color: var(--text-secondary);
    font-size: 9px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .ba-comp-val {
    width: 70px;
    text-align: right;
    flex-shrink: 0;
  }

  .ba-comp-arrow {
    color: var(--text-muted);
    font-size: 10px;
    flex-shrink: 0;
  }

  /* ── Quality & Speed Estimates ─────────────────── */
  .ba-estimates {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-top: 12px;
    border-top: 1px solid var(--border-dim);
  }

  .ba-estimate {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .ba-quality-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ── Action Panel ──────────────────────────────── */
  .action-panel {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .action-info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 1px;
    background: var(--border-dim);
  }

  .action-info-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    background: var(--bg-surface);
  }

  .action-buttons {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-top: 8px;
  }

  .result-banner {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px;
    margin-top: 4px;
  }

  .result-text {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
</style>
