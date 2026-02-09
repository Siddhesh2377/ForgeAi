<script lang="ts">
  import { open } from "@tauri-apps/plugin-dialog";
  import { model } from "$lib/model.svelte";

  function getTimestamp() {
    return new Date().toISOString().slice(0, 10).replace(/-/g, ".");
  }

  async function pickGguf() {
    const selected = await open({
      multiple: false,
      filters: [
        {
          name: "GGUF Models",
          extensions: ["gguf"],
        },
      ],
    });

    if (selected) {
      const filePath = Array.isArray(selected) ? selected[0] : selected;
      if (filePath) {
        await model.load(filePath);
      }
    }
  }

  async function pickSafetensorsFile() {
    const selected = await open({
      multiple: false,
      filters: [
        {
          name: "SafeTensors Models",
          extensions: ["safetensors"],
        },
      ],
    });

    if (selected) {
      const filePath = Array.isArray(selected) ? selected[0] : selected;
      if (filePath) {
        await model.load(filePath);
      }
    }
  }

  async function pickSafetensorsFolder() {
    const selected = await open({
      directory: true,
      multiple: false,
    });

    if (selected) {
      const dirPath = Array.isArray(selected) ? selected[0] : selected;
      if (dirPath) {
        await model.loadDir(dirPath);
      }
    }
  }

  let shardInfo = $derived(
    model.info?.shard_count && model.info.shard_count > 0
      ? `${model.info.shard_count} SHARDS`
      : null
  );

  let loadType = $derived(
    model.info?.shard_count && model.info.shard_count > 0
      ? "FOLDER"
      : model.info?.format === "gguf"
        ? "GGUF"
        : model.info?.format === "safe_tensors"
          ? "SAFETENSORS"
          : null
  );
</script>

<div class="load fade-in">
  <!-- ── Hero Panel ────────────────────────────────── -->
  <div class="hero panel">
    <div class="hero-top">
      <span class="label-xs">FRG.01</span>
      <span class="label-xs" style="color: var(--text-muted);">LOAD-ENGINE</span>
      {#if model.status === "loaded"}
        <span class="badge badge-success" style="margin-left: auto;">
          <span class="dot dot-success"></span>
          LOADED
        </span>
      {:else if model.status === "loading"}
        <span class="badge badge-info" style="margin-left: auto;">
          <span class="dot dot-working"></span>
          LOADING
        </span>
      {:else if model.status === "error"}
        <span class="badge badge-danger" style="margin-left: auto;">
          <span class="dot dot-danger"></span>
          ERROR
        </span>
      {:else}
        <span class="badge badge-dim" style="margin-left: auto;">
          <span class="dot"></span>
          IDLE
        </span>
      {/if}
    </div>

    <h1 class="hero-title">LOAD MODEL</h1>
    <p class="hero-subtitle">Model file loader &middot; Header-only parsing</p>

    <!-- Specs Grid -->
    <div class="hero-specs">
      <div class="spec-cell">
        <span class="label-xs">STATUS</span>
        <span class="spec-value"
          class:spec-value-success={model.status === "loaded"}
          class:spec-value-error={model.status === "error"}
          class:spec-value-working={model.status === "loading"}
        >
          {model.status.toUpperCase()}
        </span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">FORMAT</span>
        <span class="spec-value">{model.formatDisplay}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">SIZE</span>
        <span class="spec-value">{model.info?.file_size_display ?? "---"}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">PARAMS</span>
        <span class="spec-value">{model.info?.parameter_count_display ?? "---"}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">TENSORS</span>
        <span class="spec-value">{model.info ? String(model.info.tensor_count) : "---"}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">QUANT</span>
        <span class="spec-value">{model.info?.quantization ?? "NONE"}</span>
      </div>
    </div>

    <div class="hero-footer">
      <div class="barcode" style="max-width: 140px;"></div>
      <span class="label-xs">FRG-001-{getTimestamp()}</span>
    </div>
  </div>

  <!-- ── Loaded Model Info ──────────────────────────── -->
  {#if model.status === "loaded" && model.info}
    <div class="section">
      <div class="section-label">
        <span class="divider-label">LOADED MODEL</span>
      </div>

      <div class="picker-done panel-flat">
        <div class="picker-done-info">
          <span class="dot dot-success" style="width: 8px; height: 8px;"></span>
          <div class="picker-done-text">
            <span class="heading-sm" style="color: var(--success);">{model.info.file_name}</span>
            <span class="label-xs">
              {model.formatDisplay}
              &middot; {model.info.file_size_display}
              &middot; {model.info.parameter_count_display} params
              {#if shardInfo}&middot; {shardInfo}{/if}
              {#if loadType}&middot; {loadType}{/if}
            </span>
          </div>
        </div>
        <div class="picker-done-actions">
          <button class="btn btn-secondary btn-sm" onclick={() => model.unload()}>UNLOAD</button>
        </div>
      </div>

      {#if model.info.shard_count && model.info.shard_count > 0}
        <div class="folder-info panel-inset">
          <div class="folder-info-grid">
            <div class="folder-info-cell">
              <span class="label-xs">SHARDS</span>
              <span class="code" style="color: var(--accent);">{model.info.shard_count}</span>
            </div>
            <div class="folder-info-cell">
              <span class="label-xs">CONFIG</span>
              <span class="code" style="color: {model.info.has_config ? 'var(--success)' : 'var(--text-muted)'};">
                {model.info.has_config ? "FOUND" : "MISSING"}
              </span>
            </div>
            <div class="folder-info-cell">
              <span class="label-xs">TOKENIZER</span>
              <span class="code" style="color: {model.info.has_tokenizer ? 'var(--success)' : 'var(--text-muted)'};">
                {model.info.has_tokenizer ? "FOUND" : "MISSING"}
              </span>
            </div>
            {#if model.info.model_type}
              <div class="folder-info-cell">
                <span class="label-xs">TYPE</span>
                <span class="code">{model.info.model_type.toUpperCase()}</span>
              </div>
            {/if}
            {#if model.info.vocab_size}
              <div class="folder-info-cell">
                <span class="label-xs">VOCAB</span>
                <span class="code">{model.info.vocab_size.toLocaleString()}</span>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- ── Loading State ─────────────────────────────── -->
  {#if model.status === "loading"}
    <div class="section">
      <div class="loading-zone panel-flat">
        <span class="picker-icon" style="animation: pulse 1.2s ease infinite;">+</span>
        <span class="heading-sm" style="color: var(--info);">PARSING MODEL...</span>
        <span class="label-xs">Reading metadata and tensor headers</span>
      </div>
    </div>
  {/if}

  <!-- ── Error Banner ──────────────────────────────── -->
  {#if model.status === "error" && model.error}
    <div class="error-banner panel-flat" style="border-color: var(--danger);">
      <span class="dot dot-danger"></span>
      <span class="danger-text">{model.error}</span>
    </div>
  {/if}

  <!-- ── Load Options Grid ─────────────────────────── -->
  {#if model.status !== "loading"}
    <div class="section">
      <div class="section-label">
        <span class="divider-label">LOAD OPTIONS</span>
      </div>

      <div class="load-grid">
        <button class="load-option panel-flat" onclick={pickGguf}>
          <div class="load-option-header">
            <span class="load-option-code">01</span>
            <span class="load-option-format badge badge-accent">GGUF</span>
          </div>
          <span class="heading-sm">LOAD GGUF FILE</span>
          <span class="label-xs" style="color: var(--text-secondary);">
            Single quantized model file
          </span>
          <span class="label-xs" style="margin-top: auto; color: var(--text-muted);">
            .gguf
          </span>
        </button>

        <button class="load-option panel-flat" onclick={pickSafetensorsFile}>
          <div class="load-option-header">
            <span class="load-option-code">02</span>
            <span class="load-option-format badge badge-dim">SAFETENSORS</span>
          </div>
          <span class="heading-sm">LOAD SAFETENSORS FILE</span>
          <span class="label-xs" style="color: var(--text-secondary);">
            Single SafeTensors weight file
          </span>
          <span class="label-xs" style="margin-top: auto; color: var(--text-muted);">
            .safetensors
          </span>
        </button>

        <button class="load-option panel-flat" onclick={pickSafetensorsFolder}>
          <div class="load-option-header">
            <span class="load-option-code">03</span>
            <span class="load-option-format badge badge-dim">FOLDER</span>
          </div>
          <span class="heading-sm">LOAD MODEL FOLDER</span>
          <span class="label-xs" style="color: var(--text-secondary);">
            Directory with shards, config &amp; tokenizer
          </span>
          <span class="label-xs" style="margin-top: auto; color: var(--text-muted);">
            config.json + *.safetensors + tokenizer
          </span>
        </button>
      </div>
    </div>
  {/if}

  <!-- ── Model Metadata ────────────────────────────── -->
  {#if model.info}
    <div class="section">
      <div class="section-label">
        <span class="divider-label">MODEL METADATA</span>
      </div>

      <div class="metadata-grid">
        {#if model.info.architecture}
          <div class="metadata-cell panel-flat">
            <span class="label-xs">ARCHITECTURE</span>
            <span class="metadata-value">{model.info.architecture.toUpperCase()}</span>
          </div>
        {/if}
        {#if model.info.layer_count}
          <div class="metadata-cell panel-flat">
            <span class="label-xs">LAYERS</span>
            <span class="metadata-value">{model.info.layer_count}</span>
          </div>
        {/if}
        {#if model.info.context_length}
          <div class="metadata-cell panel-flat">
            <span class="label-xs">CONTEXT LENGTH</span>
            <span class="metadata-value">{model.info.context_length.toLocaleString()}</span>
          </div>
        {/if}
        {#if model.info.embedding_size}
          <div class="metadata-cell panel-flat">
            <span class="label-xs">EMBEDDING DIM</span>
            <span class="metadata-value">{model.info.embedding_size.toLocaleString()}</span>
          </div>
        {/if}
        {#if model.info.quantization}
          <div class="metadata-cell panel-flat">
            <span class="label-xs">QUANTIZATION</span>
            <span class="metadata-value">{model.info.quantization}</span>
          </div>
        {/if}
        <div class="metadata-cell panel-flat">
          <span class="label-xs">TENSORS</span>
          <span class="metadata-value">{model.info.tensor_count.toLocaleString()}</span>
        </div>
        <div class="metadata-cell panel-flat">
          <span class="label-xs">PARAMETERS</span>
          <span class="metadata-value">{model.info.parameter_count_display}</span>
        </div>
        <div class="metadata-cell panel-flat">
          <span class="label-xs">FILE SIZE</span>
          <span class="metadata-value">{model.info.file_size_display}</span>
        </div>
      </div>

      <!-- Extra metadata from file -->
      {#if Object.keys(model.info.metadata).length > 0}
        <div class="metadata-extra panel-inset">
          <div class="metadata-extra-header">
            <span class="label-xs">RAW METADATA ({Object.keys(model.info.metadata).length} entries)</span>
          </div>
          <div class="metadata-extra-list">
            {#each Object.entries(model.info.metadata).slice(0, 20) as [key, value]}
              <div class="metadata-extra-row">
                <span class="code metadata-key">{key}</span>
                <span class="code metadata-val">{value}</span>
              </div>
            {/each}
            {#if Object.keys(model.info.metadata).length > 20}
              <div class="metadata-extra-row">
                <span class="label-xs" style="color: var(--text-muted);">
                  ... and {Object.keys(model.info.metadata).length - 20} more entries
                </span>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </div>

    <!-- ── Tensor Preview ──────────────────────────── -->
    {#if model.info.tensor_preview.length > 0}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">TENSOR MAP ({model.info.tensor_count.toLocaleString()} TOTAL)</span>
        </div>

        <div class="tensor-table panel-inset">
          <div class="tensor-header-row">
            <span class="label-xs tensor-col-name">NAME</span>
            <span class="label-xs tensor-col-dtype">DTYPE</span>
            <span class="label-xs tensor-col-shape">SHAPE</span>
          </div>
          {#each model.info.tensor_preview as tensor}
            <div class="tensor-row">
              <span class="code tensor-col-name">{tensor.name}</span>
              <span class="code tensor-col-dtype">{tensor.dtype}</span>
              <span class="code tensor-col-shape">[{tensor.shape.join(", ")}]</span>
            </div>
          {/each}
          {#if model.info.tensor_count > model.info.tensor_preview.length}
            <div class="tensor-row tensor-more">
              <span class="label-xs">
                ... and {model.info.tensor_count - model.info.tensor_preview.length} more tensors
              </span>
            </div>
          {/if}
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .load {
    display: flex;
    flex-direction: column;
    gap: 16px;
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
  }

  .spec-value-success { color: var(--success); }
  .spec-value-error { color: var(--danger); }
  .spec-value-working { color: var(--info); }

  .hero-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border-dim);
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

  /* ── Load Options Grid ─────────────────────────── */
  .load-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .load-option {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 14px;
    cursor: pointer;
    transition: all var(--transition);
    font-family: var(--font-mono);
    text-align: left;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    min-height: 110px;
  }

  .load-option:hover {
    border-color: var(--accent);
    background: var(--accent-bg);
  }

  .load-option-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .load-option-code {
    font-size: 9px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.1em;
  }

  /* ── Loading Zone ──────────────────────────────── */
  .loading-zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 32px 16px;
    border-style: dashed;
  }

  .picker-icon {
    font-size: 28px;
    font-weight: 300;
    color: var(--text-muted);
    line-height: 1;
  }

  /* ── Loaded File Info ─────────────────────────── */
  .picker-done {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px;
    border-color: var(--success);
    gap: 12px;
  }

  .picker-done-info {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }

  .picker-done-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .picker-done-text .heading-sm {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .picker-done-actions {
    display: flex;
    gap: 6px;
    flex-shrink: 0;
  }

  /* ── Folder Info ───────────────────────────────── */
  .folder-info {
    padding: 0;
  }

  .folder-info-grid {
    display: flex;
    gap: 1px;
    background: var(--border-dim);
  }

  .folder-info-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 12px;
    background: var(--bg-inset);
    flex: 1;
    text-align: center;
  }

  /* ── Error Banner ─────────────────────────────── */
  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
  }

  /* ── Metadata Grid ────────────────────────────── */
  .metadata-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }

  .metadata-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px;
  }

  .metadata-value {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    text-transform: uppercase;
  }

  /* ── Metadata Extra ───────────────────────────── */
  .metadata-extra {
    margin-top: 4px;
  }

  .metadata-extra-header {
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-dim);
    margin-bottom: 4px;
  }

  .metadata-extra-list {
    display: flex;
    flex-direction: column;
  }

  .metadata-extra-row {
    display: flex;
    gap: 12px;
    padding: 4px 0;
    border-bottom: 1px solid var(--border-dim);
    align-items: baseline;
  }

  .metadata-extra-row:last-child {
    border-bottom: none;
  }

  .metadata-key {
    color: var(--text-muted);
    font-size: 10px;
    min-width: 200px;
    flex-shrink: 0;
    word-break: break-all;
  }

  .metadata-val {
    color: var(--text-secondary);
    font-size: 10px;
    word-break: break-all;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ── Tensor Table ─────────────────────────────── */
  .tensor-table {
    overflow: hidden;
  }

  .tensor-header-row {
    display: flex;
    gap: 12px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-surface);
  }

  .tensor-row {
    display: flex;
    gap: 12px;
    padding: 4px 12px;
    border-bottom: 1px solid var(--border-dim);
    align-items: baseline;
  }

  .tensor-row:last-child {
    border-bottom: none;
  }

  .tensor-more {
    padding: 8px 12px;
  }

  .tensor-col-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 10px;
  }

  .tensor-col-dtype {
    width: 70px;
    flex-shrink: 0;
    font-size: 10px;
    color: var(--accent);
  }

  .tensor-col-shape {
    width: 180px;
    flex-shrink: 0;
    font-size: 10px;
    color: var(--text-secondary);
  }

  /* ── Responsive ────────────────────────────────── */
  @media (max-width: 700px) {
    .load-grid {
      grid-template-columns: 1fr;
    }

    .metadata-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .hero-specs {
      grid-template-columns: repeat(3, 1fr);
    }
  }
</style>
