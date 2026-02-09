<script lang="ts">
  import { convert } from "$lib/convert.svelte";
  import { hub } from "$lib/hub.svelte";
  import { goto } from "$app/navigation";
  import { model } from "$lib/model.svelte";

  type OutType = "f16" | "f32" | "bf16" | "q8_0" | "auto";

  let selectedRepo = $state<string | null>(null);
  let outtype = $state<OutType>("f16");

  // Load deps + library on mount
  $effect(() => {
    convert.checkDeps();
    hub.loadLibrary();
  });

  // Filter library to repos only (directories with safetensors)
  let repoModels = $derived(
    hub.localModels.filter((m) => m.format === "repo"),
  );

  function handleSelectRepo(repoPath: string) {
    selectedRepo = repoPath;
    convert.reset();
    convert.detectModel(repoPath);
  }

  function handleConvert() {
    if (!selectedRepo || convert.converting) return;
    convert.run(selectedRepo, outtype);
  }

  async function handleLoadResult() {
    if (!convert.convertResult) return;
    await model.load(convert.convertResult.output_path);
    goto("/load");
  }

  let isReady = $derived(convert.deps?.ready ?? false);

  let progressPercent = $derived(
    convert.convertProgress && convert.convertProgress.percent >= 0
      ? convert.convertProgress.percent
      : null,
  );

  let canConvert = $derived(
    isReady &&
      selectedRepo !== null &&
      convert.modelInfo !== null &&
      convert.modelInfo.has_config &&
      convert.modelInfo.safetensor_count > 0 &&
      !convert.converting,
  );

  const outTypes: { value: OutType; label: string; desc: string }[] = [
    { value: "f16", label: "F16", desc: "Half precision — good default" },
    { value: "bf16", label: "BF16", desc: "Brain float16" },
    { value: "f32", label: "F32", desc: "Full precision — largest" },
    { value: "q8_0", label: "Q8_0", desc: "8-bit quantized — smallest" },
    { value: "auto", label: "AUTO", desc: "Auto-detect from source" },
  ];
</script>

<div class="convert fade-in">
  <!-- ── Hero Panel ──────────────────────────────── -->
  <div class="hero panel">
    <div class="hero-top">
      <span class="label-xs">FRG.05</span>
      <span class="label-xs" style="color: var(--text-muted);">CONVERT-ENGINE</span>
      <span class="badge {isReady ? 'badge-accent' : 'badge-dim'}" style="margin-left: auto;">
        <span class="dot {isReady ? 'dot-active' : 'dot-danger'}"></span>
        {isReady ? "READY" : "SETUP REQUIRED"}
      </span>
      {#if isReady && !convert.setupRunning}
        <button class="btn btn-ghost btn-sm" onclick={() => convert.setup()}>
          UPDATE TOOLS
        </button>
      {/if}
    </div>

    <h1 class="hero-title">CONVERT</h1>
    <p class="hero-subtitle">SafeTensors &rarr; GGUF Conversion</p>

    <div class="hero-specs">
      <div class="spec-cell">
        <span class="label-xs">PYTHON</span>
        <span class="spec-value">
          {convert.deps?.python_found ? convert.deps.python_version?.replace("Python ", "") ?? "OK" : "NOT FOUND"}
        </span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">VENV</span>
        <span class="spec-value">{convert.deps?.venv_ready ? "READY" : "---"}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">SCRIPT</span>
        <span class="spec-value">{convert.deps?.script_ready ? "READY" : "---"}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">PACKAGES</span>
        <span class="spec-value">{convert.deps?.packages_ready ? "READY" : "MISSING"}</span>
      </div>
    </div>
  </div>

  <!-- ── Setup Section ───────────────────────────── -->
  {#if (!isReady && !convert.depsLoading) || convert.setupRunning || convert.setupError}
    <div class="section">
      <div class="section-label">
        <span class="divider-label">SETUP</span>
      </div>

      {#if !convert.deps?.python_found}
        <div class="empty-state panel-flat" style="border-color: var(--danger);">
          <div class="error-inner">
            <span class="dot dot-danger"></span>
            <div>
              <span class="danger-text">PYTHON 3 NOT FOUND</span>
              <p class="label-xs" style="margin-top: 4px; color: var(--text-secondary);">
                Install Python 3.10+ and ensure it's available in your PATH.
              </p>
            </div>
          </div>
        </div>
      {:else if convert.setupRunning}
        <div class="progress-section panel">
          <div class="progress-header">
            <span class="heading-sm">INSTALLING DEPENDENCIES</span>
            <span class="badge badge-info">
              <span class="dot dot-working" style="animation: pulse 1.2s ease infinite;"></span>
              SETUP
            </span>
          </div>

          {#if convert.setupProgress}
            <div class="progress-bar-row">
              <div class="progress-track">
                <div
                  class="progress-fill"
                  style="width: {convert.setupProgress.percent}%;"
                ></div>
              </div>
            </div>
            <div class="progress-message">
              <span class="code">{convert.setupProgress.message}</span>
            </div>
          {/if}
        </div>
      {:else}
        <div class="setup-panel panel-flat">
          <div class="setup-inner">
            <span class="heading-sm" style="color: var(--text-secondary);">DEPENDENCIES NOT INSTALLED</span>
            <p class="label-xs" style="margin-top: 4px;">
              This will create a Python virtual environment and install required packages.
              Download size: ~500 MB. This is a one-time setup.
            </p>
            {#if convert.deps?.missing_packages.length}
              <div class="missing-list">
                <span class="label-xs">MISSING:</span>
                {#each convert.deps.missing_packages as pkg}
                  <span class="badge badge-dim">{pkg}</span>
                {/each}
              </div>
            {/if}
            <button class="btn btn-accent" style="margin-top: 12px;" onclick={() => convert.setup()}>
              INSTALL DEPENDENCIES
            </button>
          </div>
        </div>

        {#if convert.setupError}
          <div class="empty-state panel-flat" style="border-color: var(--danger);">
            <div class="error-inner">
              <span class="dot dot-danger"></span>
              <span class="danger-text">{convert.setupError}</span>
            </div>
          </div>
        {/if}
      {/if}
    </div>
  {/if}

  <!-- ── Source Selection ────────────────────────── -->
  {#if isReady}
    <div class="section">
      <div class="section-label">
        <span class="divider-label">SOURCE MODEL</span>
      </div>

      {#if repoModels.length === 0}
        <div class="empty-state panel-flat" style="border-style: dashed;">
          <div class="empty-inner">
            <span class="heading-sm" style="color: var(--text-muted);">NO REPOS DOWNLOADED</span>
            <span class="label-xs" style="margin-top: 4px;">
              Download a full repository from the HUB to convert SafeTensors to GGUF
            </span>
            <button
              class="btn btn-accent"
              style="margin-top: 12px;"
              onclick={() => goto("/hub")}
            >
              GO TO HUB
            </button>
          </div>
        </div>
      {:else}
        <div class="repo-list">
          {#each repoModels as repo}
            <button
              class="repo-item panel-flat"
              class:repo-item-selected={selectedRepo === repo.file_path}
              onclick={() => handleSelectRepo(repo.file_path)}
            >
              <div class="repo-item-header">
                <span class="heading-sm">{repo.file_name}</span>
                <span class="badge badge-info">REPO</span>
              </div>
              <div class="repo-item-meta">
                <span class="label-xs">{repo.file_size_display}</span>
                {#if repo.source_repo}
                  <span class="label-xs" style="color: var(--text-muted);">{repo.source_repo}</span>
                {/if}
              </div>
            </button>
          {/each}
        </div>
      {/if}
    </div>

    <!-- ── Model Info ──────────────────────────────── -->
    {#if convert.modelLoading}
      <div class="empty-state panel-flat">
        <span class="heading-sm" style="color: var(--info); animation: pulse 1.2s ease infinite;">
          ANALYZING MODEL...
        </span>
      </div>
    {:else if convert.modelError}
      <div class="empty-state panel-flat" style="border-color: var(--danger);">
        <div class="error-inner">
          <span class="dot dot-danger"></span>
          <span class="danger-text">{convert.modelError}</span>
        </div>
      </div>
    {:else if convert.modelInfo}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">MODEL ANALYSIS</span>
        </div>

        <div class="model-info panel-flat">
          <div class="info-grid">
            <div class="info-cell">
              <span class="label-xs">ARCHITECTURE</span>
              <span class="info-value">
                {convert.modelInfo.architectures.length > 0
                  ? convert.modelInfo.architectures.join(", ")
                  : "UNKNOWN"}
              </span>
            </div>
            <div class="info-cell">
              <span class="label-xs">MODEL TYPE</span>
              <span class="info-value">{convert.modelInfo.model_type ?? "---"}</span>
            </div>
            <div class="info-cell">
              <span class="label-xs">HIDDEN SIZE</span>
              <span class="info-value">{convert.modelInfo.hidden_size ?? "---"}</span>
            </div>
            <div class="info-cell">
              <span class="label-xs">LAYERS</span>
              <span class="info-value">{convert.modelInfo.num_layers ?? "---"}</span>
            </div>
            <div class="info-cell">
              <span class="label-xs">VOCAB SIZE</span>
              <span class="info-value">{convert.modelInfo.vocab_size ?? "---"}</span>
            </div>
            <div class="info-cell">
              <span class="label-xs">SAFETENSORS</span>
              <span class="info-value">
                {convert.modelInfo.safetensor_count} FILES ({convert.modelInfo.total_size_display})
              </span>
            </div>
          </div>

          <div class="info-checks">
            <div class="check-item" class:check-ok={convert.modelInfo.has_config}>
              <span class="dot {convert.modelInfo.has_config ? 'dot-success' : 'dot-danger'}"></span>
              <span class="label-xs">CONFIG.JSON</span>
            </div>
            <div class="check-item" class:check-ok={convert.modelInfo.has_tokenizer}>
              <span class="dot {convert.modelInfo.has_tokenizer ? 'dot-success' : 'dot-danger'}"></span>
              <span class="label-xs">TOKENIZER</span>
            </div>
            <div class="check-item" class:check-ok={convert.modelInfo.has_tokenizer_model}>
              <span class="dot {convert.modelInfo.has_tokenizer_model ? 'dot-success' : 'dot-warning'}"></span>
              <span class="label-xs">TOKENIZER.MODEL</span>
            </div>
          </div>

          {#if !convert.modelInfo.has_config}
            <div class="info-warning">
              <span class="dot dot-danger"></span>
              <span class="danger-text">config.json is required for conversion</span>
            </div>
          {/if}
          {#if convert.modelInfo.safetensor_count === 0}
            <div class="info-warning">
              <span class="dot dot-danger"></span>
              <span class="danger-text">No SafeTensors files found in repository</span>
            </div>
          {/if}
        </div>
      </div>

      <!-- ── Output Settings ─────────────────────────── -->
      <div class="section">
        <div class="section-label">
          <span class="divider-label">OUTPUT TYPE</span>
        </div>

        <div class="outtype-grid">
          {#each outTypes as ot}
            <button
              class="outtype-btn"
              class:outtype-btn-active={outtype === ot.value}
              onclick={() => (outtype = ot.value)}
            >
              <span class="outtype-label">{ot.label}</span>
              <span class="outtype-desc">{ot.desc}</span>
            </button>
          {/each}
        </div>
      </div>

      <!-- ── Convert Button ──────────────────────────── -->
      <div class="section">
        <button
          class="btn btn-accent convert-btn"
          disabled={!canConvert}
          onclick={handleConvert}
        >
          {convert.converting ? "CONVERTING..." : "CONVERT TO GGUF"}
        </button>
      </div>
    {/if}

    <!-- ── Progress ────────────────────────────────── -->
    {#if convert.converting && convert.convertProgress}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">CONVERSION PROGRESS</span>
        </div>

        <div class="progress-section panel">
          <div class="progress-header">
            <span class="heading-sm">{convert.convertProgress.stage.toUpperCase()}</span>
            <span class="badge badge-info">
              <span class="dot dot-working" style="animation: pulse 1.2s ease infinite;"></span>
              CONVERTING
            </span>
          </div>

          <div class="progress-bar-row">
            <div class="progress-track">
              {#if progressPercent !== null}
                <div
                  class="progress-fill"
                  style="width: {progressPercent}%;"
                ></div>
              {:else}
                <div class="progress-fill progress-fill-indeterminate"></div>
              {/if}
            </div>
          </div>

          <div class="progress-message">
            <span class="code">{convert.convertProgress.message}</span>
          </div>

          <div class="progress-actions">
            {#if progressPercent !== null}
              <span class="code" style="color: var(--accent);">
                {progressPercent.toFixed(1)}%
              </span>
            {/if}
            <button class="btn btn-sm btn-danger" onclick={() => convert.cancel()}>
              CANCEL
            </button>
          </div>
        </div>
      </div>
    {/if}

    <!-- ── Error ───────────────────────────────────── -->
    {#if convert.convertError}
      <div class="empty-state panel-flat" style="border-color: var(--danger);">
        <div class="error-inner">
          <span class="dot dot-danger"></span>
          <span class="danger-text">{convert.convertError}</span>
        </div>
      </div>
    {/if}

    <!-- ── Result ──────────────────────────────────── -->
    {#if convert.convertResult}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">RESULT</span>
        </div>

        <div class="result-panel panel" style="border-color: var(--success);">
          <div class="result-header">
            <span class="dot dot-success" style="width: 8px; height: 8px;"></span>
            <span class="heading-sm" style="color: var(--success);">CONVERSION COMPLETE</span>
          </div>

          <div class="result-grid">
            <div class="info-cell">
              <span class="label-xs">OUTPUT</span>
              <span class="info-value" style="font-size: 10px; word-break: break-all;">
                {convert.convertResult.output_path}
              </span>
            </div>
            <div class="info-cell">
              <span class="label-xs">SIZE</span>
              <span class="info-value">{convert.convertResult.output_size_display}</span>
            </div>
          </div>

          <div class="result-actions">
            <button class="btn btn-accent" onclick={handleLoadResult}>
              LOAD MODEL
            </button>
            <button class="btn btn-secondary" onclick={() => goto("/hub")}>
              BACK TO HUB
            </button>
          </div>
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  /* ── Page ──────────────────────────────────────── */
  .convert {
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
    grid-template-columns: repeat(4, 1fr);
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

  .spec-value {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-primary);
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

  /* ── Empty / Error States ──────────────────────── */
  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100px;
    padding: 16px;
  }

  .empty-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .error-inner {
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }

  /* ── Setup Panel ───────────────────────────────── */
  .setup-panel {
    padding: 16px;
  }

  .setup-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .missing-list {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    flex-wrap: wrap;
    justify-content: center;
  }

  /* ── Repo Selection ────────────────────────────── */
  .repo-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .repo-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px 12px;
    cursor: pointer;
    text-align: left;
    font-family: var(--font-mono);
    border: 1px solid var(--border);
    background: var(--bg-surface);
    transition: all var(--transition);
  }

  .repo-item:hover {
    border-color: var(--border-strong);
    background: var(--bg-hover);
  }

  .repo-item-selected {
    border-color: var(--accent);
    background: var(--accent-bg);
  }

  .repo-item-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .repo-item-meta {
    display: flex;
    gap: 12px;
  }

  /* ── Model Info ────────────────────────────────── */
  .model-info {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border-dim);
  }

  .info-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    background: var(--bg-surface);
  }

  .info-value {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-primary);
  }

  .info-checks {
    display: flex;
    gap: 16px;
    padding: 8px 0;
    border-top: 1px solid var(--border-dim);
  }

  .check-item {
    display: flex;
    align-items: center;
    gap: 6px;
    opacity: 0.5;
  }

  .check-ok {
    opacity: 1;
  }

  .info-warning {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 0;
  }

  /* ── Output Type ───────────────────────────────── */
  .outtype-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
  }

  .outtype-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 10px 8px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    cursor: pointer;
    font-family: var(--font-mono);
    transition: all var(--transition);
  }

  .outtype-btn:hover {
    border-color: var(--border-strong);
  }

  .outtype-btn-active {
    border-color: var(--accent);
    background: var(--accent-bg);
  }

  .outtype-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-primary);
  }

  .outtype-desc {
    font-size: 8px;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    text-transform: uppercase;
    text-align: center;
  }

  /* ── Convert Button ────────────────────────────── */
  .convert-btn {
    width: 100%;
    padding: 14px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
  }

  /* ── Progress ──────────────────────────────────── */
  .progress-section {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .progress-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .progress-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .progress-track {
    flex: 1;
    height: 14px;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent);
    transition: width 300ms ease;
  }

  .progress-fill-indeterminate {
    width: 30%;
    animation: indeterminate 1.5s ease-in-out infinite;
  }

  @keyframes indeterminate {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(400%);
    }
  }

  .progress-message {
    overflow: hidden;
  }

  .progress-message .code {
    font-size: 10px;
    color: var(--text-secondary);
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .progress-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  /* ── Result Panel ──────────────────────────────── */
  .result-panel {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .result-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .result-grid {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 1px;
    background: var(--border-dim);
  }

  .result-grid .info-cell {
    background: var(--bg-surface);
  }

  .result-actions {
    display: flex;
    gap: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-dim);
  }

  /* ── Responsive ────────────────────────────────── */
  @media (max-width: 600px) {
    .hero-specs {
      grid-template-columns: repeat(2, 1fr);
    }

    .info-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .outtype-grid {
      grid-template-columns: repeat(3, 1fr);
    }
  }
</style>
