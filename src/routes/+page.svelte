<script lang="ts">
  import { goto } from "$app/navigation";
  import { model } from "$lib/model.svelte";
  import { dna } from "$lib/dna.svelte";
  import { training } from "$lib/training.svelte";
  import { datastudio } from "$lib/datastudio.svelte";

  function getSerial() {
    return `FRG-${String(Date.now()).slice(-6)}`;
  }

  const serial = getSerial();

  // Overall system status derived from all modules
  const sysStatus = $derived.by(() => {
    if (training.training) return { label: "TRAINING", color: "info" };
    if (dna.merging) return { label: "MERGING", color: "info" };
    if (dna.analyzing) return { label: "ANALYZING", color: "info" };
    if (dna.profiling) return { label: "PROFILING", color: "info" };
    if (dna.status === "complete") return { label: "COMPLETE", color: "success" };
    if (dna.status === "error") return { label: "ERROR", color: "danger" };
    if (model.status === "loading") return { label: "LOADING", color: "info" };
    if (model.isLoaded) return { label: "LOADED", color: "success" };
    return { label: "IDLE", color: "accent" };
  });

  interface ModuleDef {
    code: string;
    name: string;
    href: string;
    desc: string;
    formats: string;
    statusWhenLoaded: string;
    statusWhenIdle: string;
  }

  const moduleGroups: { label: string; modules: ModuleDef[] }[] = [
    { label: "MODEL", modules: [
      { code: "01", name: "LOAD", href: "/load", desc: "Import model from disk", formats: "GGUF / SAFETENSORS", statusWhenLoaded: "loaded", statusWhenIdle: "ready" },
      { code: "02", name: "INSPECT", href: "/inspect", desc: "View architecture & tensors", formats: "LAYERS / SHAPES / METADATA", statusWhenLoaded: "ready", statusWhenIdle: "awaiting" },
      { code: "03", name: "COMPRESS", href: "/optimize", desc: "Quantize & compress", formats: "INT8 / INT4 / GPTQ", statusWhenLoaded: "ready", statusWhenIdle: "awaiting" },
    ]},
    { label: "DATA", modules: [
      { code: "04", name: "HUB", href: "/hub", desc: "Download & manage models", formats: "HUGGINGFACE / GGUF / ST", statusWhenLoaded: "ready", statusWhenIdle: "ready" },
      { code: "10", name: "DATASTUDIO", href: "/datastudio", desc: "Explore & prepare datasets", formats: "JSON / JSONL / CSV / PARQUET", statusWhenLoaded: "ready", statusWhenIdle: "ready" },
      { code: "06", name: "TRAINING", href: "/training", desc: "Fine-tune & layer surgery", formats: "LORA / QLORA / SFT / DPO", statusWhenLoaded: "ready", statusWhenIdle: "ready" },
    ]},
    { label: "TOOLS", modules: [
      { code: "05", name: "CONVERT", href: "/convert", desc: "SafeTensors to GGUF", formats: "F16 / F32 / BF16 / Q8_0", statusWhenLoaded: "ready", statusWhenIdle: "ready" },
      { code: "08", name: "M-DNA", href: "/dna", desc: "Merge multiple models", formats: "SLERP / TIES / DARE / FRANK", statusWhenLoaded: "ready", statusWhenIdle: "ready" },
      { code: "09", name: "TEST", href: "/test", desc: "Run model inference", formats: "GGUF / SAFETENSORS / PROMPT", statusWhenLoaded: "ready", statusWhenIdle: "ready" },
    ]},
  ];

  // Live activity status per module code
  const moduleActivity = $derived.by(() => {
    const act: Record<string, { label: string; color: string; active: boolean }> = {};
    if (training.training) act["06"] = { label: `TRAINING ${Math.round(training.progress?.percent ?? 0)}%`, color: "info", active: true };
    else if (training.surgeryRunning) act["06"] = { label: "SURGERY", color: "info", active: true };
    if (dna.merging) act["08"] = { label: `MERGING ${Math.round(dna.mergeProgress?.percent ?? 0)}%`, color: "info", active: true };
    else if (dna.analyzing) act["08"] = { label: "ANALYZING", color: "info", active: true };
    else if (dna.status === "complete") act["08"] = { label: "COMPLETE", color: "success", active: false };
    if (datastudio.loading) act["10"] = { label: "LOADING", color: "info", active: true };
    else if (datastudio.dataset) act["10"] = { label: `${datastudio.dataset.rows.toLocaleString()} ROWS`, color: "success", active: false };
    if (model.status === "loading") act["01"] = { label: "LOADING", color: "info", active: true };
    return act;
  });

  function getModuleStatus(mod: ModuleDef): string {
    return model.isLoaded ? mod.statusWhenLoaded : mod.statusWhenIdle;
  }

  const specs = $derived([
    { label: "STATUS", value: sysStatus.label, color: sysStatus.color },
    { label: "MODEL", value: model.info?.file_name ?? "---" },
    { label: "FORMAT", value: model.formatDisplay },
    { label: "PARAMS", value: model.info?.parameter_count_display ?? "---" },
    { label: "SIZE", value: model.info?.file_size_display ?? "---" },
    { label: "QUANT", value: model.info?.quantization ?? "NONE" },
  ]);
</script>

<div class="dashboard fade-in">
  <!-- ── Hero Panel ──────────────────────────────── -->
  <div class="hero panel">
    <div class="hero-header">
      <div class="hero-id">
        <span class="label-xs">{serial}</span>
        <span class="label-xs" style="color: var(--text-muted);">REV.00</span>
      </div>
      <div class="badge" class:badge-accent={sysStatus.color === "accent"} class:badge-info={sysStatus.color === "info"} class:badge-success={sysStatus.color === "success"} class:badge-danger={sysStatus.color === "danger"}>
        <span class="dot" class:dot-active={sysStatus.color === "accent"} class:dot-working={sysStatus.color === "info"} class:dot-success={sysStatus.color === "success"} class:dot-danger={sysStatus.color === "danger"}></span>
        {sysStatus.label}
      </div>
    </div>

    <div class="hero-body">
      <h1 class="hero-title">FORGEAI</h1>
      <p class="hero-subtitle">MODEL OPTIMIZATION WORKBENCH</p>
    </div>

    <div class="hero-specs">
      {#each specs as spec}
        <div class="spec-cell">
          <span class="label-xs">{spec.label}</span>
          <span class="spec-value" style={spec.color ? `color: var(--${spec.color})` : ''}>{spec.value}</span>
        </div>
      {/each}
    </div>

    <div class="hero-footer">
      <div class="barcode" style="max-width: 180px;"></div>
      <span class="label-xs">{serial} / CPU / LOCAL</span>
    </div>
  </div>

  <!-- ── Active Task Banner ────────────────────────── -->
  {#if dna.merging}
    <div class="task-banner panel-flat" style="border-color: var(--info);">
      <div class="task-banner-inner">
        <span class="dot dot-working"></span>
        <span class="heading-sm" style="color: var(--info); margin-left: 8px;">MERGE IN PROGRESS</span>
        <span class="label-xs" style="margin-left: auto; color: var(--info);">
          {Math.round(dna.mergeProgress?.percent ?? 0)}%
          {#if dna.mergeProgress?.current_tensor}
            &middot; {dna.mergeProgress.current_tensor}
          {/if}
        </span>
      </div>
      <div class="task-banner-bar">
        <div class="task-banner-fill" style="width: {dna.mergeProgress?.percent ?? 0}%;"></div>
      </div>
    </div>
  {:else if dna.analyzing}
    <div class="task-banner panel-flat" style="border-color: var(--info);">
      <div class="task-banner-inner">
        <span class="dot dot-working"></span>
        <span class="heading-sm" style="color: var(--info); margin-left: 8px;">ANALYZING LAYERS</span>
        <span class="label-xs" style="margin-left: auto; color: var(--info);">{Math.round(dna.analysisProgress?.percent ?? 0)}%</span>
      </div>
      <div class="task-banner-bar">
        <div class="task-banner-fill" style="width: {dna.analysisProgress?.percent ?? 0}%;"></div>
      </div>
    </div>
  {:else if dna.status === "complete" && dna.mergeResult}
    <div class="task-banner panel-flat" style="border-color: var(--success);">
      <div class="task-banner-inner">
        <span class="dot dot-success"></span>
        <span class="heading-sm" style="color: var(--success); margin-left: 8px;">MERGE COMPLETE</span>
        <span class="label-xs" style="margin-left: auto;">{dna.mergeResult.output_size_display} &middot; {dna.mergeResult.tensors_written} TENSORS</span>
      </div>
    </div>
  {/if}

  <!-- ── Load Prompt ─────────────────────────────── -->
  {#if model.isLoaded && model.info}
    <div class="load-prompt panel-flat" style="border-style: solid; border-color: var(--success);">
      <div class="load-prompt-inner">
        <span class="dot dot-success" style="width: 8px; height: 8px;"></span>
        <p class="heading-sm" style="color: var(--success); margin-top: 8px;">
          MODEL LOADED
        </p>
        <p class="label-xs" style="margin-top: 4px;">
          {model.info.file_name} &middot; {model.formatDisplay} &middot; {model.info.parameter_count_display}
        </p>
        <button class="btn btn-accent" style="margin-top: 12px;" onclick={() => goto('/load')}>
          VIEW DETAILS
        </button>
      </div>
    </div>
  {:else}
    <div class="load-prompt panel-flat">
      <div class="load-prompt-inner">
        <div class="crosshair">+</div>
        <p class="heading-sm" style="color: var(--text-secondary);">NO MODEL LOADED</p>
        <p class="label-xs" style="margin-top: 4px;">
          Use the LOAD module to begin
        </p>
        <p class="label-xs" style="margin-top: 8px; color: var(--text-muted);">
          Supported: GGUF &middot; SafeTensors
        </p>
        <button class="btn btn-accent" style="margin-top: 12px;" onclick={() => goto('/load')}>
          LOAD MODEL
        </button>
      </div>
    </div>
  {/if}

  <!-- ── Module Grid ─────────────────────────────── -->
  {#each moduleGroups as group}
    <div class="section-header">
      <span class="divider-label">{group.label}</span>
    </div>

    <div class="module-grid">
      {#each group.modules as mod}
        {@const status = getModuleStatus(mod)}
        {@const activity = moduleActivity[mod.code]}
        {#if status === "awaiting"}
          <div class="module-card panel-flat">
            <div class="module-header">
              <span class="module-code">{mod.code}</span>
              <span class="module-name">{mod.name}</span>
              <span class="dot"></span>
            </div>
            <p class="module-desc">{mod.desc}</p>
            <div class="module-footer">
              <span class="label-xs">{mod.formats}</span>
            </div>
            <div class="module-overlay">
              <span class="danger-text">REQUIRES MODEL</span>
            </div>
          </div>
        {:else}
          <a class="module-card module-ready panel-flat" href={mod.href}>
            <div class="module-header">
              <span class="module-code">{mod.code}</span>
              <span class="module-name">{mod.name}</span>
              {#if activity}
                <span class="dot" class:dot-working={activity.active} class:dot-success={activity.color === "success"}></span>
              {:else if status === "loaded"}
                <span class="dot dot-success"></span>
              {:else}
                <span class="dot dot-active"></span>
              {/if}
            </div>
            <p class="module-desc">{mod.desc}</p>
            {#if activity}
              <div class="module-activity">
                <span class="label-xs" style="color: var(--{activity.color});">{activity.label}</span>
              </div>
            {/if}
            <div class="module-footer">
              <span class="label-xs">{mod.formats}</span>
            </div>
          </a>
        {/if}
      {/each}
    </div>
  {/each}

  <!-- ── System Info ─────────────────────────────── -->
  <div class="section-header">
    <span class="divider-label">SYSTEM</span>
  </div>

  <div class="sys-grid">
    <div class="sys-cell panel-inset">
      <span class="label-xs">RUNTIME</span>
      <span class="code">Tauri v2 + Rust</span>
    </div>
    <div class="sys-cell panel-inset">
      <span class="label-xs">EXECUTION</span>
      <span class="code">CPU (Local)</span>
    </div>
    <div class="sys-cell panel-inset">
      <span class="label-xs">INTERFACE</span>
      <span class="code">SvelteKit 5</span>
    </div>
    <div class="sys-cell panel-inset">
      <span class="label-xs">VERSION</span>
      <span class="code">v0.1.0-dev</span>
    </div>
  </div>
</div>

<style>
  .dashboard {
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-height: 100%;
  }

  /* ── Hero ──────────────────────────────────────── */
  .hero {
    padding: 16px;
  }

  .hero-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
  }

  .hero-id {
    display: flex;
    gap: 12px;
  }

  .hero-body {
    margin-bottom: 20px;
  }

  .hero-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 0.2em;
    color: var(--text-primary);
    line-height: 1;
  }

  .hero-subtitle {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.18em;
    color: var(--text-secondary);
    margin-top: 6px;
    text-transform: uppercase;
  }

  .hero-specs {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    border: 1px solid var(--border-dim);
  }

  .spec-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    border-right: 1px solid var(--border-dim);
  }

  .spec-cell:last-child {
    border-right: none;
  }

  .spec-value {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--text-primary);
  }

  .hero-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border-dim);
  }

  /* ── Load Prompt ───────────────────────────────── */
  .load-prompt {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 140px;
    border-style: dashed;
  }

  .load-prompt-inner {
    text-align: center;
  }

  .crosshair {
    font-size: 24px;
    color: var(--text-muted);
    line-height: 1;
    margin-bottom: 8px;
    font-weight: 300;
  }

  /* ── Task Banner ────────────────────────────────── */
  .task-banner {
    border-style: solid;
    padding: 10px 14px;
  }

  .task-banner-inner {
    display: flex;
    align-items: center;
  }

  .task-banner-bar {
    height: 3px;
    background: var(--bg-inset);
    margin-top: 8px;
    overflow: hidden;
  }

  .task-banner-fill {
    height: 100%;
    background: var(--info);
    transition: width 300ms ease;
  }

  /* ── Section Header ────────────────────────────── */
  .section-header {
    padding: 4px 0;
  }

  /* ── Module Grid ───────────────────────────────── */
  .module-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
  }

  .module-card {
    position: relative;
    background: var(--bg-surface);
    border: none;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 12px;
    transition: background var(--transition);
  }

  .module-card:hover {
    background: var(--bg-hover);
  }

  .module-ready {
    cursor: pointer;
    text-decoration: none;
    color: inherit;
  }

  .module-header {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .module-code {
    font-size: 9px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.1em;
  }

  .module-name {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.1em;
    flex: 1;
  }

  .module-desc {
    font-size: 10px;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
  }

  .module-footer {
    margin-top: auto;
    padding-top: 6px;
    border-top: 1px solid var(--border-dim);
  }

  .module-activity {
    padding: 2px 0;
    font-size: 10px;
    letter-spacing: 0.06em;
  }

  .module-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: color-mix(in srgb, var(--bg-surface) 85%, transparent);
    backdrop-filter: blur(2px);
    -webkit-backdrop-filter: blur(2px);
    border: 1px solid var(--on-danger);
  }

  /* ── System Grid ───────────────────────────────── */
  .sys-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }

  .sys-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  /* ── Responsive ────────────────────────────────── */
  @media (max-width: 700px) {
    .hero-specs {
      grid-template-columns: repeat(3, 1fr);
    }

    .module-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .sys-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
