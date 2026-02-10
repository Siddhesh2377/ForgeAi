<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { theme, FONT_FAMILIES, type ThemeMode, type FontFamily, type FontSize } from "$lib/theme.svelte";

  interface GpuInfo {
    has_nvidia: boolean;
    nvidia_name: string | null;
    nvidia_vram: string | null;
    cuda_version: string | null;
    has_vulkan: boolean;
    has_metal: boolean;
    recommended_variant: string;
    os: string;
    arch: string;
  }

  interface ToolsStatus {
    installed: boolean;
    version: string | null;
    variant: string | null;
    path: string | null;
  }

  const modes: { id: ThemeMode; label: string }[] = [
    { id: "dark", label: "DARK" },
    { id: "light", label: "LIGHT" },
  ];

  const FONT_SIZES: FontSize[] = [10, 11, 12, 13, 14];

  const VARIANTS = [
    { id: "cpu", label: "CPU", desc: "Universal, no GPU acceleration" },
    { id: "cuda", label: "CUDA", desc: "NVIDIA GPU acceleration" },
    { id: "vulkan", label: "VULKAN", desc: "Cross-platform GPU (AMD/NVIDIA/Intel)" },
  ];

  interface SystemInfo {
    total_ram_mb: number;
    available_ram_mb: number;
    used_ram_mb: number;
    cpu_name: string;
    cpu_cores: number;
    cpu_threads: number;
  }

  interface AppSettings {
    memory_limit_mb: number | null;
  }

  // ── System Info State ──────────────────────────────
  let sysInfo = $state<SystemInfo | null>(null);
  let sysInfoLoading = $state(true);
  let memoryLimitMb = $state(2000);

  // ── GPU & Tools State ────────────────────────────
  let gpu = $state<GpuInfo | null>(null);
  let gpuLoading = $state(true);
  let tools = $state<ToolsStatus | null>(null);
  let toolsLoading = $state(true);

  let selectedVariant = $state("cpu");
  let downloading = $state(false);
  let downloadError = $state<string | null>(null);
  let removing = $state(false);

  // ── Convert Environment State ────────────────────
  interface ConvertDepsStatus {
    python_found: boolean;
    python_version: string | null;
    python_path: string | null;
    venv_ready: boolean;
    script_ready: boolean;
    packages_ready: boolean;
    missing_packages: string[];
    ready: boolean;
  }

  let convertDeps = $state<ConvertDepsStatus | null>(null);
  let convertDepsLoading = $state(true);
  let convertCleaning = $state(false);
  let convertCleanError = $state<string | null>(null);

  // ── Training Environment State ────────────────────
  interface TrainingDepsStatus {
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

  let trainingDeps = $state<TrainingDepsStatus | null>(null);
  let trainingDepsLoading = $state(true);
  let trainingCleaning = $state(false);
  let trainingCleanError = $state<string | null>(null);

  async function loadGpuInfo() {
    try {
      gpu = await invoke<GpuInfo>("detect_gpu");
      selectedVariant = gpu.recommended_variant;
    } catch (e) {
      console.error("GPU detection failed:", e);
    } finally {
      gpuLoading = false;
    }
  }

  async function loadToolsStatus() {
    try {
      tools = await invoke<ToolsStatus>("get_tools_status");
    } catch (e) {
      console.error("Tools status check failed:", e);
    } finally {
      toolsLoading = false;
    }
  }

  async function handleDownload() {
    downloading = true;
    downloadError = null;
    try {
      await invoke("download_llama_cpp", { variant: selectedVariant });
      await loadToolsStatus();
    } catch (e) {
      downloadError = String(e);
    } finally {
      downloading = false;
    }
  }

  async function handleRemove() {
    removing = true;
    try {
      await invoke("remove_tools");
      await loadToolsStatus();
    } catch (e) {
      console.error("Remove failed:", e);
    } finally {
      removing = false;
    }
  }

  async function loadTrainingDeps() {
    trainingDepsLoading = true;
    try {
      trainingDeps = await invoke<TrainingDepsStatus>("training_check_deps");
    } catch (e) {
      console.error("Training deps check failed:", e);
    } finally {
      trainingDepsLoading = false;
    }
  }

  async function handleCleanTraining() {
    trainingCleaning = true;
    trainingCleanError = null;
    try {
      await invoke("training_clean_env");
      await loadTrainingDeps();
    } catch (e) {
      trainingCleanError = String(e);
    } finally {
      trainingCleaning = false;
    }
  }

  async function loadConvertDeps() {
    convertDepsLoading = true;
    try {
      convertDeps = await invoke<ConvertDepsStatus>("convert_check_deps");
    } catch (e) {
      console.error("Convert deps check failed:", e);
    } finally {
      convertDepsLoading = false;
    }
  }

  async function handleCleanConvert() {
    convertCleaning = true;
    convertCleanError = null;
    try {
      await invoke("convert_clean_env");
      await loadConvertDeps();
    } catch (e) {
      convertCleanError = String(e);
    } finally {
      convertCleaning = false;
    }
  }

  async function loadSystemInfo() {
    try {
      sysInfo = await invoke<SystemInfo>("get_system_info");
      const saved = await invoke<AppSettings>("load_settings");
      if (saved.memory_limit_mb) {
        memoryLimitMb = saved.memory_limit_mb;
      } else if (sysInfo) {
        memoryLimitMb = Math.max(200, Math.round((sysInfo.total_ram_mb * 0.5) / 200) * 200);
      }
    } catch (e) {
      console.error("System info failed:", e);
    } finally {
      sysInfoLoading = false;
    }
  }

  async function saveMemoryLimit() {
    try {
      await invoke("save_settings", { settings: { memory_limit_mb: memoryLimitMb } });
    } catch (e) {
      console.error("Settings save failed:", e);
    }
  }

  $effect(() => {
    loadGpuInfo();
    loadToolsStatus();
    loadSystemInfo();
    loadTrainingDeps();
    loadConvertDeps();
  });
</script>

<div class="settings fade-in">
  <!-- ── Header ──────────────────────────────────── -->
  <div class="settings-header panel">
    <div class="settings-header-top">
      <span class="label-xs">FRG.07</span>
      <span class="label-xs" style="color: var(--text-muted);">CONFIGURATION</span>
    </div>
    <h1 class="settings-title">SETTINGS</h1>
    <p class="settings-desc">Appearance &amp; tools configuration</p>
  </div>

  <!-- ── Mode Toggle ─────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">MODE</span>
    </div>

    <div class="mode-grid">
      {#each modes as mode}
        <button
          class="mode-card"
          class:mode-active={theme.mode === mode.id}
          onclick={() => theme.setMode(mode.id)}
        >
          <div class="mode-preview" class:mode-preview-light={mode.id === "light"}>
            <div class="mode-preview-bar"></div>
            <div class="mode-preview-body">
              <div class="mode-preview-sidebar"></div>
              <div class="mode-preview-content">
                <div class="mode-preview-line"></div>
                <div class="mode-preview-line short"></div>
              </div>
            </div>
          </div>
          <div class="mode-info">
            <span class="mode-name">{mode.label}</span>
            {#if theme.mode === mode.id}
              <span class="dot dot-active"></span>
            {/if}
          </div>
        </button>
      {/each}
    </div>
  </div>

  <!-- ── Font Family ──────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">FONT FAMILY</span>
    </div>

    <div class="font-grid">
      {#each FONT_FAMILIES as font}
        <button
          class="font-card"
          class:font-active={theme.fontFamily === font.id}
          onclick={() => theme.setFontFamily(font.id)}
        >
          <span class="font-preview" style="font-family: {font.css};">Aa 0123</span>
          <span class="font-name">{font.name}</span>
          {#if theme.fontFamily === font.id}
            <span class="dot dot-active"></span>
          {/if}
        </button>
      {/each}
    </div>
  </div>

  <!-- ── Font Size ───────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">FONT SIZE</span>
    </div>

    <div class="fontsize-row">
      {#each FONT_SIZES as size}
        <button
          class="fontsize-btn"
          class:fontsize-active={theme.fontSize === size}
          onclick={() => theme.setFontSize(size)}
        >
          {size}px
        </button>
      {/each}
      <span class="label-xs" style="margin-left: 12px; color: var(--text-muted);">
        CURRENT: {theme.fontSize}px
      </span>
    </div>
  </div>

  <!-- ── System Info ───────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">SYSTEM INFO</span>
    </div>

    <div class="tools-panel panel-flat">
      {#if sysInfoLoading}
        <div class="tools-row">
          <span class="label-xs" style="color: var(--info); animation: pulse 1.2s ease infinite;">READING SYSTEM...</span>
        </div>
      {:else if sysInfo}
        <div class="gpu-grid">
          <div class="gpu-cell" style="grid-column: 1 / -1;">
            <span class="label-xs">CPU</span>
            <span class="code">{sysInfo.cpu_name}</span>
          </div>
          <div class="gpu-cell">
            <span class="label-xs">CORES / THREADS</span>
            <span class="code">{sysInfo.cpu_cores} / {sysInfo.cpu_threads}</span>
          </div>
          <div class="gpu-cell">
            <span class="label-xs">TOTAL RAM</span>
            <span class="code">{(sysInfo.total_ram_mb / 1024).toFixed(1)} GB</span>
          </div>
          <div class="gpu-cell">
            <span class="label-xs">AVAILABLE RAM</span>
            <span class="code" style="color: var(--success);">{(sysInfo.available_ram_mb / 1024).toFixed(1)} GB</span>
          </div>
          <div class="gpu-cell">
            <span class="label-xs">USED RAM</span>
            <span class="code" style="color: var(--accent);">{(sysInfo.used_ram_mb / 1024).toFixed(1)} GB</span>
          </div>
        </div>
      {:else}
        <div class="tools-row">
          <span class="label-xs" style="color: var(--text-muted);">SYSTEM INFO UNAVAILABLE</span>
        </div>
      {/if}
    </div>
  </div>

  <!-- ── RAM Allocation ──────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">RAM ALLOCATION</span>
    </div>

    <div class="tools-panel panel">
      <div class="ram-control">
        <div class="ram-header">
          <span class="label-xs">MERGE MEMORY LIMIT</span>
          <span class="code" style="color: var(--accent);">{(memoryLimitMb / 1024).toFixed(1)} GB ({memoryLimitMb} MB)</span>
        </div>
        <input
          type="range"
          min="200"
          max={sysInfo?.total_ram_mb ?? 8000}
          step="200"
          bind:value={memoryLimitMb}
          onchange={saveMemoryLimit}
          class="ram-slider"
        />
        <div class="ram-labels">
          <span class="label-xs" style="color: var(--text-muted);">200 MB</span>
          <span class="label-xs" style="color: var(--text-muted);">
            {sysInfo ? (sysInfo.total_ram_mb / 1024).toFixed(1) + ' GB' : '--'}
          </span>
        </div>
        <p class="tools-desc" style="margin-top: 8px;">
          Controls how much RAM the merge engine can use. Streaming mode processes one tensor at a time for minimal memory usage.
        </p>
      </div>
    </div>
  </div>

  <!-- ── GPU Detection ─────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">GPU DETECTION</span>
    </div>

    <div class="tools-panel panel-flat">
      {#if gpuLoading}
        <div class="tools-row">
          <span class="label-xs" style="color: var(--info); animation: pulse 1.2s ease infinite;">SCANNING HARDWARE...</span>
        </div>
      {:else if gpu}
        <div class="gpu-grid">
          <div class="gpu-cell">
            <span class="label-xs">PLATFORM</span>
            <span class="code">{gpu.os.toUpperCase()} / {gpu.arch.toUpperCase()}</span>
          </div>
          <div class="gpu-cell">
            <span class="label-xs">NVIDIA</span>
            {#if gpu.has_nvidia}
              <span class="code" style="color: var(--success);">{gpu.nvidia_name ?? "DETECTED"}</span>
            {:else}
              <span class="code" style="color: var(--text-muted);">NOT FOUND</span>
            {/if}
          </div>
          {#if gpu.has_nvidia}
            <div class="gpu-cell">
              <span class="label-xs">VRAM</span>
              <span class="code">{gpu.nvidia_vram ?? "---"}</span>
            </div>
            <div class="gpu-cell">
              <span class="label-xs">CUDA</span>
              <span class="code">{gpu.cuda_version ?? "---"}</span>
            </div>
          {/if}
          <div class="gpu-cell">
            <span class="label-xs">VULKAN</span>
            <span class="code" style="color: {gpu.has_vulkan ? 'var(--success)' : 'var(--text-muted)'};">
              {gpu.has_vulkan ? "AVAILABLE" : "NOT FOUND"}
            </span>
          </div>
          {#if gpu.has_metal}
            <div class="gpu-cell">
              <span class="label-xs">METAL</span>
              <span class="code" style="color: var(--success);">AVAILABLE</span>
            </div>
          {/if}
          <div class="gpu-cell">
            <span class="label-xs">RECOMMENDED</span>
            <span class="code" style="color: var(--accent);">{gpu.recommended_variant.toUpperCase()}</span>
          </div>
        </div>
      {:else}
        <div class="tools-row">
          <span class="label-xs" style="color: var(--text-muted);">GPU DETECTION UNAVAILABLE</span>
        </div>
      {/if}
    </div>
  </div>

  <!-- ── llama.cpp Tools ───────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">LLAMA.CPP TOOLS</span>
    </div>

    <div class="tools-panel panel">
      {#if toolsLoading}
        <div class="tools-row">
          <span class="label-xs" style="color: var(--info); animation: pulse 1.2s ease infinite;">CHECKING...</span>
        </div>
      {:else if tools?.installed}
        <!-- Installed state -->
        <div class="tools-status-row">
          <span class="dot dot-success"></span>
          <span class="heading-sm" style="color: var(--success);">INSTALLED</span>
        </div>
        <div class="gpu-grid" style="margin-top: 8px;">
          {#if tools.version}
            <div class="gpu-cell">
              <span class="label-xs">VERSION</span>
              <span class="code">{tools.version}</span>
            </div>
          {/if}
          {#if tools.variant}
            <div class="gpu-cell">
              <span class="label-xs">BUILD</span>
              <span class="code">{tools.variant.toUpperCase()}</span>
            </div>
          {/if}
          {#if tools.path}
            <div class="gpu-cell" style="grid-column: 1 / -1;">
              <span class="label-xs">PATH</span>
              <span class="code" style="font-size: 9px; word-break: break-all; color: var(--text-muted);">{tools.path}</span>
            </div>
          {/if}
        </div>
        <div class="tools-actions" style="margin-top: 12px;">
          <button class="btn btn-accent" onclick={handleDownload} disabled={downloading}>
            {downloading ? "REINSTALLING..." : "REINSTALL"}
          </button>
          <button class="btn btn-danger" onclick={handleRemove} disabled={removing}>
            {removing ? "REMOVING..." : "REMOVE"}
          </button>
        </div>
      {:else}
        <!-- Not installed state -->
        <div class="tools-status-row">
          <span class="dot dot-paused"></span>
          <span class="heading-sm" style="color: var(--text-muted);">NOT INSTALLED</span>
        </div>
        <p class="tools-desc">
          Download llama.cpp tools for model quantization. The correct build
          for your platform will be fetched from the latest GitHub release.
        </p>

        <!-- Variant selector -->
        <div class="variant-grid">
          {#each VARIANTS as v}
            <button
              class="variant-card"
              class:variant-active={selectedVariant === v.id}
              onclick={() => selectedVariant = v.id}
            >
              <div class="variant-header">
                <span class="variant-name">{v.label}</span>
                {#if selectedVariant === v.id}
                  <span class="dot dot-active" style="margin-left: auto;"></span>
                {/if}
                {#if gpu?.recommended_variant === v.id}
                  <span class="badge badge-accent" style="margin-left: auto;">REC</span>
                {/if}
              </div>
              <span class="variant-desc">{v.desc}</span>
              {#if v.id === "cuda" && gpu?.os !== "windows"}
                <span class="variant-note">No prebuilt CUDA release — Vulkan build will be used</span>
              {/if}
            </button>
          {/each}
        </div>

        <div class="tools-actions">
          <button class="btn btn-accent" onclick={handleDownload} disabled={downloading}>
            {#if downloading}
              <span style="animation: pulse 1.2s ease infinite;">DOWNLOADING...</span>
            {:else}
              DOWNLOAD &amp; INSTALL
            {/if}
          </button>
        </div>

        {#if downloadError}
          <div class="tools-error panel-flat" style="border-color: var(--danger);">
            <span class="dot dot-danger"></span>
            <span class="danger-text" style="flex: 1;">{downloadError}</span>
          </div>
        {/if}
      {/if}
    </div>
  </div>

  <!-- ── Training Environment ─────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">TRAINING ENVIRONMENT</span>
    </div>

    <div class="tools-panel panel">
      {#if trainingDepsLoading}
        <div class="tools-row">
          <span class="label-xs" style="color: var(--info); animation: pulse 1.2s ease infinite;">CHECKING...</span>
        </div>
      {:else if trainingDeps?.ready}
        <div class="tools-status-row">
          <span class="dot dot-success"></span>
          <span class="heading-sm" style="color: var(--success);">INSTALLED</span>
        </div>
        <div class="gpu-grid" style="margin-top: 8px;">
          {#if trainingDeps.python_version}
            <div class="gpu-cell">
              <span class="label-xs">PYTHON</span>
              <span class="code">{trainingDeps.python_version}</span>
            </div>
          {/if}
          {#if trainingDeps.torch_version}
            <div class="gpu-cell">
              <span class="label-xs">TORCH</span>
              <span class="code">{trainingDeps.torch_version}</span>
            </div>
          {/if}
          <div class="gpu-cell">
            <span class="label-xs">CUDA</span>
            <span class="code" style="color: {trainingDeps.cuda_available ? 'var(--success)' : 'var(--text-muted)'};">
              {trainingDeps.cuda_available ? trainingDeps.cuda_version ?? 'YES' : 'N/A'}
            </span>
          </div>
          <div class="gpu-cell">
            <span class="label-xs">PACKAGES</span>
            <span class="code" style="color: var(--success);">READY</span>
          </div>
        </div>
        <div class="tools-actions" style="margin-top: 12px;">
          <button class="btn btn-danger" onclick={handleCleanTraining} disabled={trainingCleaning}>
            {trainingCleaning ? "REMOVING..." : "CLEAN / DELETE"}
          </button>
        </div>
        <p class="tools-desc" style="margin-top: 4px;">
          Removes the training Python environment (venv, torch, transformers, PEFT, etc). You can reinstall from the Training page.
        </p>
      {:else if trainingDeps?.venv_ready}
        <div class="tools-status-row">
          <span class="dot dot-active"></span>
          <span class="heading-sm" style="color: var(--accent);">PARTIAL</span>
        </div>
        <p class="tools-desc">
          Training venv exists but some packages are missing: {trainingDeps.missing_packages.join(', ') || 'unknown'}
        </p>
        <div class="tools-actions" style="margin-top: 8px;">
          <button class="btn btn-danger" onclick={handleCleanTraining} disabled={trainingCleaning}>
            {trainingCleaning ? "REMOVING..." : "CLEAN / DELETE"}
          </button>
        </div>
      {:else}
        <div class="tools-status-row">
          <span class="dot dot-paused"></span>
          <span class="heading-sm" style="color: var(--text-muted);">NOT INSTALLED</span>
        </div>
        <p class="tools-desc">
          No training environment found. Install from the Training page.
        </p>
      {/if}

      {#if trainingCleanError}
        <div class="tools-error panel-flat" style="border-color: var(--danger);">
          <span class="dot dot-danger"></span>
          <span class="danger-text" style="flex: 1;">{trainingCleanError}</span>
        </div>
      {/if}
    </div>
  </div>

  <!-- ── Convert Environment ─────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">CONVERT ENVIRONMENT</span>
    </div>

    <div class="tools-panel panel">
      {#if convertDepsLoading}
        <div class="tools-row">
          <span class="label-xs" style="color: var(--info); animation: pulse 1.2s ease infinite;">CHECKING...</span>
        </div>
      {:else if convertDeps?.ready}
        <div class="tools-status-row">
          <span class="dot dot-success"></span>
          <span class="heading-sm" style="color: var(--success);">INSTALLED</span>
        </div>
        <div class="gpu-grid" style="margin-top: 8px;">
          {#if convertDeps.python_version}
            <div class="gpu-cell">
              <span class="label-xs">PYTHON</span>
              <span class="code">{convertDeps.python_version}</span>
            </div>
          {/if}
          <div class="gpu-cell">
            <span class="label-xs">VENV</span>
            <span class="code" style="color: var(--success);">READY</span>
          </div>
          <div class="gpu-cell">
            <span class="label-xs">SCRIPT</span>
            <span class="code" style="color: {convertDeps.script_ready ? 'var(--success)' : 'var(--text-muted)'};">
              {convertDeps.script_ready ? "READY" : "MISSING"}
            </span>
          </div>
          <div class="gpu-cell">
            <span class="label-xs">PACKAGES</span>
            <span class="code" style="color: var(--success);">READY</span>
          </div>
        </div>
        <div class="tools-actions" style="margin-top: 12px;">
          <button class="btn btn-danger" onclick={handleCleanConvert} disabled={convertCleaning}>
            {convertCleaning ? "REMOVING..." : "CLEAN / DELETE"}
          </button>
        </div>
        <p class="tools-desc" style="margin-top: 4px;">
          Removes the convert Python environment (venv, gguf, numpy, convert script). You can reinstall from the Convert page.
        </p>
      {:else if convertDeps?.venv_ready}
        <div class="tools-status-row">
          <span class="dot dot-active"></span>
          <span class="heading-sm" style="color: var(--accent);">PARTIAL</span>
        </div>
        <p class="tools-desc">
          Convert venv exists but some packages are missing: {convertDeps.missing_packages.join(', ') || 'unknown'}
        </p>
        <div class="tools-actions" style="margin-top: 8px;">
          <button class="btn btn-danger" onclick={handleCleanConvert} disabled={convertCleaning}>
            {convertCleaning ? "REMOVING..." : "CLEAN / DELETE"}
          </button>
        </div>
      {:else}
        <div class="tools-status-row">
          <span class="dot dot-paused"></span>
          <span class="heading-sm" style="color: var(--text-muted);">NOT INSTALLED</span>
        </div>
        <p class="tools-desc">
          No convert environment found. Install from the Convert page.
        </p>
      {/if}

      {#if convertCleanError}
        <div class="tools-error panel-flat" style="border-color: var(--danger);">
          <span class="dot dot-danger"></span>
          <span class="danger-text" style="flex: 1;">{convertCleanError}</span>
        </div>
      {/if}
    </div>
  </div>

  <!-- ── Color Semantics ─────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">COLOR SEMANTICS</span>
    </div>

    <div class="semantics-grid">
      <div class="semantic-card panel-flat">
        <div class="semantic-dot" style="background: var(--accent);"></div>
        <div class="semantic-info">
          <span class="semantic-name">AMBER</span>
          <span class="semantic-role">Idle / Default</span>
        </div>
      </div>
      <div class="semantic-card panel-flat">
        <div class="semantic-dot" style="background: var(--info);"></div>
        <div class="semantic-info">
          <span class="semantic-name">BLUE</span>
          <span class="semantic-role">Working / Processing</span>
        </div>
      </div>
      <div class="semantic-card panel-flat">
        <div class="semantic-dot" style="background: var(--success);"></div>
        <div class="semantic-info">
          <span class="semantic-name">GREEN</span>
          <span class="semantic-role">Success / Complete</span>
        </div>
      </div>
      <div class="semantic-card panel-flat">
        <div class="semantic-dot" style="background: var(--danger);"></div>
        <div class="semantic-info">
          <span class="semantic-name">RED</span>
          <span class="semantic-role">Danger / Error</span>
        </div>
      </div>
      <div class="semantic-card panel-flat">
        <div class="semantic-dot" style="background: var(--gray);"></div>
        <div class="semantic-info">
          <span class="semantic-name">GRAY</span>
          <span class="semantic-role">Paused / Inactive</span>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Live Preview ────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">PREVIEW</span>
    </div>

    <div class="preview panel">
      <div class="preview-header">
        <span class="label-xs">LIVE PREVIEW</span>
        <span class="badge badge-accent">
          <span class="dot dot-active"></span>
          AMBER
        </span>
      </div>

      <div class="preview-body">
        <div class="preview-row">
          <span class="heading-sm" style="color: var(--accent);">ACCENT TEXT</span>
          <span class="label" style="color: var(--text-primary);">Primary text</span>
          <span class="label" style="color: var(--text-secondary);">Secondary</span>
          <span class="label" style="color: var(--text-muted);">Muted</span>
        </div>

        <div class="preview-row">
          <button class="btn btn-accent">ACTION</button>
          <button class="btn">DEFAULT</button>
          <span class="badge badge-accent">
            <span class="dot dot-active"></span>
            BADGE
          </span>
          <span class="badge badge-dim">INACTIVE</span>
        </div>

        <div class="preview-row">
          <span class="badge badge-info"><span class="dot dot-working"></span> WORKING</span>
          <span class="badge badge-success"><span class="dot dot-success"></span> SUCCESS</span>
          <span class="badge badge-danger"><span class="dot dot-danger"></span> ERROR</span>
          <span class="badge badge-dim"><span class="dot dot-paused"></span> PAUSED</span>
        </div>

        <div class="preview-bar-row">
          <div class="preview-bar" style="background: var(--accent); width: 40%;"></div>
          <div class="preview-bar" style="background: var(--info); width: 25%;"></div>
          <div class="preview-bar" style="background: var(--success); width: 20%;"></div>
          <div class="preview-bar" style="background: var(--danger); width: 10%;"></div>
          <div class="preview-bar" style="background: var(--gray); width: 5%;"></div>
        </div>
      </div>

      <div class="preview-footer">
        <div class="barcode" style="max-width: 140px;"></div>
        <span class="label-xs">
          {theme.mode.toUpperCase()} / AMBER / #F59E0B
        </span>
      </div>
    </div>
  </div>
</div>

<style>
  .settings {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* ── Header ────────────────────────────────────── */
  .settings-header {
    padding: 16px;
  }

  .settings-header-top {
    display: flex;
    gap: 12px;
    margin-bottom: 12px;
  }

  .settings-title {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 0.18em;
    line-height: 1;
    color: var(--text-primary);
  }

  .settings-desc {
    font-size: 10px;
    color: var(--text-secondary);
    letter-spacing: 0.06em;
    margin-top: 6px;
    text-transform: uppercase;
  }

  /* ── Section ───────────────────────────────────── */
  .section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-label {
    padding: 0;
  }

  /* ── Mode Grid ─────────────────────────────────── */
  .mode-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .mode-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    cursor: pointer;
    transition:
      border-color var(--theme-transition),
      background-color var(--theme-transition);
    font-family: var(--font-mono);
  }

  .mode-card:hover {
    border-color: var(--border-strong);
  }

  .mode-active {
    border-color: var(--accent);
  }

  .mode-active:hover {
    border-color: var(--accent);
  }

  /* Mini app preview */
  .mode-preview {
    height: 64px;
    background: #0b0b0b;
    border: 1px solid #2a2a2a;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: background-color var(--theme-transition);
  }

  .mode-preview-light {
    background: #edecea;
    border-color: #c8c8c0;
  }

  .mode-preview-bar {
    height: 8px;
    background: #181818;
    border-bottom: 1px solid #2a2a2a;
  }

  .mode-preview-light .mode-preview-bar {
    background: #f5f5f2;
    border-color: #c8c8c0;
  }

  .mode-preview-body {
    display: flex;
    flex: 1;
  }

  .mode-preview-sidebar {
    width: 24px;
    background: #111111;
    border-right: 1px solid #2a2a2a;
  }

  .mode-preview-light .mode-preview-sidebar {
    background: #f5f5f2;
    border-color: #c8c8c0;
  }

  .mode-preview-content {
    flex: 1;
    padding: 6px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .mode-preview-line {
    height: 3px;
    background: #2a2a2a;
    width: 80%;
  }

  .mode-preview-light .mode-preview-line {
    background: #c8c8c0;
  }

  .mode-preview-line.short {
    width: 50%;
  }

  .mode-info {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .mode-name {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-secondary);
    transition: color var(--theme-transition);
  }

  .mode-active .mode-name {
    color: var(--accent);
  }

  /* ── Font Settings ────────────────────────────────── */
  .font-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
  }

  .font-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    padding: 12px 8px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    cursor: pointer;
    font-family: var(--font-mono);
    transition: all 120ms ease;
  }

  .font-card:hover {
    border-color: var(--border-strong);
  }

  .font-active {
    border-color: var(--accent);
  }

  .font-preview {
    font-size: 16px;
    font-weight: 500;
    color: var(--text-primary);
    letter-spacing: 0.02em;
  }

  .font-name {
    font-size: 8px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .font-active .font-name {
    color: var(--accent);
  }

  .fontsize-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .fontsize-btn {
    padding: 6px 14px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    cursor: pointer;
    transition: all 120ms ease;
  }

  .fontsize-btn:hover {
    border-color: var(--border-strong);
    color: var(--text-primary);
  }

  .fontsize-active {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-bg);
  }

  /* ── GPU & Tools ─────────────────────────────────── */
  .tools-panel {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .tools-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .tools-status-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .tools-desc {
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    line-height: 1.6;
    margin: 4px 0;
  }

  .tools-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .tools-error {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 10px;
    margin-top: 4px;
  }

  .gpu-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 1px;
    background: var(--border-dim);
  }

  .gpu-cell {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 8px;
    background: var(--bg-surface);
  }

  /* ── Variant Selector ────────────────────────────── */
  .variant-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin: 4px 0;
  }

  .variant-card {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
    cursor: pointer;
    font-family: var(--font-mono);
    text-align: left;
    transition: all 120ms ease;
  }

  .variant-card:hover {
    border-color: var(--border-strong);
    background: var(--bg-hover);
  }

  .variant-active {
    border-color: var(--accent) !important;
    background: var(--accent-bg) !important;
  }

  .variant-header {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .variant-name {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--text-primary);
  }

  .variant-desc {
    font-size: 8px;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .variant-note {
    font-size: 7px;
    color: var(--info);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 2px;
  }

  /* ── RAM Slider ─────────────────────────────────── */
  .ram-control {
    display: flex;
    flex-direction: column;
  }

  .ram-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .ram-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 2px;
    background: var(--border);
    outline: none;
    margin: 12px 0 8px;
    cursor: pointer;
  }

  .ram-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    background: var(--accent);
    cursor: pointer;
    border: none;
  }

  .ram-slider::-moz-range-thumb {
    width: 12px;
    height: 12px;
    background: var(--accent);
    cursor: pointer;
    border: none;
    border-radius: 0;
  }

  .ram-labels {
    display: flex;
    justify-content: space-between;
  }

  /* ── Color Semantics ────────────────────────────── */
  .semantics-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
  }

  .semantic-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 12px 8px;
    text-align: center;
  }

  .semantic-dot {
    width: 16px;
    height: 16px;
    transition: background-color var(--theme-transition);
  }

  .semantic-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .semantic-name {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-primary);
  }

  .semantic-role {
    font-size: 8px;
    font-weight: 500;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  /* ── Preview ───────────────────────────────────── */
  .preview {
    padding: 16px;
  }

  .preview-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .preview-body {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .preview-row {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }

  .preview-bar-row {
    display: flex;
    gap: 2px;
    height: 6px;
  }

  .preview-bar {
    height: 100%;
    transition: background-color var(--theme-transition);
  }

  .preview-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border-dim);
  }
</style>
