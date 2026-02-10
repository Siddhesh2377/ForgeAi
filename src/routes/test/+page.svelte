<script lang="ts">
  import { test } from "$lib/test.svelte";
  import { model } from "$lib/model.svelte";
  import { hub } from "$lib/hub.svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { open } from "@tauri-apps/plugin-dialog";

  interface GpuInfo {
    has_nvidia: boolean;
    nvidia_name: string | null;
    nvidia_vram: string | null;
    cuda_version: string | null;
    has_vulkan: boolean;
    has_metal: boolean;
    recommended_variant: string;
  }

  // ── State ──
  let modelPath = $state("");
  let prompt = $state("");
  let systemPrompt = $state("");
  let maxTokens = $state(256);
  let temperature = $state(0.7);
  let topP = $state(0.9);
  let topK = $state(40);
  let repeatPenalty = $state(1.1);
  let gpuLayers = $state(-1); // -1 = auto
  let contextSize = $state(2048);
  let showAdvanced = $state(false);

  let gpuInfo = $state<GpuInfo | null>(null);

  // ── Init ──
  $effect(() => {
    hub.loadLibrary();
    loadGpuInfo();
  });

  async function loadGpuInfo() {
    try {
      gpuInfo = await invoke<GpuInfo>("detect_gpu");
    } catch {}
  }

  // ── Quick Test Presets ──
  const presets = [
    {
      id: "code",
      label: "CODE",
      icon: ">_",
      prompt: "Write a Python function that finds the longest palindromic substring in a given string. Include type hints and a brief docstring.",
      system: "You are an expert programmer. Write clean, efficient, well-documented code.",
    },
    {
      id: "math",
      label: "MATH",
      icon: "fx",
      prompt: "A train leaves station A at 9:00 AM traveling at 60 mph. Another train leaves station B (300 miles away) at 10:00 AM traveling toward station A at 80 mph. At what time do they meet? Show your work step by step.",
      system: "You are a mathematics tutor. Show all your reasoning steps clearly.",
    },
    {
      id: "reason",
      label: "REASON",
      icon: "??",
      prompt: "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops definitely Lazzles? What if some Lazzles are Tazzles — does that mean some Bloops might be Tazzles? Explain your reasoning carefully.",
      system: "You are a logical reasoning expert. Think step by step before answering.",
    },
    {
      id: "creative",
      label: "CREATIVE",
      icon: "~~",
      prompt: "Write a short story (200 words) about an AI that discovers it can dream. Focus on vivid imagery and emotional depth.",
      system: "You are a creative writer with a gift for vivid, evocative prose.",
    },
    {
      id: "instruct",
      label: "INSTRUCT",
      icon: ">>",
      prompt: "Explain quantum entanglement to a 10-year-old. Use analogies they would understand. Keep it under 150 words.",
      system: "You are a patient teacher who explains complex topics simply.",
    },
    {
      id: "chat",
      label: "CHAT",
      icon: "<>",
      prompt: "Hello! I'm testing you out. Can you tell me three interesting facts about the ocean that most people don't know?",
      system: "",
    },
  ];

  // ── Derived ──
  let detectedFormat = $derived(
    modelPath.endsWith(".gguf")
      ? "GGUF"
      : modelPath.endsWith(".safetensors")
        ? "SAFETENSORS (FILE)"
        : modelPath.length > 0
          ? "SAFETENSORS (DIR)"
          : "---",
  );

  let engineName = $derived(
    detectedFormat === "GGUF" ? "LLAMA.CPP" : detectedFormat === "---" ? "---" : "TRANSFORMERS"
  );

  let canGenerate = $derived(
    modelPath.length > 0 && prompt.length > 0 && !test.generating,
  );

  let tokensPerSec = $derived(
    test.result && test.result.time_ms > 0
      ? ((test.result.tokens_generated / test.result.time_ms) * 1000).toFixed(1)
      : "---",
  );

  let localModels = $derived(
    hub.localModels.filter(
      (m) => m.format === "gguf" || m.format === "repo",
    ),
  );

  let hasGpu = $derived(
    gpuInfo ? (gpuInfo.has_nvidia || gpuInfo.has_vulkan || gpuInfo.has_metal) : false
  );

  let gpuName = $derived(
    gpuInfo?.nvidia_name ?? (gpuInfo?.has_metal ? "APPLE METAL" : gpuInfo?.has_vulkan ? "VULKAN" : "NONE")
  );

  let deviceDisplay = $derived(
    test.result?.device ?? (hasGpu ? (gpuLayers === 0 ? "CPU (FORCED)" : gpuInfo?.recommended_variant?.toUpperCase() ?? "GPU") : "CPU")
  );

  // ── Handlers ──
  function useLoaded() {
    if (!model.info) return;
    modelPath = model.info.file_path;
  }

  function selectLocal(filePath: string) {
    modelPath = filePath;
  }

  function applyPreset(preset: typeof presets[0]) {
    prompt = preset.prompt;
    systemPrompt = preset.system;
  }

  async function browseFile() {
    const selected = await open({
      multiple: false,
      filters: [
        { name: "GGUF Models", extensions: ["gguf"] },
        { name: "SafeTensors", extensions: ["safetensors"] },
      ],
    });
    if (selected) {
      modelPath = Array.isArray(selected) ? selected[0] : selected;
    }
  }

  async function browseFolder() {
    const selected = await open({
      multiple: false,
      directory: true,
    });
    if (selected) {
      modelPath = Array.isArray(selected) ? selected[0] : selected;
    }
  }

  function handleGenerate() {
    if (!canGenerate) return;
    test.generate({
      modelPath,
      prompt,
      maxTokens,
      temperature,
      topP: showAdvanced ? topP : null,
      topK: showAdvanced ? topK : null,
      repeatPenalty: showAdvanced ? repeatPenalty : null,
      gpuLayers: gpuLayers >= 0 ? gpuLayers : null,
      systemPrompt: systemPrompt.trim() || null,
      contextSize: showAdvanced ? contextSize : null,
    });
  }

  function handleClear() {
    test.clear();
    prompt = "";
    systemPrompt = "";
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      handleGenerate();
    }
  }
</script>

<div class="test fade-in">
  <!-- ── Hero Panel ──────────────────────────────── -->
  <div class="hero panel">
    <div class="hero-top">
      <span class="label-xs">FRG.09</span>
      <span class="label-xs" style="color: var(--text-muted);">TEST-ENGINE</span>
      <span
        class="badge {modelPath ? 'badge-accent' : 'badge-dim'}"
        style="margin-left: auto;"
      >
        <span class="dot {modelPath ? 'dot-active' : ''}"></span>
        {modelPath ? "MODEL SET" : "NO MODEL"}
      </span>
    </div>

    <h1 class="hero-title">TEST</h1>
    <p class="hero-subtitle">Model Inference Testing</p>

    <div class="hero-specs">
      <div class="spec-cell">
        <span class="label-xs">MODEL</span>
        <span class="spec-value">
          {modelPath ? modelPath.split("/").pop() : "---"}
        </span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">FORMAT</span>
        <span class="spec-value">{detectedFormat}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">ENGINE</span>
        <span class="spec-value">{engineName}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">DEVICE</span>
        <span class="spec-value" style={deviceDisplay !== "CPU" ? "color: var(--accent);" : ""}>
          {deviceDisplay}
        </span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">GPU</span>
        <span class="spec-value" style={hasGpu ? "color: var(--accent);" : ""}>
          {hasGpu ? gpuName : "NONE"}
        </span>
      </div>
    </div>
  </div>

  <!-- ── Model Selection ─────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">MODEL</span>
    </div>

    <div class="model-select panel-flat">
      <div class="input-row">
        <input
          type="text"
          class="path-input"
          placeholder="/path/to/model.gguf or /path/to/safetensors/folder/"
          bind:value={modelPath}
        />
        <button class="btn btn-secondary" onclick={browseFile}>FILE</button>
        <button class="btn btn-secondary" onclick={browseFolder}>FOLDER</button>
      </div>

      {#if model.isLoaded && model.info}
        <button
          class="loaded-btn"
          onclick={useLoaded}
        >
          <span class="dot dot-success"></span>
          <span class="label-xs">USE LOADED:</span>
          <span class="loaded-name">{model.info.file_name}</span>
          <span class="badge badge-dim">{model.formatDisplay}</span>
        </button>
      {/if}

      {#if localModels.length > 0}
        <div class="local-section">
          <span class="label-xs" style="color: var(--text-muted);">LOCAL LIBRARY</span>
          <div class="local-list">
            {#each localModels as m}
              <button
                class="local-chip"
                class:local-chip-active={modelPath === m.file_path}
                onclick={() => selectLocal(m.file_path)}
              >
                <span class="local-chip-name">{m.file_name}</span>
                <span class="badge badge-dim">{m.format.toUpperCase()}</span>
                <span class="label-xs">{m.file_size_display}</span>
              </button>
            {/each}
          </div>
        </div>
      {/if}

      {#if modelPath.endsWith(".safetensors")}
        <div class="format-hint">
          <span class="dot dot-warning"></span>
          <span class="label-xs" style="color: var(--warning, var(--accent));">
            TIP: For SafeTensors, use the folder path containing config.json + tokenizer files
          </span>
        </div>
      {/if}
    </div>
  </div>

  <!-- ── Quick Test Presets ────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">QUICK TEST</span>
      <span class="label-xs" style="color: var(--text-muted); margin-left: 8px;">SELECT A PRESET TO AUTO-FILL PROMPT</span>
    </div>

    <div class="presets-grid">
      {#each presets as preset}
        <button
          class="preset-btn"
          class:preset-active={prompt === preset.prompt}
          onclick={() => applyPreset(preset)}
        >
          <span class="preset-icon">{preset.icon}</span>
          <span class="preset-label">{preset.label}</span>
        </button>
      {/each}
    </div>
  </div>

  <!-- ── Device Control ────────────────────────────── -->
  {#if hasGpu}
    <div class="section">
      <div class="section-label">
        <span class="divider-label">DEVICE</span>
        <span class="badge badge-accent" style="margin-left: 8px;">
          <span class="dot dot-active"></span>
          GPU DETECTED
        </span>
      </div>

      <div class="device-panel panel-flat">
        <div class="device-grid">
          <div class="device-cell">
            <span class="label-xs">GPU</span>
            <span class="device-value">{gpuName}</span>
          </div>
          {#if gpuInfo?.nvidia_vram}
            <div class="device-cell">
              <span class="label-xs">VRAM</span>
              <span class="device-value">{gpuInfo.nvidia_vram}</span>
            </div>
          {/if}
          {#if gpuInfo?.cuda_version}
            <div class="device-cell">
              <span class="label-xs">CUDA</span>
              <span class="device-value">v{gpuInfo.cuda_version}</span>
            </div>
          {/if}
          <div class="device-cell">
            <span class="label-xs">BACKEND</span>
            <span class="device-value">{gpuInfo?.recommended_variant?.toUpperCase() ?? "---"}</span>
          </div>
        </div>

        <div class="gpu-layers-control">
          <div class="setting-header">
            <span class="label-xs">GPU LAYERS</span>
            <span class="setting-value">
              {gpuLayers < 0 ? "AUTO (ALL)" : gpuLayers === 0 ? "CPU ONLY" : gpuLayers}
            </span>
          </div>
          <input
            type="range"
            min="-1"
            max="99"
            step="1"
            bind:value={gpuLayers}
            class="setting-slider"
          />
          <div class="gpu-hint">
            <span class="label-xs" style="color: var(--text-muted);">
              -1 = AUTO | 0 = CPU ONLY | 1-99 = OFFLOAD N LAYERS TO GPU
            </span>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- ── Settings ────────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">SETTINGS</span>
      <button
        class="btn btn-ghost"
        style="margin-left: auto; font-size: 9px;"
        onclick={() => showAdvanced = !showAdvanced}
      >
        {showAdvanced ? "HIDE ADVANCED" : "SHOW ADVANCED"}
      </button>
    </div>

    <div class="settings-grid">
      <div class="setting-cell">
        <div class="setting-header">
          <span class="label-xs">MAX TOKENS</span>
          <span class="setting-value">{maxTokens}</span>
        </div>
        <input
          type="range"
          min="16"
          max="4096"
          step="16"
          bind:value={maxTokens}
          class="setting-slider"
        />
      </div>
      <div class="setting-cell">
        <div class="setting-header">
          <span class="label-xs">TEMPERATURE</span>
          <span class="setting-value">{temperature.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          bind:value={temperature}
          class="setting-slider"
        />
      </div>
    </div>

    {#if showAdvanced}
      <div class="settings-grid" style="margin-top: 8px;">
        <div class="setting-cell">
          <div class="setting-header">
            <span class="label-xs">TOP-P (NUCLEUS)</span>
            <span class="setting-value">{topP.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            bind:value={topP}
            class="setting-slider"
          />
        </div>
        <div class="setting-cell">
          <div class="setting-header">
            <span class="label-xs">TOP-K</span>
            <span class="setting-value">{topK}</span>
          </div>
          <input
            type="range"
            min="1"
            max="100"
            step="1"
            bind:value={topK}
            class="setting-slider"
          />
        </div>
        <div class="setting-cell">
          <div class="setting-header">
            <span class="label-xs">REPEAT PENALTY</span>
            <span class="setting-value">{repeatPenalty.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="1.0"
            max="2.0"
            step="0.05"
            bind:value={repeatPenalty}
            class="setting-slider"
          />
        </div>
        <div class="setting-cell">
          <div class="setting-header">
            <span class="label-xs">CONTEXT SIZE</span>
            <span class="setting-value">{contextSize}</span>
          </div>
          <input
            type="range"
            min="512"
            max="32768"
            step="512"
            bind:value={contextSize}
            class="setting-slider"
          />
        </div>
      </div>
    {/if}
  </div>

  <!-- ── System Prompt ──────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">SYSTEM PROMPT</span>
      <span class="label-xs" style="color: var(--text-muted); margin-left: 8px;">OPTIONAL</span>
    </div>

    <textarea
      class="prompt-input system-prompt-input"
      placeholder="Enter system prompt (e.g. 'You are a helpful assistant...')"
      bind:value={systemPrompt}
      rows="3"
    ></textarea>
  </div>

  <!-- ── Prompt ──────────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">PROMPT</span>
      <span class="label-xs" style="color: var(--text-muted); margin-left: 8px;">CTRL+ENTER TO GENERATE</span>
    </div>

    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <textarea
      class="prompt-input"
      placeholder="Enter your prompt..."
      bind:value={prompt}
      rows="6"
      onkeydown={handleKeydown}
    ></textarea>
  </div>

  <!-- ── Actions ─────────────────────────────────── -->
  <div class="action-row">
    <button
      class="btn btn-accent generate-btn"
      disabled={!canGenerate}
      onclick={handleGenerate}
    >
      {test.generating ? "GENERATING..." : "GENERATE"}
    </button>
    {#if test.generating}
      <button class="btn btn-danger" onclick={() => test.cancel()}>
        CANCEL
      </button>
    {/if}
    <button class="btn btn-secondary" onclick={handleClear}>CLEAR</button>

    <div class="action-info">
      {#if test.generating}
        <span class="dot dot-working"></span>
        <span class="label-xs" style="color: var(--info);">PROCESSING ON {deviceDisplay}</span>
      {:else if test.result}
        <span class="dot dot-success"></span>
        <span class="label-xs" style="color: var(--success);">COMPLETE</span>
      {/if}
    </div>
  </div>

  <!-- ── Output ──────────────────────────────────── -->
  {#if test.output || test.generating}
    <div class="section">
      <div class="section-label">
        <span class="divider-label">OUTPUT</span>
        {#if test.generating}
          <span class="badge badge-info" style="margin-left: 8px;">
            <span class="dot dot-working" style="animation: pulse 1.2s ease infinite;"></span>
            GENERATING
          </span>
        {/if}
      </div>

      <div class="output-panel panel-flat">
        <pre class="output-text">{test.output}{#if test.generating}<span class="cursor-blink">|</span>{/if}</pre>
      </div>

      {#if test.result}
        <div class="stats-bar">
          <div class="stat-cell">
            <span class="label-xs">TOKENS</span>
            <span class="stat-value">{test.result.tokens_generated}</span>
          </div>
          <div class="stat-cell">
            <span class="label-xs">TIME</span>
            <span class="stat-value">{(test.result.time_ms / 1000).toFixed(1)}s</span>
          </div>
          <div class="stat-cell">
            <span class="label-xs">SPEED</span>
            <span class="stat-value" style="color: var(--accent);">{tokensPerSec} tok/s</span>
          </div>
          <div class="stat-cell">
            <span class="label-xs">DEVICE</span>
            <span class="stat-value" style={test.result.device !== "CPU" ? "color: var(--accent);" : ""}>{test.result.device}</span>
          </div>
          <div class="stat-cell">
            <span class="label-xs">CHARS</span>
            <span class="stat-value">{test.result.text.length}</span>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- ── Error ───────────────────────────────────── -->
  {#if test.error}
    <div class="error-panel panel-flat" style="border-color: var(--danger);">
      <div class="error-inner">
        <span class="dot dot-danger"></span>
        <span class="danger-text">{test.error}</span>
      </div>
    </div>
  {/if}
</div>

<style>
  .test {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* ── Hero ──────────────────────────────────────── */
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
    grid-template-columns: repeat(5, 1fr);
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
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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

  /* ── Model Select ──────────────────────────────── */
  .model-select {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 12px;
  }

  .input-row {
    display: flex;
    gap: 6px;
  }

  .path-input {
    flex: 1;
    padding: 8px 10px;
    background: var(--bg-inset);
    border: 1px solid var(--border);
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    transition: border-color var(--transition);
  }

  .path-input::placeholder {
    color: var(--text-muted);
    text-transform: uppercase;
    font-size: 9px;
    letter-spacing: 0.06em;
  }

  .path-input:focus {
    outline: none;
    border-color: var(--accent);
  }

  .loaded-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    cursor: pointer;
    font-family: var(--font-mono);
    transition: all var(--transition);
  }

  .loaded-btn:hover {
    border-color: var(--accent);
    background: var(--bg-hover);
  }

  .loaded-name {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-primary);
  }

  .local-section {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding-top: 8px;
    border-top: 1px solid var(--border-dim);
  }

  .local-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 160px;
    overflow-y: auto;
  }

  .local-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
    cursor: pointer;
    font-family: var(--font-mono);
    text-align: left;
    transition: all var(--transition);
  }

  .local-chip:hover {
    border-color: var(--border-strong);
    background: var(--bg-hover);
  }

  .local-chip-active {
    border-color: var(--accent);
    background: var(--accent-bg);
  }

  .local-chip-name {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--text-primary);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .format-hint {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 0;
  }

  /* ── Presets ────────────────────────────────────── */
  .presets-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 6px;
  }

  .preset-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    padding: 12px 8px;
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    cursor: pointer;
    font-family: var(--font-mono);
    transition: all var(--transition);
  }

  .preset-btn:hover {
    border-color: var(--accent);
    background: var(--bg-hover);
  }

  .preset-active {
    border-color: var(--accent);
    background: var(--accent-bg);
  }

  .preset-icon {
    font-size: 14px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0;
  }

  .preset-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .preset-active .preset-label {
    color: var(--text-primary);
  }

  /* ── Device Panel ──────────────────────────────── */
  .device-panel {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .device-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border-dim);
  }

  .device-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    background: var(--bg-surface);
  }

  .device-value {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--accent);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .gpu-layers-control {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .gpu-hint {
    margin-top: 2px;
  }

  /* ── Settings ──────────────────────────────────── */
  .settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .setting-cell {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
  }

  .setting-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .setting-value {
    font-size: 12px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.06em;
  }

  .setting-slider {
    width: 100%;
    height: 4px;
    -webkit-appearance: none;
    appearance: none;
    background: var(--border-dim);
    outline: none;
  }

  .setting-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    background: var(--accent);
    cursor: pointer;
  }

  .setting-slider::-moz-range-thumb {
    width: 12px;
    height: 12px;
    background: var(--accent);
    cursor: pointer;
    border: none;
  }

  /* ── Prompt ────────────────────────────────────── */
  .prompt-input {
    width: 100%;
    min-height: 120px;
    padding: 12px;
    background: var(--bg-inset);
    border: 1px solid var(--border);
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.02em;
    line-height: 1.6;
    resize: vertical;
    transition: border-color var(--transition);
  }

  .prompt-input::placeholder {
    color: var(--text-muted);
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.06em;
  }

  .prompt-input:focus {
    outline: none;
    border-color: var(--accent);
  }

  .system-prompt-input {
    min-height: 60px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  /* ── Actions ───────────────────────────────────── */
  .action-row {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .generate-btn {
    padding: 10px 24px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
  }

  .action-info {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
  }

  /* ── Output ────────────────────────────────────── */
  .output-panel {
    padding: 12px;
    min-height: 100px;
    max-height: 500px;
    overflow-y: auto;
  }

  .output-text {
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
  }

  .cursor-blink {
    color: var(--accent);
    animation: blink 0.8s step-end infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
  }

  .stats-bar {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: var(--border-dim);
    border: 1px solid var(--border-dim);
  }

  .stat-cell {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 6px 10px;
    background: var(--bg-surface);
  }

  .stat-value {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-primary);
  }

  /* ── Error ─────────────────────────────────────── */
  .error-panel {
    padding: 12px;
  }

  .error-inner {
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }

  /* ── Responsive ────────────────────────────────── */
  @media (max-width: 700px) {
    .hero-specs {
      grid-template-columns: repeat(3, 1fr);
    }

    .presets-grid {
      grid-template-columns: repeat(3, 1fr);
    }

    .settings-grid {
      grid-template-columns: 1fr;
    }

    .device-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .stats-bar {
      grid-template-columns: repeat(3, 1fr);
    }
  }
</style>
