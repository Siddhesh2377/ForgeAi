<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { goto } from "$app/navigation";
  import { training, TRAINING_PRESETS, type TrainingMethod, type TrainingMode, type PresetId } from "$lib/training.svelte";
  import { datastudio } from "$lib/datastudio.svelte";
  import { model } from "$lib/model.svelte";

  onMount(() => {
    training.checkDeps();
    if (model.isLoaded) {
      training.modelPath = model.info?.file_path ?? null;
      training.modelName = model.info?.file_name ?? "";
      training.modelLayers = model.info?.layer_count ?? 0;
      training.modelArch = model.info?.architecture ?? "";
      training.modelParams = model.info?.parameter_count_display ?? "";
      training.loadModelCapabilities();
    }
  });

  onDestroy(() => {
    training.destroy();
  });

  const methods: { id: TrainingMethod; name: string; desc: string; recommended?: boolean }[] = [
    { id: "lora", name: "LORA", desc: "Low-rank adaptation — fast, memory efficient", recommended: true },
    { id: "qlora", name: "QLORA", desc: "4-bit quantized base + LoRA — minimal VRAM" },
    { id: "sft", name: "SFT", desc: "Supervised fine-tuning — all trainable parameters" },
    { id: "dpo", name: "DPO", desc: "Direct preference optimization — needs chosen/rejected pairs" },
    { id: "full_finetune", name: "FULL", desc: "Full fine-tune — highest quality, max VRAM" },
  ];

  const capAbbrev: Record<string, string> = {
    tool_calling: "TOOL",
    reasoning: "REAS",
    code: "CODE",
    math: "MATH",
    multilingual: "MLNG",
    instruct: "INST",
    safety: "SAFE",
  };

  function formatEta(seconds: number | null): string {
    if (seconds === null || seconds <= 0) return "--";
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    if (m > 60) {
      const h = Math.floor(m / 60);
      return `${h}h ${m % 60}m`;
    }
    return `${m}m ${s}s`;
  }

  function formatLoss(loss: number | null): string {
    if (loss === null) return "--";
    return loss.toFixed(4);
  }

  function useLoadedModel() {
    training.modelPath = model.info?.file_path ?? null;
    training.modelName = model.info?.file_name ?? "";
    training.modelLayers = model.info?.layer_count ?? 0;
    training.modelArch = model.info?.architecture ?? "";
    training.modelParams = model.info?.parameter_count_display ?? "";
    training.loadModelCapabilities();
  }

  function markCustom() {
    training.activePreset = "custom";
  }

  function openDataStudio() {
    if (training.dataset?.path) {
      datastudio.pendingPath = training.dataset.path;
    }
    goto("/datastudio");
  }

  let logsEl: HTMLDivElement;

  $effect(() => {
    if (logsEl && training.setupLogs.length > 0) {
      logsEl.scrollTop = logsEl.scrollHeight;
    }
  });

  // Surgery layer data with layer details
  let surgeryLayers = $derived.by(() => {
    const count = training.modelLayers;
    if (count <= 0) return [];

    const details = training.layerDetails;

    return Array.from({ length: count }, (_, i) => {
      const detail = details?.find(d => d.index === i) ?? null;
      return {
        index: i,
        isRemoved: training.layersToRemove.includes(i),
        isDuplicated: training.layersToDuplicate.some(d => d.source === i),
        detail,
      };
    });
  });

  function componentBarPercent(detail: NonNullable<typeof surgeryLayers[0]['detail']>) {
    const total = detail.total_bytes;
    if (total === 0) return { attn: 0, mlp: 0, norm: 0, other: 0 };
    return {
      attn: (detail.attention_bytes / total) * 100,
      mlp: (detail.mlp_bytes / total) * 100,
      norm: (detail.norm_bytes / total) * 100,
      other: (detail.other_bytes / total) * 100,
    };
  }

  function cellDisplay(val: any, maxLen = 60): string {
    if (val === null || val === undefined) return "";
    if (typeof val === "string") {
      return val.length > maxLen ? val.slice(0, maxLen) + "..." : val;
    }
    const s = JSON.stringify(val);
    return s.length > maxLen ? s.slice(0, maxLen) + "..." : s;
  }

  function groupTensorsByComponent(tensors: { name: string; dtype: string; shape: number[]; memory_display: string; component: string }[]) {
    const groups: Record<string, typeof tensors> = {};
    for (const t of tensors) {
      const key = t.component;
      if (!groups[key]) groups[key] = [];
      groups[key].push(t);
    }
    return groups;
  }
</script>

<div class="train-page">
  <!-- ── Hero Panel ── -->
  <div class="panel hero">
    <div class="hero-top">
      <div>
        <span class="label-xs" style="color: var(--text-muted);">MODULE 06</span>
        <h1 class="heading-lg">TRAINING & FINE-TUNING</h1>
        <p class="label" style="color: var(--text-secondary);">TARGETED GPU TRAINING & LAYER SURGERY</p>
      </div>
      <div class="hero-status">
        {#if training.deps?.ready}
          <span class="badge badge-success">READY</span>
        {:else if training.deps}
          <span class="badge badge-danger">SETUP REQUIRED</span>
        {:else}
          <span class="badge">CHECKING...</span>
        {/if}
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════ -->
  <!-- ── TOP 2-COLUMN GRID: Environment | Model+Output  -->
  <!-- ══════════════════════════════════════════════════ -->
  <div class="top-grid">
    <!-- ── Left Column: ENVIRONMENT ── -->
    <div class="panel env-panel">
      <div class="divider-label">ENVIRONMENT</div>

      {#if training.deps}
        <div class="env-list">
          <div class="env-row">
            <span class="dot" class:dot-success={training.deps.python_found} class:dot-danger={!training.deps.python_found}></span>
            <span class="label">PYTHON</span>
            <span class="code env-val">{training.deps.python_version ?? "NOT FOUND"}</span>
          </div>
          <div class="env-row">
            <span class="dot" class:dot-success={training.deps.cuda_available} class:dot-danger={!training.deps.cuda_available}></span>
            <span class="label">CUDA</span>
            <span class="code env-val">{training.deps.cuda_version ?? "N/A"}</span>
          </div>
          <div class="env-row">
            <span class="dot" class:dot-success={training.deps.packages_ready} class:dot-danger={!training.deps.packages_ready}></span>
            <span class="label">PACKAGES</span>
            <span class="code env-val">{training.deps.packages_ready ? "READY" : `${training.deps.missing_packages.length} MISSING`}</span>
          </div>
          <div class="env-row">
            <span class="dot" class:dot-success={training.deps.torch_version !== null} class:dot-danger={training.deps.torch_version === null}></span>
            <span class="label">TORCH</span>
            <span class="code env-val">{training.deps.torch_version ?? "NOT INSTALLED"}</span>
          </div>
        </div>

        {#if training.setupRunning}
          <div class="env-progress">
            <div class="progress-bar">
              <div class="progress-fill" style="width: {training.setupProgress?.percent ?? 0}%"></div>
            </div>
            <span class="label-xs env-progress-msg">{training.setupProgress?.message ?? "..."}</span>
            {#if training.setupLogs.length > 0}
              <div class="setup-logs" bind:this={logsEl}>
                {#each training.setupLogs as line}
                  <div class="log-line">{line}</div>
                {/each}
              </div>
            {/if}
          </div>
        {:else if !training.deps.ready}
          <button class="btn btn-accent" onclick={() => training.setup()}>
            INSTALL DEPENDENCIES
          </button>
        {:else}
          <div class="env-actions">
            <button class="btn btn-sm" onclick={() => training.setup()}>UPDATE</button>
            <button class="btn btn-sm" onclick={() => training.checkDeps()}>RE-CHECK</button>
          </div>
        {/if}

        {#if training.setupError}
          <div class="danger-text">{training.setupError}</div>
        {/if}
      {:else}
        <span class="label-xs" style="color: var(--text-muted);">CHECKING DEPENDENCIES...</span>
      {/if}
    </div>

    <!-- ── Right Column: MODEL + OUTPUT ── -->
    <div class="right-col">
      <div class="panel">
        <div class="divider-label">MODEL</div>
        <div class="model-row">
          <button class="btn btn-ghost" onclick={() => training.browseModel()}>SAFETENSORS DIR</button>
          <button class="btn btn-ghost" onclick={() => training.browseModelFile()}>GGUF FILE</button>
          {#if model.isLoaded && !training.modelPath}
            <button class="btn btn-accent" onclick={useLoadedModel}>USE LOADED</button>
          {/if}
        </div>
        {#if training.modelPath}
          <div class="model-info">
            <div class="info-item">
              <span class="label-xs">NAME</span>
              <span class="code">{training.modelName}</span>
            </div>
            {#if training.modelArch}
              <div class="info-item">
                <span class="label-xs">ARCH</span>
                <span class="code">{training.modelArch}</span>
              </div>
            {/if}
            {#if training.modelLayers > 0}
              <div class="info-item">
                <span class="label-xs">LAYERS</span>
                <span class="code">{training.modelLayers}</span>
              </div>
            {/if}
            {#if training.modelParams}
              <div class="info-item">
                <span class="label-xs">PARAMS</span>
                <span class="code">{training.modelParams}</span>
              </div>
            {/if}
          </div>
        {/if}
      </div>

      <div class="panel">
        <div class="divider-label">OUTPUT</div>
        <div class="output-row">
          <button class="btn btn-ghost" onclick={() => training.selectOutputPath()}>SELECT DIRECTORY</button>
          {#if training.outputPath}
            <span class="code output-path">{training.outputPath}</span>
          {/if}
        </div>
        {#if training.isLoraMethod && training.mode === "finetune"}
          <label class="module-check" style="margin-top: 8px;">
            <input type="checkbox" bind:checked={training.mergeAdapter} />
            <span class="label">MERGE ADAPTER INTO BASE MODEL</span>
          </label>
        {/if}
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════ -->
  <!-- ── MODE TOGGLE ── -->
  <!-- ══════════════════════════════════════════════════ -->
  <div class="mode-toggle">
    <button
      class="mode-btn"
      class:mode-active={training.mode === "finetune"}
      onclick={() => training.mode = "finetune"}
    >FINE-TUNE</button>
    <button
      class="mode-btn"
      class:mode-active={training.mode === "surgery"}
      onclick={() => training.mode = "surgery"}
    >LAYER SURGERY</button>
  </div>

  <!-- ══════════════════════════════════════════════════ -->
  <!-- ── FINE-TUNE MODE (2-Column Layout) ── -->
  <!-- ══════════════════════════════════════════════════ -->
  {#if training.mode === "finetune"}

    <!-- ── PRESETS ── -->
    <div class="panel">
      <div class="divider-label">PRESETS</div>
      <div class="preset-grid">
        {#each TRAINING_PRESETS as preset}
          <button
            class="preset-card"
            class:preset-active={training.activePreset === preset.id}
            onclick={() => training.applyPreset(preset)}
          >
            <span class="preset-name">{preset.name}</span>
            <span class="preset-vram">{preset.vram}</span>
            <span class="preset-desc">{preset.desc}</span>
            <span class="preset-detail">{preset.method.toUpperCase()} · R{preset.loraRank} · SEQ {preset.maxSeqLength}</span>
          </button>
        {/each}
        <button
          class="preset-card"
          class:preset-active={training.activePreset === "custom"}
          onclick={() => training.activePreset = "custom"}
        >
          <span class="preset-name">CUSTOM</span>
          <span class="preset-vram">--</span>
          <span class="preset-desc">Manual configuration</span>
          <span class="preset-detail">SET YOUR OWN PARAMETERS</span>
        </button>
      </div>
    </div>

    <div class="finetune-grid">
      <!-- ── LEFT COLUMN: Dataset ── -->
      <div class="finetune-col">
        <div class="panel">
          <div class="divider-label">DATASET</div>
          <div class="dataset-actions">
            <button class="btn btn-ghost" onclick={() => training.browseDataset()}>
              {training.dataset ? "CHANGE" : "BROWSE..."}
            </button>
            {#if training.dataset}
              <button class="btn btn-ghost" onclick={openDataStudio}>DATASTUDIO</button>
            {/if}
          </div>

          {#if training.datasetLoading}
            <span class="label-xs" style="color: var(--info);">LOADING...</span>
          {/if}

          {#if training.datasetError}
            <div class="danger-text">{training.datasetError}</div>
          {/if}

          {#if training.dataset}
            {#if training.dataset.preview.length > 0}
              <div class="dataset-preview">
                <table>
                  <thead>
                    <tr>
                      {#each training.dataset.columns.slice(0, 4) as col}
                        <th class="label-xs">{col.toUpperCase()}</th>
                      {/each}
                    </tr>
                  </thead>
                  <tbody>
                    {#each training.dataset.preview.slice(0, 5) as row}
                      <tr>
                        {#each training.dataset.columns.slice(0, 4) as col}
                          <td class="code">{cellDisplay(row[col])}</td>
                        {/each}
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {/if}

            <div class="dataset-meta">
              <span class="badge">{training.dataset.format.toUpperCase()}</span>
              {#if training.dataset.detected_template}
                <span class="badge badge-accent">{training.dataset.detected_template.toUpperCase()}</span>
              {/if}
              <span class="code">{training.dataset.rows.toLocaleString()} rows</span>
              <span class="code">{training.dataset.size_display}</span>
            </div>
          {/if}
        </div>
      </div>

      <!-- ── RIGHT COLUMN: Method + Hyperparameters + LoRA ── -->
      <div class="finetune-col">
        <!-- Method -->
        <div class="panel">
          <div class="divider-label">METHOD</div>
          <div class="method-grid">
            {#each methods as m}
              <button
                class="method-card"
                class:method-active={training.method === m.id}
                onclick={() => { training.method = m.id; markCustom(); }}
              >
                <span class="method-name">{m.name}</span>
                {#if m.recommended}
                  <span class="badge badge-accent" style="font-size: 7px; padding: 1px 4px;">REC</span>
                {/if}
                <span class="method-desc">{m.desc}</span>
              </button>
            {/each}
          </div>
        </div>

        <!-- Hyperparameters -->
        <div class="panel">
          <div class="divider-label">HYPERPARAMETERS</div>
          <div class="param-grid">
            <div class="param-item">
              <label class="label-xs" for="lr">LEARNING RATE</label>
              <input id="lr" type="number" step="0.0001" bind:value={training.learningRate} oninput={markCustom} />
            </div>
            <div class="param-item">
              <label class="label-xs" for="epochs">EPOCHS</label>
              <input id="epochs" type="number" min="1" max="100" bind:value={training.epochs} oninput={markCustom} />
            </div>
            <div class="param-item">
              <label class="label-xs" for="batch">BATCH SIZE</label>
              <input id="batch" type="number" min="1" max="128" bind:value={training.batchSize} oninput={markCustom} />
            </div>
            <div class="param-item">
              <label class="label-xs" for="seqlen">MAX SEQ LENGTH</label>
              <input id="seqlen" type="number" min="128" max="32768" step="128" bind:value={training.maxSeqLength} oninput={markCustom} />
            </div>
          </div>

          <button class="btn btn-ghost btn-sm" style="margin-top: 8px;" onclick={() => training.showAdvanced = !training.showAdvanced}>
            {training.showAdvanced ? "▾ HIDE ADVANCED" : "▸ SHOW ADVANCED"}
          </button>

          {#if training.showAdvanced}
            <div class="param-grid" style="margin-top: 8px;">
              <div class="param-item">
                <label class="label-xs" for="warmup">WARMUP STEPS</label>
                <input id="warmup" type="number" min="0" bind:value={training.warmupSteps} oninput={markCustom} />
              </div>
              <div class="param-item">
                <label class="label-xs" for="wd">WEIGHT DECAY</label>
                <input id="wd" type="number" step="0.001" bind:value={training.weightDecay} oninput={markCustom} />
              </div>
              <div class="param-item">
                <label class="label-xs" for="ga">GRAD ACCUMULATION</label>
                <input id="ga" type="number" min="1" bind:value={training.gradientAccumulationSteps} oninput={markCustom} />
              </div>
              <div class="param-item">
                <label class="label-xs" for="ss">SAVE STEPS</label>
                <input id="ss" type="number" min="50" bind:value={training.saveSteps} oninput={markCustom} />
              </div>
            </div>
          {/if}
        </div>

        <!-- LoRA Config -->
        {#if training.isLoraMethod}
          <div class="panel">
            <div class="divider-label">LORA CONFIG</div>
            <div class="param-grid">
              <div class="param-item">
                <label class="label-xs" for="rank">RANK</label>
                <input id="rank" type="number" min="4" max="256" step="4" bind:value={training.loraRank} oninput={markCustom} />
              </div>
              <div class="param-item">
                <label class="label-xs" for="alpha">ALPHA</label>
                <input id="alpha" type="number" min="1" bind:value={training.loraAlpha} oninput={markCustom} />
              </div>
              <div class="param-item">
                <label class="label-xs" for="dropout">DROPOUT</label>
                <input id="dropout" type="number" min="0" max="1" step="0.01" bind:value={training.loraDropout} oninput={markCustom} />
              </div>
              {#if training.method === "qlora"}
                <div class="param-item">
                  <label class="label-xs" for="qbits">QUANT BITS</label>
                  <select id="qbits" bind:value={training.quantizationBits}>
                    <option value={4}>4-BIT</option>
                    <option value={8}>8-BIT</option>
                  </select>
                </div>
              {/if}
            </div>

            <!-- Target Modules -->
            <div style="margin-top: 12px;">
              <span class="label-xs" style="color: var(--text-muted);">TARGET MODULES</span>
              <div class="module-grid">
                {#each training.availableModules as group}
                  {#each group.modules as mod}
                    <label class="module-check">
                      <input
                        type="checkbox"
                        checked={training.targetModules.includes(mod)}
                        onchange={() => training.toggleModule(mod)}
                      />
                      <span class="code">{mod}</span>
                    </label>
                  {/each}
                {/each}
                {#if training.availableModules.length === 0}
                  <span class="label-xs" style="color: var(--text-muted);">SELECT MODEL TO DETECT MODULES</span>
                {/if}
              </div>
            </div>
          </div>
        {/if}

        <!-- DPO Beta -->
        {#if training.method === "dpo"}
          <div class="panel">
            <div class="divider-label">DPO CONFIG</div>
            <div class="param-grid">
              <div class="param-item">
                <label class="label-xs" for="beta">BETA</label>
                <input id="beta" type="number" min="0.01" max="1" step="0.01" bind:value={training.dpoBeta} />
              </div>
            </div>
            <p class="label-xs" style="color: var(--text-muted); margin-top: 4px;">
              DATASET MUST HAVE "CHOSEN" AND "REJECTED" COLUMNS
            </p>
          </div>
        {/if}
      </div>
    </div>

    <!-- ── LAYER TARGETING (Full-width) ── -->
    {#if training.isLoraMethod}
      <div class="panel">
        <div class="divider-label">LAYER TARGETING</div>
        <p class="label-xs" style="color: var(--text-secondary); margin-bottom: 8px;">
          SELECT CAPABILITIES TO TARGET SPECIFIC LAYERS (EMPTY = ALL LAYERS)
        </p>

        {#if training.layerCapabilities.length > 0}
          <div class="cap-grid">
            {#each training.layerCapabilities as cap}
              <button
                class="cap-toggle"
                class:cap-on={training.capabilityToggles[cap.capability]}
                onclick={() => training.toggleCapability(cap.capability)}
              >
                <span class="cap-name">{cap.name}</span>
                <span class="cap-layers">LAYERS {cap.layers[0]}–{cap.layers[cap.layers.length - 1]}</span>
                <span class="cap-count">{cap.layers.length}L</span>
              </button>
            {/each}
          </div>
          {#if training.selectedLayers.length > 0}
            <div class="layer-summary">
              <span class="label-xs" style="color: var(--accent);">{training.selectedLayers.length} LAYERS SELECTED</span>
              <span class="code" style="font-size: 9px;">[{training.selectedLayers.join(", ")}]</span>
            </div>
          {/if}
        {:else}
          <span class="label-xs" style="color: var(--text-muted);">SELECT MODEL TO DETECT LAYER CAPABILITIES</span>
        {/if}
      </div>
    {/if}

    <!-- Start Training -->
    <div class="action-bar">
      {#if !training.training && !training.result}
        <button
          class="btn btn-accent"
          disabled={!training.canTrain}
          onclick={() => training.run()}
        >START TRAINING</button>
      {/if}

      {#if training.training}
        <button class="btn btn-danger" onclick={() => training.cancel()}>CANCEL</button>
      {/if}
    </div>

    <!-- Progress -->
    {#if training.training || training.result}
      <div class="panel">
        <div class="divider-label">
          {training.training ? "PROGRESS" : "RESULT"}
        </div>

        {#if training.training && training.progress}
          <div class="progress-info">
            <div class="progress-stats">
              <span class="badge badge-info">{training.progress.stage.toUpperCase()}</span>
              {#if training.progress.epoch !== null}
                <span class="code">EPOCH {training.progress.epoch}/{training.epochs}</span>
              {/if}
              {#if training.progress.step !== null}
                <span class="code">STEP {training.progress.step}/{training.progress.total_steps ?? "?"}</span>
              {/if}
            </div>
            <div class="progress-bar">
              <div class="progress-fill" style="width: {Math.max(0, training.progress.percent)}%"></div>
            </div>
            <div class="progress-stats">
              <span class="code">LOSS: {formatLoss(training.progress.loss)}</span>
              <span class="code">ETA: {formatEta(training.progress.eta_seconds ?? null)}</span>
              {#if training.progress.gpu_memory_used_mb}
                <span class="code">VRAM: {(training.progress.gpu_memory_used_mb / 1024).toFixed(1)} GB</span>
              {/if}
            </div>
          </div>

          {#if training.lossHistory.length > 2}
            {@const minLoss = Math.min(...training.lossHistory.map(h => h.loss))}
            {@const maxLoss = Math.max(...training.lossHistory.map(h => h.loss))}
            {@const range = Math.max(maxLoss - minLoss, 0.001)}
            <div class="loss-chart">
              <span class="label-xs" style="color: var(--text-muted);">LOSS</span>
              <svg viewBox="0 0 200 40" class="sparkline">
                <polyline
                  fill="none"
                  stroke="var(--accent)"
                  stroke-width="1"
                  points={training.lossHistory.map((h, i) => {
                    const x = (i / Math.max(training.lossHistory.length - 1, 1)) * 200;
                    const y = 38 - ((h.loss - minLoss) / range) * 36;
                    return `${x},${y}`;
                  }).join(" ")}
                />
              </svg>
            </div>
          {/if}
        {/if}

        {#if training.result}
          <div class="result-info">
            <div class="info-item">
              <span class="label-xs">METHOD</span>
              <span class="code">{training.result.method.toUpperCase()}</span>
            </div>
            <div class="info-item">
              <span class="label-xs">EPOCHS</span>
              <span class="code">{training.result.epochs_completed}</span>
            </div>
            <div class="info-item">
              <span class="label-xs">FINAL LOSS</span>
              <span class="code">{formatLoss(training.result.final_loss)}</span>
            </div>
            <div class="info-item">
              <span class="label-xs">OUTPUT SIZE</span>
              <span class="code">{training.result.output_size_display}</span>
            </div>
            <div class="info-item">
              <span class="label-xs">ADAPTER MERGED</span>
              <span class="code">{training.result.adapter_merged ? "YES" : "NO"}</span>
            </div>
          </div>
          <div class="code" style="font-size: 9px; margin-top: 8px; color: var(--text-secondary);">
            {training.result.output_path}
          </div>
        {/if}
      </div>
    {/if}

    {#if training.error}
      <div class="panel" style="border-color: var(--danger);">
        <div class="danger-text">{training.error}</div>
      </div>
    {/if}

  <!-- ══════════════════════════════════════════════════ -->
  <!-- ── SURGERY MODE (Rich Layer Table) ── -->
  <!-- ══════════════════════════════════════════════════ -->
  {:else}

    {#if training.modelLayers > 0}
      <div class="panel">
        <div class="divider-label">LAYERS</div>

        <div class="surgery-summary">
          <span class="code">{training.modelLayers}</span>
          <span class="label-xs">→</span>
          <span class="code" style="color: var(--accent);">{training.surgeryPreview.final}</span>
          <span class="label-xs">LAYERS</span>
          {#if training.surgeryPreview.removed > 0}
            <span class="badge badge-danger" style="font-size: 7px;">-{training.surgeryPreview.removed} REMOVED</span>
          {/if}
          {#if training.surgeryPreview.added > 0}
            <span class="badge badge-success" style="font-size: 7px;">+{training.surgeryPreview.added} DUPLICATED</span>
          {/if}
          {#if training.layerDetailsLoading}
            <span class="label-xs" style="color: var(--info); margin-left: auto;">LOADING DETAILS...</span>
          {/if}
        </div>

        <div class="layer-table">
          <div class="layer-header">
            <span class="label-xs lh-idx">IDX</span>
            <span class="label-xs lh-mem">MEMORY</span>
            <span class="label-xs lh-bar">COMPONENTS</span>
            <span class="label-xs lh-caps">CAPS</span>
            <span class="label-xs lh-act">ACT</span>
          </div>

          {#each surgeryLayers as layer}
            {#if layer.isRemoved}
              <div class="layer-row-surgery layer-removed">
                <span class="code lh-idx">{layer.index}</span>
                <span class="code lh-mem" style="color: var(--danger); font-size: 9px;" >MARKED FOR REMOVAL</span>
                <span class="lh-bar"></span>
                <span class="lh-caps"></span>
                <span class="lh-act">
                  <button class="btn-icon" title="Restore" onclick={() => training.toggleLayerRemove(layer.index)}>↩</button>
                  <button class="btn-icon" title="Duplicate" onclick={() => training.addDuplicate(layer.index)}>+</button>
                </span>
              </div>
            {:else}
              <!-- svelte-ignore a11y_click_events_have_key_events -->
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="layer-row-surgery"
                class:layer-duped={layer.isDuplicated}
                class:layer-expanded={training.expandedSurgeryLayer === layer.index}
                onclick={() => training.expandedSurgeryLayer = training.expandedSurgeryLayer === layer.index ? null : layer.index}
              >
                <span class="code lh-idx">{layer.index}</span>
                <span class="code lh-mem">{layer.detail?.display ?? "--"}</span>
                <span class="lh-bar">
                  {#if layer.detail}
                    {@const pct = componentBarPercent(layer.detail)}
                    <div class="component-bar">
                      {#if pct.attn > 0}
                        <div class="cb-attn" style="width: {pct.attn}%;" title="Attention {pct.attn.toFixed(0)}%"></div>
                      {/if}
                      {#if pct.mlp > 0}
                        <div class="cb-mlp" style="width: {pct.mlp}%;" title="MLP {pct.mlp.toFixed(0)}%"></div>
                      {/if}
                      {#if pct.norm > 0}
                        <div class="cb-norm" style="width: {pct.norm}%;" title="Norm {pct.norm.toFixed(0)}%"></div>
                      {/if}
                      {#if pct.other > 0}
                        <div class="cb-other" style="width: {pct.other}%;" title="Other {pct.other.toFixed(0)}%"></div>
                      {/if}
                    </div>
                  {/if}
                </span>
                <span class="lh-caps">
                  {#if layer.detail}
                    {#each layer.detail.capabilities as cap}
                      <span class="cap-badge">{capAbbrev[cap] ?? cap.slice(0, 4).toUpperCase()}</span>
                    {/each}
                  {/if}
                </span>
                <span class="lh-act" onclick={(e: MouseEvent) => e.stopPropagation()}>
                  <button class="btn-icon btn-icon-danger" title="Remove" onclick={() => training.toggleLayerRemove(layer.index)}>×</button>
                  <button class="btn-icon" title="Duplicate" onclick={() => training.addDuplicate(layer.index)}>+</button>
                </span>
              </div>

              <!-- Expanded tensor detail -->
              {#if training.expandedSurgeryLayer === layer.index && layer.detail}
                <div class="layer-detail-expand">
                  {#each Object.entries(groupTensorsByComponent(layer.detail.tensors)) as [component, tensors]}
                    <div class="tensor-group">
                      <div class="tensor-group-header">
                        <span class="label-xs" style="color: var(--component-{component}, var(--text-muted));">{component.toUpperCase()}</span>
                        <span class="label-xs" style="color: var(--text-muted);">({tensors.length} tensors)</span>
                      </div>
                      {#each tensors as t}
                        <div class="tensor-row">
                          <span class="code tensor-name">{t.name}</span>
                          <span class="code tensor-dtype">{t.dtype}</span>
                          <span class="code tensor-shape">[{t.shape.join("×")}]</span>
                          <span class="code tensor-mem">{t.memory_display}</span>
                        </div>
                      {/each}
                    </div>
                  {/each}
                </div>
              {/if}
            {/if}
          {/each}
        </div>

        <!-- Component bar legend -->
        <div class="component-legend">
          <span class="legend-item"><span class="legend-swatch cb-attn"></span><span class="label-xs">ATTENTION</span></span>
          <span class="legend-item"><span class="legend-swatch cb-mlp"></span><span class="label-xs">MLP</span></span>
          <span class="legend-item"><span class="legend-swatch cb-norm"></span><span class="label-xs">NORM</span></span>
          <span class="legend-item"><span class="legend-swatch cb-other"></span><span class="label-xs">OTHER</span></span>
        </div>

        {#if training.layersToDuplicate.length > 0}
          <div style="margin-top: 8px;">
            <span class="label-xs" style="color: var(--text-muted);">DUPLICATIONS:</span>
            {#each training.layersToDuplicate as dup, i}
              <div class="dup-item">
                <span class="code">LAYER {dup.source} → INSERT AT {dup.insertAt}</span>
                <button class="btn btn-ghost btn-sm" style="padding: 1px 4px; font-size: 8px; color: var(--danger);" onclick={() => training.removeDuplicate(i)}>×</button>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {:else if training.modelPath}
      <div class="panel">
        <span class="label-xs" style="color: var(--text-muted);">COULD NOT DETECT LAYERS — TRY A DIFFERENT MODEL</span>
      </div>
    {:else}
      <div class="panel">
        <span class="label-xs" style="color: var(--text-muted);">SELECT A MODEL ABOVE TO BEGIN SURGERY</span>
      </div>
    {/if}

    <div class="action-bar">
      {#if !training.surgeryRunning && !training.surgeryResult}
        <button
          class="btn btn-accent"
          disabled={!training.modelPath || !training.outputPath || (training.layersToRemove.length === 0 && training.layersToDuplicate.length === 0)}
          onclick={() => training.runSurgery()}
        >EXECUTE SURGERY</button>
      {/if}
      {#if training.surgeryRunning}
        <button class="btn btn-danger" onclick={() => training.cancelSurgery()}>CANCEL</button>
      {/if}
    </div>

    {#if training.surgeryRunning && training.progress}
      <div class="panel">
        <div class="divider-label">PROGRESS</div>
        <div class="progress-info">
          <div class="progress-bar">
            <div class="progress-fill" style="width: {Math.max(0, training.progress.percent)}%"></div>
          </div>
          <span class="label-xs" style="color: var(--text-secondary);">{training.progress.message}</span>
        </div>
      </div>
    {/if}

    {#if training.surgeryResult}
      <div class="panel">
        <div class="divider-label">RESULT</div>
        <div class="result-info">
          <div class="info-item">
            <span class="label-xs">ORIGINAL</span>
            <span class="code">{training.surgeryResult.original_layers} LAYERS</span>
          </div>
          <div class="info-item">
            <span class="label-xs">FINAL</span>
            <span class="code" style="color: var(--accent);">{training.surgeryResult.final_layers} LAYERS</span>
          </div>
          <div class="info-item">
            <span class="label-xs">TENSORS</span>
            <span class="code">{training.surgeryResult.tensors_written}</span>
          </div>
          <div class="info-item">
            <span class="label-xs">SIZE</span>
            <span class="code">{training.surgeryResult.output_size_display}</span>
          </div>
        </div>
      </div>
    {/if}

    {#if training.error}
      <div class="panel" style="border-color: var(--danger);">
        <div class="danger-text">{training.error}</div>
      </div>
    {/if}

  {/if}
</div>

<style>
  .train-page {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .hero {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .hero-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  .hero-status {
    flex-shrink: 0;
  }

  /* ── Top 2-column grid ── */
  .top-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    align-items: start;
  }
  .right-col {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  /* ── Environment ── */
  .env-panel {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .env-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .env-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    border-bottom: 1px solid var(--border-dim);
  }
  .env-row:last-child {
    border-bottom: none;
  }
  .env-val {
    margin-left: auto;
    color: var(--text-secondary);
    font-size: 10px;
  }
  .env-progress {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .env-progress-msg {
    color: var(--text-secondary);
    font-size: 9px;
    word-break: break-all;
  }
  .setup-logs {
    max-height: 180px;
    overflow-y: auto;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
    padding: 6px 8px;
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--text-muted);
    line-height: 1.5;
    margin-top: 4px;
  }
  .log-line {
    white-space: pre-wrap;
    word-break: break-all;
  }
  .env-actions {
    display: flex;
    gap: 6px;
  }
  .btn-sm {
    padding: 5px 12px;
    font-size: 9px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 120ms ease;
  }
  .btn-sm:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
    border-color: var(--accent);
  }

  /* ── Mode Toggle ── */
  .mode-toggle {
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
  }
  .mode-btn {
    flex: 1;
    padding: 10px;
    background: var(--bg-surface);
    border: none;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 120ms ease;
  }
  .mode-btn:hover {
    background: var(--bg-hover);
  }
  .mode-active {
    background: var(--accent-bg);
    color: var(--accent);
    border-bottom: 2px solid var(--accent);
  }

  /* ── Presets ── */
  .preset-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
  }
  .preset-card {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px 10px;
    border: 1px solid var(--border-dim);
    background: var(--bg-surface);
    cursor: pointer;
    text-align: left;
    font-family: var(--font-mono);
    transition: all 120ms ease;
  }
  .preset-card:hover {
    border-color: var(--border);
    background: var(--bg-hover);
  }
  .preset-active {
    border-color: var(--accent);
    background: var(--accent-bg);
  }
  .preset-name {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--text-primary);
  }
  .preset-active .preset-name {
    color: var(--accent);
  }
  .preset-vram {
    font-size: 10px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.05em;
  }
  .preset-desc {
    font-size: 8px;
    color: var(--text-muted);
    letter-spacing: 0.05em;
  }
  .preset-detail {
    font-size: 7px;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    margin-top: 2px;
    opacity: 0.7;
  }

  /* ── Fine-tune 2-Column Grid ── */
  .finetune-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    align-items: start;
  }
  .finetune-col {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  /* ── Model ── */
  .model-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }
  .model-info {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    padding: 8px 0 0;
    border-top: 1px solid var(--border-dim);
  }
  .info-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  /* ── Output ── */
  .output-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .output-path {
    font-size: 9px;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* ── Dataset ── */
  .dataset-actions {
    display: flex;
    gap: 6px;
    margin-bottom: 6px;
  }
  .dataset-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 8px 0 0;
  }
  .dataset-preview {
    max-height: 160px;
    overflow-y: auto;
    border: 1px solid var(--border-dim);
    background: var(--bg-inset);
  }
  .dataset-preview table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9px;
  }
  .dataset-preview th {
    padding: 4px 8px;
    text-align: left;
    border-bottom: 1px solid var(--border-dim);
    background: var(--bg-surface);
    position: sticky;
    top: 0;
  }
  .dataset-preview td {
    padding: 3px 8px;
    border-bottom: 1px solid var(--border-dim);
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* ── Method ── */
  .method-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 6px;
  }
  .method-card {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px 10px;
    border: 1px solid var(--border-dim);
    background: var(--bg-surface);
    cursor: pointer;
    text-align: left;
    font-family: var(--font-mono);
    transition: all 120ms ease;
  }
  .method-card:hover {
    border-color: var(--border);
    background: var(--bg-hover);
  }
  .method-active {
    border-color: var(--accent);
    background: var(--accent-bg);
  }
  .method-name {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--text-primary);
  }
  .method-active .method-name {
    color: var(--accent);
  }
  .method-desc {
    font-size: 8px;
    color: var(--text-muted);
    letter-spacing: 0.05em;
  }

  /* ── Params ── */
  .param-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }
  .param-item {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .param-item input, .param-item select {
    padding: 5px 8px;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 10px;
    outline: none;
  }
  .param-item input:focus, .param-item select:focus {
    border-color: var(--accent);
  }

  /* ── Modules ── */
  .module-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
  }
  .module-check {
    display: flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
    padding: 3px 6px;
    border: 1px solid var(--border-dim);
    background: var(--bg-surface);
    font-size: 10px;
  }
  .module-check input {
    accent-color: var(--accent);
  }

  /* ── Capability Toggles ── */
  .cap-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 6px;
  }
  .cap-toggle {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 6px 10px;
    border: 1px solid var(--border-dim);
    background: var(--bg-surface);
    cursor: pointer;
    font-family: var(--font-mono);
    text-align: left;
    transition: all 120ms ease;
  }
  .cap-toggle:hover {
    border-color: var(--border);
  }
  .cap-on {
    border-color: var(--accent);
    background: var(--accent-bg);
  }
  .cap-name {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--text-primary);
  }
  .cap-on .cap-name {
    color: var(--accent);
  }
  .cap-layers, .cap-count {
    font-size: 8px;
    color: var(--text-muted);
    letter-spacing: 0.05em;
  }
  .layer-summary {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    padding: 6px 8px;
    border: 1px solid var(--accent-dim);
    background: var(--accent-bg);
  }

  /* ── Action Bar ── */
  .action-bar {
    display: flex;
    gap: 8px;
  }

  /* ── Progress ── */
  .progress-bar {
    height: 4px;
    background: var(--border-dim);
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent);
    transition: width 300ms ease;
  }
  .progress-info {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .progress-stats {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .result-info {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }

  /* ── Loss Chart ── */
  .loss-chart {
    margin-top: 8px;
    padding: 8px;
    background: var(--bg-inset);
    border: 1px solid var(--border-dim);
  }
  .sparkline {
    width: 100%;
    height: 40px;
  }

  /* ══════════════════════════════════════════════════ */
  /* ── Surgery (Rich Layer Table) ── */
  /* ══════════════════════════════════════════════════ */
  .surgery-summary {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 0;
    flex-wrap: wrap;
  }
  .layer-table {
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid var(--border-dim);
  }
  .layer-header {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 5px 8px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-dim);
    position: sticky;
    top: 0;
    z-index: 2;
  }
  .lh-idx { width: 36px; text-align: center; flex-shrink: 0; }
  .lh-mem { width: 70px; flex-shrink: 0; font-size: 10px; }
  .lh-bar { flex: 1; min-width: 80px; }
  .lh-caps { width: 120px; flex-shrink: 0; display: flex; gap: 3px; flex-wrap: wrap; }
  .lh-act { width: 50px; flex-shrink: 0; display: flex; gap: 2px; justify-content: center; }

  .layer-row-surgery {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-bottom: 1px solid var(--border-dim);
    transition: background 120ms ease;
    cursor: pointer;
  }
  .layer-row-surgery:hover {
    background: var(--bg-hover);
  }
  .layer-removed {
    background: rgba(239, 68, 68, 0.06);
    cursor: default;
  }
  .layer-duped {
    background: rgba(34, 197, 94, 0.06);
  }
  .layer-expanded {
    background: var(--accent-bg);
    border-bottom-color: var(--accent);
  }

  /* ── Component Bar ── */
  .component-bar {
    display: flex;
    height: 10px;
    width: 100%;
    overflow: hidden;
    border: 1px solid var(--border-dim);
  }
  .cb-attn { background: #3b82f6; }
  .cb-mlp { background: #f59e0b; }
  .cb-norm { background: #22c55e; }
  .cb-other { background: #6b7280; }

  /* ── Capability Badges ── */
  .cap-badge {
    font-family: var(--font-mono);
    font-size: 7px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 1px 3px;
    border: 1px solid var(--border-dim);
    background: var(--bg-surface);
    color: var(--text-muted);
    white-space: nowrap;
  }

  /* ── Action Buttons (icon-style) ── */
  .btn-icon {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
    transition: all 120ms ease;
    padding: 0;
  }
  .btn-icon:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--bg-hover);
  }
  .btn-icon-danger:hover {
    border-color: var(--danger);
    color: var(--danger);
  }

  /* ── Expanded Layer Detail ── */
  .layer-detail-expand {
    padding: 8px 12px 8px 44px;
    background: var(--bg-inset);
    border-bottom: 1px solid var(--border-dim);
  }
  .tensor-group {
    margin-bottom: 8px;
  }
  .tensor-group:last-child {
    margin-bottom: 0;
  }
  .tensor-group-header {
    display: flex;
    gap: 6px;
    align-items: center;
    margin-bottom: 3px;
  }
  .tensor-row {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 1px 0;
    font-size: 9px;
  }
  .tensor-name {
    width: 140px;
    flex-shrink: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .tensor-dtype {
    width: 50px;
    flex-shrink: 0;
    color: var(--text-muted);
  }
  .tensor-shape {
    width: 100px;
    flex-shrink: 0;
    color: var(--text-muted);
  }
  .tensor-mem {
    margin-left: auto;
    color: var(--text-secondary);
  }

  /* ── Component Legend ── */
  .component-legend {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    padding: 4px 0;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .legend-swatch {
    width: 10px;
    height: 10px;
    display: inline-block;
  }

  /* ── Component CSS Custom Properties ── */
  :global(:root) {
    --component-attention: #3b82f6;
    --component-mlp: #f59e0b;
    --component-norm: #22c55e;
    --component-other: #6b7280;
  }

  /* ── Duplications ── */
  .dup-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 2px 0;
  }

  @media (max-width: 700px) {
    .top-grid { grid-template-columns: 1fr; }
    .finetune-grid { grid-template-columns: 1fr; }
    .param-grid { grid-template-columns: repeat(2, 1fr); }
    .method-grid { grid-template-columns: 1fr; }
    .preset-grid { grid-template-columns: repeat(2, 1fr); }
    .lh-caps { display: none; }
    .lh-bar { min-width: 60px; }
  }
</style>
