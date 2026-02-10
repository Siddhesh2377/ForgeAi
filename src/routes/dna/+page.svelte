<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { dna, MERGE_PRESETS, type MergePreset, type DnaMode, type DnaTab, type ParentModelInfo, type LayerProfile, type HoveredLayer, type LayerAnalysis, type Capability } from "$lib/dna.svelte";

  function getSerial() {
    return `FRG-${String(Date.now()).slice(-6)}`;
  }

  const serial = getSerial();

  // ── Isometric helpers (ported from inspect page) ──────
  const cos30 = Math.cos(Math.PI / 6);
  const sin30 = 0.5;

  function iso(x: number, y: number, z: number): string {
    return `${((x - y) * cos30).toFixed(1)},${((x + y) * sin30 - z).toFixed(1)}`;
  }

  function isoXY(x: number, y: number, z: number): [number, number] {
    return [(x - y) * cos30, (x + y) * sin30 - z];
  }

  function slabColor(rgb: [number, number, number], factor: number): string {
    return `rgb(${Math.round(rgb[0] * factor)}, ${Math.round(rgb[1] * factor)}, ${Math.round(rgb[2] * factor)})`;
  }

  function layerRgb(parentId: string, layerIndex: number): [number, number, number] {
    // Use analysis color when computed
    const analysis = dna.getLayerAnalysis(parentId, layerIndex);
    if (analysis) {
      return hexToRgb(analysis.color);
    }

    // Fallback: attn/mlp ratio blue→green
    const comps = dna.layerComponents[parentId];
    if (!comps) return [100, 100, 100];
    const layer = comps.find(c => c.layer_index === layerIndex);
    if (!layer) return [100, 100, 100];

    const attnCount = layer.attention_tensors.length;
    const total = attnCount + layer.mlp_tensors.length + layer.norm_tensors.length + layer.other_tensors.length;
    const ratio = total > 0 ? attnCount / total : 0.5;

    const r = Math.round(59 * ratio + 34 * (1 - ratio));
    const g = Math.round(130 * ratio + 197 * (1 - ratio));
    const b = Math.round(246 * ratio + 94 * (1 - ratio));
    return [r, g, b];
  }

  function hexToRgb(hex: string): [number, number, number] {
    const m = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [100, 100, 100];
  }

  // ── Multi-tower geometry ──────────────────────────────
  interface TowerDef {
    id: string;
    label: string;
    layerCount: number;
    isOffspring: boolean;
    originX: number;
  }

  // ── Pan / Zoom state ─────────────────────────────────
  let viewZoom = $state(1);
  let viewPanX = $state(0);
  let viewPanY = $state(0);
  let isDragging = $state(false);
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartPanX = 0;
  let dragStartPanY = 0;

  function handleWheel(e: WheelEvent) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    viewZoom = Math.max(0.3, Math.min(4, viewZoom + delta));
  }

  function handleDragStart(e: MouseEvent) {
    if (e.button !== 0) return;
    isDragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragStartPanX = viewPanX;
    dragStartPanY = viewPanY;
  }

  function handleDragMove(e: MouseEvent) {
    if (!isDragging) return;
    const scale = 1 / viewZoom;
    viewPanX = dragStartPanX - (e.clientX - dragStartX) * scale;
    viewPanY = dragStartPanY - (e.clientY - dragStartY) * scale;
  }

  function handleDragEnd() {
    isDragging = false;
  }

  function resetView() {
    viewZoom = 1;
    viewPanX = 0;
    viewPanY = 0;
  }

  let towerDefs = $derived.by((): TowerDef[] => {
    const W = 60;
    const gap = 50;
    const towers: TowerDef[] = [];

    dna.parents.forEach((p, idx) => {
      towers.push({
        id: p.id,
        label: `P${idx + 1}`,
        layerCount: p.layer_count ?? 0,
        isOffspring: false,
        originX: idx * (W + gap),
      });
    });

    if (dna.parents.length >= 2) {
      towers.push({
        id: "__offspring__",
        label: "OUT",
        layerCount: dna.maxLayers,
        isOffspring: true,
        originX: dna.parents.length * (W + gap) + gap * 0.5,
      });
    }

    return towers;
  });

  let isoGeometry = $derived.by(() => {
    if (towerDefs.length === 0) return { towers: [] as any[], viewBox: "0 0 200 200" };

    const W = 60, D = 40;
    const maxLayers = Math.max(...towerDefs.map(t => t.layerCount), 1);
    const H = Math.max(2, Math.min(5, Math.floor(200 / maxLayers)));
    const gapZ = Math.max(1, Math.floor(H * 0.3));
    const step = H + gapZ;

    const towers = towerDefs.map(tower => {
      const polys = [];
      for (let i = 0; i < Math.min(tower.layerCount, 64); i++) {
        const z = i * step;
        const ox = tower.originX;
        polys.push({
          top: `${iso(ox, 0, z + H)} ${iso(ox + W, 0, z + H)} ${iso(ox + W, D, z + H)} ${iso(ox, D, z + H)}`,
          front: `${iso(ox, D, z + H)} ${iso(ox + W, D, z + H)} ${iso(ox + W, D, z)} ${iso(ox, D, z)}`,
          right: `${iso(ox + W, 0, z + H)} ${iso(ox + W, D, z + H)} ${iso(ox + W, D, z)} ${iso(ox + W, 0, z)}`,
        });
      }
      return { ...tower, polys, W, D, H, step };
    });

    // Compute viewBox bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const tower of towers) {
      const layerCount = Math.min(tower.layerCount, 64);
      for (let i = 0; i < layerCount; i++) {
        const z0 = i * step;
        const z1 = z0 + H;
        const ox = tower.originX;
        for (const pt of [[ox,0,z0],[ox+W,0,z0],[ox,D,z0],[ox+W,D,z0],[ox,0,z1],[ox+W,0,z1],[ox,D,z1],[ox+W,D,z1]]) {
          const [sx, sy] = isoXY(pt[0], pt[1], pt[2]);
          if (sx < minX) minX = sx;
          if (sx > maxX) maxX = sx;
          if (sy < minY) minY = sy;
          if (sy > maxY) maxY = sy;
        }
      }
    }

    if (minX === Infinity) return { towers, viewBox: "0 0 200 200", baseW: 200, baseH: 200, baseCX: 100, baseCY: 100 };

    const pad = 20;
    const baseW = maxX - minX + 2 * pad;
    const baseH = maxY - minY + 2 * pad;
    const baseCX = minX - pad + baseW / 2;
    const baseCY = minY - pad + baseH / 2;

    return { towers, viewBox: "", baseW, baseH, baseCX, baseCY };
  });

  let finalViewBox = $derived.by(() => {
    const g = isoGeometry;
    const w = g.baseW / viewZoom;
    const h = g.baseH / viewZoom;
    const cx = g.baseCX + viewPanX;
    const cy = g.baseCY + viewPanY;
    return `${cx - w / 2} ${cy - h / 2} ${w} ${h}`;
  });

  function getSlabRgb(tower: TowerDef, layerIndex: number): [number, number, number] {
    if (!tower.isOffspring) {
      return layerRgb(tower.id, layerIndex);
    }
    // Offspring: use assigned parent color
    const assignment = dna.layerAssignments.find(a => a.layerIndex === layerIndex);
    if (!assignment) return [50, 50, 50];
    const parent = dna.parents.find(p => p.id === assignment.sourceParentId);
    if (!parent) return [50, 50, 50];
    return hexToRgb(parent.color);
  }

  function isSlabAssigned(tower: TowerDef, layerIndex: number): boolean {
    if (!tower.isOffspring) return true;
    return !!dna.layerAssignments.find(a => a.layerIndex === layerIndex);
  }

  // Mouse tracking for tooltip
  let mouseX = $state(0);
  let mouseY = $state(0);

  function handleIsoMouseMove(e: MouseEvent) {
    mouseX = e.clientX;
    mouseY = e.clientY;
  }

  function handleSlabClick(tower: TowerDef, layerIndex: number) {
    if (!tower.isOffspring) {
      dna.assignLayer(layerIndex, tower.id);
    }
  }

  // ── Tab helpers ───────────────────────────────────────

  function getSpecBadge(spec: string): { label: string; color: string } {
    switch (spec) {
      case "syntactic": return { label: "SYN", color: "#4ECDC4" };
      case "semantic": return { label: "SEM", color: "#A78BFA" };
      case "reasoning": return { label: "RSN", color: "#FF6B35" };
      default: return { label: "UNK", color: "#666" };
    }
  }

  let methodsForMode = $derived(
    dna.methods.filter((m) => {
      if (dna.mode === "easy") return m.difficulty === "easy";
      if (dna.mode === "intermediate")
        return m.difficulty === "easy" || m.difficulty === "intermediate";
      return true;
    })
  );

  let currentMethod = $derived(
    dna.methods.find((m) => m.id === dna.selectedMethod)
  );

  // Tooltip data for hovered slab
  let tooltipData = $derived.by(() => {
    if (!dna.hoveredLayer) return null;
    const { parentId, layerIndex } = dna.hoveredLayer;
    if (!parentId) return null;
    return dna.getLayerTooltipData(parentId, layerIndex);
  });

  let tooltipAnalysis = $derived.by((): LayerAnalysis | null => {
    if (!dna.hoveredLayer) return null;
    const { parentId, layerIndex } = dna.hoveredLayer;
    if (!parentId || parentId === "__offspring__") return null;
    return dna.getLayerAnalysis(parentId, layerIndex);
  });

  let tooltipParent = $derived.by(() => {
    if (!dna.hoveredLayer?.parentId) return null;
    if (dna.hoveredLayer.parentId === "__offspring__") return null;
    return dna.parents.find(p => p.id === dna.hoveredLayer!.parentId) ?? null;
  });

  onMount(() => { dna.init(); });
  onDestroy(() => { dna.destroy(); });
</script>

<svelte:window onmousemove={(e) => { mouseX = e.clientX; mouseY = e.clientY; }} />

<div class="dna-page fade-in">
  <!-- ── Hero Panel ──────────────────────────────────── -->
  <div class="hero panel">
    <div class="hero-header">
      <div class="hero-id">
        <span class="label-xs">{serial}</span>
        <span class="label-xs" style="color: var(--text-muted);">DNA-MERGE</span>
      </div>
      <div style="display: flex; gap: 8px; align-items: center;">
        <div class="badge" class:badge-accent={dna.status === 'ready' || dna.status === 'complete'} class:badge-danger={dna.status === 'error'}>
          <span class="dot" class:dot-active={dna.status !== 'idle' && dna.status !== 'error'}></span>
          {dna.status.toUpperCase()}
        </div>
        <button class="btn btn-xs btn-danger-ghost" onclick={() => dna.reset()}>RESET</button>
      </div>
    </div>

    <div class="hero-body">
      <h1 class="hero-title">M-DNA FORGE</h1>
      <p class="hero-subtitle">GENETIC MODEL ENGINEERING</p>
    </div>

    <div class="hero-specs">
      <div class="spec-cell">
        <span class="label-xs">PARENTS</span>
        <span class="spec-value">{dna.loadedCount}/5</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">METHOD</span>
        <span class="spec-value">{currentMethod?.name ?? '---'}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">MODE</span>
        <span class="spec-value">{dna.mode.toUpperCase()}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">LAYERS</span>
        <span class="spec-value">{dna.maxLayers || '---'}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">EST.SIZE</span>
        <span class="spec-value">{dna.preview?.estimated_output_display ?? '---'}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">STATUS</span>
        <span class="spec-value">{dna.canMerge ? 'READY' : 'CONFIG'}</span>
      </div>
    </div>

    <div class="hero-footer">
      <div class="barcode" style="max-width: 180px;"></div>
      <span class="label-xs">HYBRID OFFSPRING / {dna.loadedCount} GENES / LOCAL</span>
    </div>
  </div>

  <!-- ── 2-Column Grid ───────────────────────────────── -->
  <div class="dna-grid">

    <!-- ── Left: Isometric Visualization ──────────────── -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="dna-viz panel-flat"
      class:iso-grabbing={isDragging}
      onmousemove={(e) => { handleIsoMouseMove(e); handleDragMove(e); }}
      onmouseup={handleDragEnd}
      onmouseleave={handleDragEnd}
    >
      {#if dna.parents.length >= 1}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div class="iso-viewport" onwheel={handleWheel} onmousedown={handleDragStart}>
          <svg
            viewBox={finalViewBox}
            class="iso-svg"
            preserveAspectRatio="xMidYMid meet"
          >
            <defs>
              <pattern id="dna-dots" width="4" height="4" patternUnits="userSpaceOnUse">
                <rect x="0" y="0" width="1.5" height="1.5" fill="white" opacity="0.05"/>
              </pattern>
            </defs>

            {#each isoGeometry.towers as tower}
              {#each tower.polys as poly, i}
                {@const rgb = getSlabRgb(tower, i)}
                {@const assigned = isSlabAssigned(tower, i)}
                {@const isHovered = dna.hoveredLayer?.layerIndex === i && (
                  dna.hoveredLayer?.parentId === tower.id ||
                  (tower.isOffspring && dna.hoveredLayer?.parentId !== null)
                )}
                {@const isDimmed = dna.hoveredLayer !== null && !isHovered}
                <!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events -->
                <g
                  class="iso-slab"
                  class:iso-dimmed={isDimmed && assigned}
                  class:iso-unassigned={!assigned}
                  onmouseenter={() => dna.hoveredLayer = { parentId: tower.isOffspring ? '__offspring__' : tower.id, layerIndex: i }}
                  onmouseleave={() => dna.hoveredLayer = null}
                  onclick={() => handleSlabClick(tower, i)}
                >
                  <polygon
                    points={poly.right}
                    fill={assigned ? slabColor(rgb, 0.35) : 'var(--border-dim)'}
                    stroke={assigned ? slabColor(rgb, 0.25) : 'var(--border-dim)'}
                    stroke-width="0.5"
                    opacity={assigned ? 1 : 0.3}
                  />
                  <polygon
                    points={poly.front}
                    fill={assigned ? slabColor(rgb, 0.55) : 'var(--border-dim)'}
                    stroke={assigned ? slabColor(rgb, 0.4) : 'var(--border-dim)'}
                    stroke-width="0.5"
                    opacity={assigned ? 1 : 0.3}
                  />
                  <polygon
                    points={poly.top}
                    fill={assigned ? slabColor(rgb, 0.85) : 'var(--border-dim)'}
                    stroke={assigned ? slabColor(rgb, 0.65) : 'var(--border-dim)'}
                    stroke-width="0.5"
                    opacity={assigned ? 1 : 0.3}
                  />
                  {#if assigned}
                    <polygon points={poly.top} fill="url(#dna-dots)" />
                  {/if}
                </g>
              {/each}
              <!-- Tower label below base -->
              {@const labelPos = isoXY(tower.originX + tower.W / 2, tower.D + 8, -6)}
              <text
                x={labelPos[0]}
                y={labelPos[1]}
                class="iso-tower-label"
                text-anchor="middle"
                dominant-baseline="hanging"
              >{tower.label}</text>
              <!-- Layer count above top -->
              {@const topZ = Math.min(tower.layerCount, 64) * tower.step + tower.H}
              {@const countPos = isoXY(tower.originX + tower.W / 2, tower.D / 2, topZ + 8)}
              <text
                x={countPos[0]}
                y={countPos[1]}
                class="iso-tower-count"
                text-anchor="middle"
                dominant-baseline="auto"
              >{tower.layerCount}L</text>
            {/each}
          </svg>
        </div>

        <!-- Footer: zoom controls + legend -->
        <div class="iso-footer">
          <div class="iso-controls">
            <button class="zoom-btn" onclick={() => viewZoom = Math.min(4, viewZoom + 0.2)}>+</button>
            <button class="zoom-btn" onclick={() => viewZoom = Math.max(0.3, viewZoom - 0.2)}>−</button>
            <button class="zoom-btn zoom-reset" onclick={resetView}>⟲</button>
            <span class="label-xs" style="color: var(--text-muted);">{Math.round(viewZoom * 100)}%</span>
            <div style="flex: 1;"></div>
            <span class="label-xs" style="color: var(--text-muted);">{dna.maxLayers} LAYERS</span>
          </div>
          {#if dna.categories.length > 0 && dna.isAnalyzed}
            <div class="iso-legend">
              {#each dna.categories as cat}
                <div class="iso-legend-item">
                  <span class="iso-legend-dot" style="background: {cat.color};"></span>
                  <span class="label-xs">{cat.label}</span>
                </div>
              {/each}
            </div>
          {:else}
            <div class="iso-legend">
              {#each dna.parents as parent, i}
                <div class="iso-legend-item">
                  <span class="iso-legend-dot" style="background: {parent.color};"></span>
                  <span class="label-xs">P{i + 1}</span>
                </div>
              {/each}
              <div class="iso-legend-sep"></div>
              <div class="iso-legend-item">
                <span class="iso-legend-dot" style="background: #3b82f6;"></span>
                <span class="label-xs">ATTN</span>
              </div>
              <div class="iso-legend-item">
                <span class="iso-legend-dot" style="background: #22c55e;"></span>
                <span class="label-xs">MLP</span>
              </div>
            </div>
          {/if}
        </div>

      {:else}
        <div class="iso-empty">
          <span class="label-xs" style="color: var(--text-muted);">LOAD PARENT MODELS TO VISUALIZE</span>
        </div>
      {/if}
    </div>

    <!-- ── Right: Tabbed Panel ────────────────────────── -->
    <div class="dna-tabs-col">
      <div class="tab-bar">
        {#each [["files", "FILES"], ["layers", "LAYERS"], ["settings", "SETTINGS"]] as [key, label]}
          <button
            class="tab-btn"
            class:tab-active={dna.activeTab === key}
            onclick={() => dna.activeTab = key as DnaTab}
          >
            {label}
          </button>
        {/each}
      </div>

      <div class="tab-body">
        <!-- ── Tab: FILES ──────────────────────────────── -->
        {#if dna.activeTab === "files"}
          <div class="tab-content">
            <div class="parent-grid">
              {#each dna.parents as parent, i (parent.id)}
                <div class="parent-card panel-inset" style="--gene-color: {parent.color}">
                  <div class="parent-header">
                    <div style="flex: 1;">
                      <div class="parent-id-row">
                        <span class="label-xs">PARENT {i + 1}</span>
                        <span class="dot dot-active"></span>
                      </div>
                      <div class="parent-name" title={parent.name}>
                        {parent.name.length > 22 ? parent.name.slice(0, 22) + '...' : parent.name}
                      </div>
                    </div>
                    <div class="gene-dot" style="background: {parent.color};"></div>
                  </div>
                  <div class="parent-info">
                    <div class="info-row"><span class="label-xs">PARAMS</span><span class="code-sm">{parent.parameter_count_display}</span></div>
                    <div class="info-row"><span class="label-xs">LAYERS</span><span class="code-sm">{parent.layer_count ?? '---'}</span></div>
                    <div class="info-row"><span class="label-xs">FORMAT</span><span class="badge badge-sm">{parent.format === 'safe_tensors' ? 'ST' : 'GGUF'}</span></div>
                    <div class="info-row"><span class="label-xs">SIZE</span><span class="code-sm">{parent.file_size_display}</span></div>
                    <div class="info-row"><span class="label-xs">TENSORS</span><span class="code-sm">{parent.tensor_count}</span></div>
                  </div>
                  <div class="parent-actions">
                    {#if currentMethod?.requires_base}
                      <button
                        class="btn btn-xs"
                        class:btn-accent={dna.baseParentId === parent.id}
                        onclick={() => dna.baseParentId = dna.baseParentId === parent.id ? null : parent.id}
                      >
                        {dna.baseParentId === parent.id ? 'BASE' : 'SET BASE'}
                      </button>
                    {/if}
                    <button class="btn btn-xs btn-danger-ghost" onclick={() => dna.removeParent(parent.id)}>REMOVE</button>
                  </div>
                </div>
              {/each}

              {#if dna.parents.length < 5}
                <div class="add-card panel-inset">
                  <button class="btn btn-sm" onclick={() => dna.addParent()}>+ LOAD FILE</button>
                  <button class="btn btn-sm btn-secondary" onclick={() => dna.addParentDir()}>+ LOAD DIR</button>
                </div>
              {/if}
            </div>

            <!-- Compatibility -->
            {#if dna.compatReport}
              <div class="compat-section">
                <span class="divider-label">COMPATIBILITY</span>
                <div class="compat-grid">
                  <div class="compat-item">
                    <span class="dot" class:dot-active={dna.compatReport.architecture_match}></span>
                    <span class="label-xs">ARCH</span>
                  </div>
                  <div class="compat-item">
                    <span class="dot" class:dot-active={dna.compatReport.dimension_match}></span>
                    <span class="label-xs">DIM</span>
                  </div>
                  <div class="compat-item">
                    <span class="dot" class:dot-active={dna.compatReport.layer_count_match}></span>
                    <span class="label-xs">LAYERS</span>
                  </div>
                  <div class="compat-item">
                    <span class="label-xs">SHARED</span>
                    <span class="code-sm">{dna.compatReport.shared_tensor_count}/{dna.compatReport.total_tensor_count}</span>
                  </div>
                </div>
                {#each dna.compatReport.errors as err}
                  <div class="compat-error"><span class="label-xs danger-text">{err}</span></div>
                {/each}
                {#each dna.compatReport.warnings as warn}
                  <div class="compat-warn"><span class="label-xs" style="color: var(--accent)">{warn}</span></div>
                {/each}

                <!-- Dimension details -->
                {#if dna.compatReport.dimension_details.length > 0}
                  <div class="dim-section" style="margin-top: 8px;">
                    <span class="label-xs" style="color: var(--text-muted); margin-bottom: 4px;">DIMENSION ANALYSIS</span>
                    <div class="dim-grid">
                      {#each dna.compatReport.dimension_details as dim}
                        <div class="dim-item" class:dim-error={dim.severity === "error"} class:dim-warn={dim.severity === "warning"}>
                          <span class="label-xs">{dim.dimension_name.toUpperCase().replace("_", " ")}</span>
                          <div class="dim-values">
                            {#each dim.values as [name, val]}
                              <span class="code-sm" title={name}>{val.toLocaleString()}</span>
                            {/each}
                          </div>
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}

                <!-- Resolution strategies -->
                {#if dna.compatReport.resolution_strategies.length > 0}
                  <div class="strat-section" style="margin-top: 6px;">
                    <span class="label-xs" style="color: var(--text-muted); margin-bottom: 4px;">RESOLUTION STRATEGIES</span>
                    {#each dna.compatReport.resolution_strategies as strat}
                      <div class="strat-item">
                        <div class="strat-header">
                          <span class="badge" class:badge-accent={strat.quality_estimate === "high"} class:badge-dim={strat.quality_estimate !== "high"}>{strat.name.toUpperCase().replace("_", " ")}</span>
                          <span class="label-xs" style="color: var(--text-muted);">{strat.quality_estimate.toUpperCase()}</span>
                        </div>
                        <span class="label-xs" style="color: var(--text-secondary);">{strat.description}</span>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}

            <!-- Capability filter -->
            {#if dna.allDetectedCapabilities.length > 0}
              <div class="cap-filter-section">
                <span class="divider-label">CAPABILITY FILTER</span>
                <div class="cap-toggle-grid">
                  {#each dna.allDetectedCapabilities as cap}
                    {@const enabled = dna.capabilityToggles[cap.id] ?? true}
                    <button
                      class="cap-toggle"
                      class:cap-on={enabled}
                      class:cap-off={!enabled}
                      onclick={() => dna.toggleCapability(cap.id)}
                      title="{cap.name}: {cap.affected_layers.length} layers, {(cap.confidence * 100).toFixed(0)}% confidence"
                    >
                      <span class="cap-toggle-name">{cap.name}</span>
                      <span class="cap-toggle-meta">{cap.affected_layers.length}L / {(cap.confidence * 100).toFixed(0)}%</span>
                    </button>
                  {/each}
                </div>
                {#if dna.disabledLayers.length > 0}
                  <div class="cap-impact">
                    <span class="label-xs" style="color: var(--danger);">{dna.disabledLayers.length} LAYERS EXCLUDED</span>
                  </div>
                {/if}
              </div>
            {/if}

            <!-- Composition -->
            {#if dna.layerAssignments.length > 0}
              <div class="comp-section">
                <span class="divider-label">COMPOSITION</span>
                <div class="composition-bar">
                  {#each dna.compositionStats as stat}
                    {#if stat.percentage > 0}
                      <div class="comp-seg" style="width: {stat.percentage}%; background: {stat.color};" title="{stat.name}: {stat.percentage.toFixed(0)}%"></div>
                    {/if}
                  {/each}
                </div>
                <div class="comp-legend">
                  {#each dna.compositionStats.filter(s => s.percentage > 0) as stat}
                    <div class="legend-item">
                      <span class="legend-dot" style="background: {stat.color};"></span>
                      <span class="label-xs">{stat.name.slice(0, 14)} {stat.percentage.toFixed(0)}%</span>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </div>

        <!-- ── Tab: LAYERS ─────────────────────────────── -->
        {:else if dna.activeTab === "layers"}
          <div class="tab-content">
            {#if dna.parents.length >= 2}
              <div class="layer-actions">
                <button class="btn btn-xs" onclick={() => dna.autoAssign('split')}>AUTO: SPLIT</button>
                <button class="btn btn-xs" onclick={() => dna.autoAssign('interleave')}>AUTO: INTERLEAVE</button>
                {#if dna.hasGaps}
                  <span class="label-xs" style="color: var(--accent); margin-left: auto;">GAPS DETECTED</span>
                {/if}
              </div>
            {/if}

            <!-- Layer assignment list -->
            <div class="layer-list">
              {#each Array(Math.min(dna.maxLayers, 64)) as _, idx}
                {@const assignment = dna.layerAssignments.find(a => a.layerIndex === idx)}
                {@const assignedParent = assignment ? dna.parents.find(p => p.id === assignment.sourceParentId) : null}
                {@const firstParent = dna.parents[0]}
                {@const analysis = firstParent ? dna.getLayerAnalysis(firstParent.id, idx) : null}
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <div
                  class="layer-row"
                  class:layer-row-empty={!assignment}
                  class:layer-row-highlight={dna.hoveredLayer?.layerIndex === idx}
                  onmouseenter={(e) => { mouseX = e.clientX; mouseY = e.clientY; dna.hoveredLayer = { parentId: assignedParent?.id ?? firstParent?.id ?? null, layerIndex: idx }; }}
                  onmouseleave={() => dna.hoveredLayer = null}
                >
                  <span class="layer-idx-label">{idx}</span>
                  {#if analysis}
                    <span class="layer-cat-badge" style="background: {analysis.color}20; color: {analysis.color}; border-color: {analysis.color}50;">
                      {analysis.label}
                    </span>
                  {/if}
                  <div class="layer-source">
                    {#if assignedParent}
                      <span class="layer-dot" style="background: {assignedParent.color};"></span>
                      <span class="code-sm">{assignedParent.name.slice(0, 18)}</span>
                    {:else}
                      <span class="code-sm" style="color: var(--text-muted);">---</span>
                    {/if}
                  </div>
                  <div class="layer-pick">
                    {#each dna.parents as p}
                      <button
                        class="layer-pick-btn"
                        class:layer-pick-active={assignment?.sourceParentId === p.id}
                        style="background: {p.color}; opacity: {assignment?.sourceParentId === p.id ? 1 : 0.25};"
                        onclick={() => dna.assignLayer(idx, p.id)}
                        title="Assign to {p.name}"
                      ></button>
                    {/each}
                  </div>
                </div>
              {/each}
            </div>

            <!-- Profile cards -->
            {#if dna.profiles.length > 0}
              <div style="margin-top: 12px;">
                <span class="divider-label">PROFILING RESULTS</span>
              </div>
              <div class="profile-strip">
                {#each dna.profiles as profile}
                  {@const badge = getSpecBadge(profile.specialization)}
                  <div class="profile-card panel-inset">
                    <div class="profile-hdr">
                      <span class="label-xs">L{profile.layer_index}</span>
                      <span class="badge badge-sm" style="background: {badge.color}20; color: {badge.color}; border-color: {badge.color}40;">
                        {badge.label}
                      </span>
                    </div>
                    <div class="profile-bar"><div class="profile-fill" style="width: {profile.confidence * 100}%; background: {badge.color};"></div></div>
                    <span class="code-sm">{profile.top_predictions[0]?.token?.slice(0, 22) ?? '---'}</span>
                    <span class="label-xs" style="color: var(--text-muted);">H={profile.entropy.toFixed(2)}</span>
                  </div>
                {/each}
              </div>
            {/if}
          </div>

        <!-- ── Tab: SETTINGS ───────────────────────────── -->
        {:else if dna.activeTab === "settings"}
          <div class="tab-content">
            <!-- Presets -->
            <span class="divider-label">PRESETS</span>
            <div class="preset-grid">
              {#each MERGE_PRESETS as preset}
                <button
                  class="preset-card"
                  class:preset-active={dna.activePresetId === preset.id}
                  onclick={() => dna.applyPreset(preset)}
                >
                  <span class="preset-name">{preset.name}</span>
                  <span class="preset-desc">{preset.desc}</span>
                </button>
              {/each}
            </div>

            <!-- Mode -->
            <span class="divider-label" style="margin-top: 12px;">MODE</span>
            <div class="mode-bar">
              {#each ["easy", "intermediate", "advanced"] as m}
                <button class="mode-btn" class:mode-active={dna.mode === m} onclick={() => dna.mode = m as DnaMode}>
                  {m.toUpperCase()}
                </button>
              {/each}
            </div>

            <!-- Method -->
            <span class="divider-label" style="margin-top: 12px;">MERGE METHOD</span>
            <div class="method-grid">
              {#each methodsForMode as method}
                <button
                  class="method-btn"
                  class:method-active={dna.selectedMethod === method.id}
                  onclick={() => dna.selectedMethod = method.id}
                >
                  <span class="method-name">{method.name}</span>
                  <span class="method-desc">{method.description.slice(0, 40)}</span>
                  <span class="method-badge">{method.difficulty.toUpperCase().slice(0, 3)}</span>
                </button>
              {/each}
            </div>

            <!-- Params -->
            {#if currentMethod}
              <div class="param-section">
                <span class="label-xs">PARAMETERS — {currentMethod.name}</span>

                {#if dna.selectedMethod === 'slerp'}
                  <div class="param-row">
                    <span class="label-xs">INTERPOLATION T</span>
                    <input type="range" min="0" max="1" step="0.05" value={dna.methodParams.t ?? 0.5}
                      oninput={(e) => dna.methodParams = { ...dna.methodParams, t: parseFloat(e.currentTarget.value) }}
                      class="range-input" />
                    <span class="code-sm">{(dna.methodParams.t ?? 0.5).toFixed(2)}</span>
                  </div>
                {/if}
                {#if dna.selectedMethod === 'task_arithmetic'}
                  <div class="param-row">
                    <span class="label-xs">SCALING</span>
                    <input type="range" min="0" max="2" step="0.1" value={dna.methodParams.scaling ?? 1.0}
                      oninput={(e) => dna.methodParams = { ...dna.methodParams, scaling: parseFloat(e.currentTarget.value) }}
                      class="range-input" />
                    <span class="code-sm">{(dna.methodParams.scaling ?? 1.0).toFixed(1)}</span>
                  </div>
                {/if}
                {#if dna.selectedMethod === 'dare'}
                  <div class="param-row">
                    <span class="label-xs">DENSITY</span>
                    <input type="range" min="0.1" max="1" step="0.05" value={dna.methodParams.density ?? 0.5}
                      oninput={(e) => dna.methodParams = { ...dna.methodParams, density: parseFloat(e.currentTarget.value) }}
                      class="range-input" />
                    <span class="code-sm">{(dna.methodParams.density ?? 0.5).toFixed(2)}</span>
                  </div>
                {/if}
                {#if dna.selectedMethod === 'ties'}
                  <div class="param-row">
                    <span class="label-xs">TRIM</span>
                    <input type="range" min="0" max="0.5" step="0.05" value={dna.methodParams.trim_threshold ?? 0.2}
                      oninput={(e) => dna.methodParams = { ...dna.methodParams, trim_threshold: parseFloat(e.currentTarget.value) }}
                      class="range-input" />
                    <span class="code-sm">{(dna.methodParams.trim_threshold ?? 0.2).toFixed(2)}</span>
                  </div>
                {/if}
                {#if dna.selectedMethod === 'della'}
                  <div class="param-row">
                    <span class="label-xs">DENSITY</span>
                    <input type="range" min="0.1" max="1" step="0.05" value={dna.methodParams.della_density ?? 0.7}
                      oninput={(e) => dna.methodParams = { ...dna.methodParams, della_density: parseFloat(e.currentTarget.value) }}
                      class="range-input" />
                    <span class="code-sm">{(dna.methodParams.della_density ?? 0.7).toFixed(2)}</span>
                  </div>
                  <div class="param-row">
                    <span class="label-xs">LAMBDA</span>
                    <input type="range" min="0" max="2" step="0.1" value={dna.methodParams.lambda ?? 1.0}
                      oninput={(e) => dna.methodParams = { ...dna.methodParams, lambda: parseFloat(e.currentTarget.value) }}
                      class="range-input" />
                    <span class="code-sm">{(dna.methodParams.lambda ?? 1.0).toFixed(1)}</span>
                  </div>
                {/if}
                {#if dna.selectedMethod === 'moe_conversion'}
                  <div class="param-row">
                    <span class="label-xs">EXPERTS</span>
                    <span class="code-sm">{dna.parents.length} (auto)</span>
                  </div>
                {/if}
              </div>
            {/if}

            <!-- Output -->
            <span class="divider-label" style="margin-top: 12px;">OUTPUT</span>
            <div class="output-config">
              <div class="param-row">
                <span class="label-xs">FORMAT</span>
                <div style="display: flex; gap: 4px;">
                  <button class="btn btn-xs" class:btn-accent={dna.outputFormat === 'safe_tensors'} onclick={() => dna.outputFormat = 'safe_tensors'}>ST</button>
                  <button class="btn btn-xs" class:btn-accent={dna.outputFormat === 'gguf'} onclick={() => dna.outputFormat = 'gguf'}>GGUF</button>
                </div>
              </div>
              <div class="param-row">
                <span class="label-xs">NAME</span>
                <input type="text" class="input-sm" bind:value={dna.modelName} placeholder="merged-model" />
              </div>
              <button class="btn btn-sm btn-secondary" onclick={() => dna.selectOutputPath()}>
                {dna.outputPath ? dna.outputPath.split('/').pop() : 'SELECT OUTPUT PATH'}
              </button>
            </div>

            <!-- Performance -->
            <span class="divider-label" style="margin-top: 12px;">PERFORMANCE</span>
            <div class="param-section">
              <div class="param-row">
                <span class="label-xs">BATCH SIZE</span>
                <input type="range" min="1" max="16" step="1" value={dna.mergeBatchSize}
                  oninput={(e) => dna.mergeBatchSize = parseInt(e.currentTarget.value)}
                  class="range-input" />
                <span class="code-sm">{dna.mergeBatchSize}</span>
              </div>
              <span class="label-xs" style="color: var(--text-muted);">TENSORS TO PROCESS CONCURRENTLY. HIGHER = FASTER, MORE RAM.</span>
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>

  <!-- ── Bottom: Actions + Progress ──────────────────── -->
  <div class="action-panel panel">
    <div class="action-buttons">
      <button
        class="btn btn-compute"
        disabled={!dna.canAnalyze || dna.analyzing}
        onclick={() => dna.analyzeLayers()}
      >
        {#if dna.analyzing}
          COMPUTING...
        {:else if dna.isAnalyzed}
          RE-COMPUTE LAYERS
        {:else}
          COMPUTE LAYERS
        {/if}
      </button>
      <button class="btn btn-secondary" disabled={dna.parents.length < 2} onclick={() => dna.getPreview()}>
        PREVIEW
      </button>
      <button class="btn btn-accent" disabled={!dna.canMerge || dna.merging} onclick={() => dna.merge()}>
        {dna.merging ? 'MERGING...' : 'BUILD MERGE'}
      </button>
      {#if dna.merging}
        <button class="btn btn-danger-ghost" onclick={() => dna.cancelMerge()}>CANCEL</button>
      {/if}
      {#if dna.analyzing}
        <button class="btn btn-danger-ghost" onclick={() => dna.cancelAnalysis()}>CANCEL</button>
      {/if}
    </div>

    {#if dna.analysisProgress}
      <div class="analysis-progress">
        <div class="progress-header">
          <span class="label-xs">{dna.analysisProgress.stage.toUpperCase()} — LAYER {dna.analysisProgress.layer_index + 1}/{dna.analysisProgress.total_layers}</span>
          <span class="code-sm">{dna.analysisProgress.percent.toFixed(0)}%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width: {dna.analysisProgress.percent}%; background: #a78bfa;"></div></div>
      </div>
    {/if}

    {#if dna.preview}
      <div class="preview-row">
        <span class="label-xs">OPS: {dna.preview.total_operations} ({dna.preview.merge_operations} merge, {dna.preview.copy_operations} copy)</span>
        <span class="label-xs">EST: {dna.preview.estimated_output_display}</span>
      </div>
    {/if}

    {#if dna.error}
      <div class="error-banner"><span class="danger-text">{dna.error}</span></div>
    {/if}
  </div>

  <!-- ── Progress ────────────────────────────────────── -->
  {#if dna.mergeProgress}
    <div class="progress-panel panel">
      <div class="progress-header">
        <span class="label-xs">{dna.mergeProgress.stage.toUpperCase()}</span>
        <span class="code-sm">{dna.mergeProgress.percent.toFixed(1)}%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width: {dna.mergeProgress.percent}%;"></div></div>
      <div class="progress-detail">
        <span class="code-sm">{dna.mergeProgress.message}</span>
        {#if dna.mergeProgress.current_tensor}
          <span class="label-xs" style="color: var(--text-muted);">{dna.mergeProgress.current_tensor}</span>
        {/if}
        <span class="label-xs">{dna.mergeProgress.tensors_done}/{dna.mergeProgress.tensors_total} tensors</span>
      </div>
    </div>
  {/if}

  <!-- ── Merge Result ────────────────────────────────── -->
  {#if dna.mergeResult}
    <div class="result-panel panel">
      <div class="result-header">
        <span class="label-xs" style="color: var(--success);">MERGE COMPLETE</span>
        <span class="badge badge-sm" style="background: var(--on-success); color: var(--success);">{dna.mergeResult.method}</span>
      </div>
      <div class="info-row"><span class="label-xs">OUTPUT</span><span class="code-sm">{dna.mergeResult.output_path.split('/').pop()}</span></div>
      <div class="info-row"><span class="label-xs">SIZE</span><span class="code-sm">{dna.mergeResult.output_size_display}</span></div>
      <div class="info-row"><span class="label-xs">TENSORS</span><span class="code-sm">{dna.mergeResult.tensors_written}</span></div>
      {#if dna.mergeResult.copied_files && dna.mergeResult.copied_files.length > 0}
        <div class="info-row"><span class="label-xs">COPIED</span><span class="code-sm">{dna.mergeResult.copied_files.join(', ')}</span></div>
      {/if}
    </div>
  {/if}

  <!-- ── Global floating tooltip (follows mouse everywhere) ── -->
  {#if dna.hoveredLayer && (tooltipData || tooltipAnalysis)}
    {@const hov = dna.hoveredLayer}
    {@const td = tooltipData}
    {@const ta = tooltipAnalysis}
    {@const tp = tooltipParent}
    <div class="iso-tooltip" style="left: {mouseX + 16}px; top: {mouseY - 16}px;">
      <div class="iso-tooltip-bar" style="background: {ta?.color ?? tp?.color ?? 'var(--accent)'};"></div>
      <div class="iso-tooltip-header">
        <span class="iso-tooltip-type">{tp ? tp.name.slice(0, 14) : 'OFFSPRING'}</span>
        <span class="heading-sm">LAYER {hov.layerIndex}</span>
      </div>
      <div class="iso-tooltip-body">
        {#if ta}
          <div class="iso-tooltip-cat" style="background: {ta.color}20; border-color: {ta.color}60;">
            <span class="iso-tooltip-cat-label" style="color: {ta.color};">{ta.label}</span>
            <span class="iso-tooltip-cat-desc">{ta.description}</span>
          </div>
          <div class="iso-tooltip-row">
            <span class="label-xs">CONFIDENCE</span>
            <span class="code" style="color: {ta.color};">{(ta.confidence * 100).toFixed(0)}%</span>
          </div>
          <div class="iso-tooltip-sep"></div>
        {/if}
        {#if td}
          <div class="iso-tooltip-row">
            <span class="label-xs">TENSORS</span>
            <span class="code">{td.totalTensors}</span>
          </div>
          <div class="iso-tooltip-sep"></div>
          <div class="iso-tooltip-row">
            <span class="label-xs" style="color: var(--info);">ATTENTION</span>
            <span class="code">{td.attn.length}</span>
          </div>
          <div class="iso-tooltip-row">
            <span class="label-xs" style="color: var(--success);">MLP</span>
            <span class="code">{td.mlp.length}</span>
          </div>
          {#if td.norm.length > 0}
            <div class="iso-tooltip-row">
              <span class="label-xs" style="color: var(--gray);">NORMS</span>
              <span class="code">{td.norm.length}</span>
            </div>
          {/if}
        {/if}
        {#if ta}
          <div class="iso-tooltip-sep"></div>
          <div class="iso-tooltip-row">
            <span class="label-xs">MLP DOM.</span>
            <span class="code">{(ta.mlp_dominance * 100).toFixed(0)}%</span>
          </div>
          <div class="iso-tooltip-row">
            <span class="label-xs">NORM L2</span>
            <span class="code">{ta.norm_l2.toFixed(2)}</span>
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .dna-page {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  /* ── Hero ──────────────────────────────────── */
  .hero { padding: 20px; }
  .hero-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
  .hero-id { display: flex; gap: 12px; }
  .hero-body { margin-bottom: 16px; }
  .hero-title { font-size: 28px; font-weight: 800; letter-spacing: 0.22em; color: var(--text-primary); line-height: 1; }
  .hero-subtitle { font-size: 10px; font-weight: 500; letter-spacing: 0.18em; color: var(--text-secondary); margin-top: 6px; text-transform: uppercase; }
  .hero-specs { display: grid; grid-template-columns: repeat(6, 1fr); border: 1px solid var(--border-dim); }
  .spec-cell { display: flex; flex-direction: column; gap: 4px; padding: 8px 10px; border-right: 1px solid var(--border-dim); }
  .spec-cell:last-child { border-right: none; }
  .spec-value { font-size: 12px; font-weight: 600; letter-spacing: 0.06em; color: var(--text-primary); }
  .hero-footer { display: flex; align-items: center; justify-content: space-between; margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--border-dim); }

  /* ── 2-Column Grid ─────────────────────────── */
  .dna-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    min-height: 420px;
  }

  /* ── Isometric Viz ─────────────────────────── */
  .dna-viz {
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
    min-height: 340px;
    padding: 20px 16px 12px;
  }

  .iso-svg { width: 100%; height: 100%; max-height: 400px; }

  .iso-slab { cursor: pointer; transition: opacity 200ms ease; }
  .iso-dimmed { opacity: 0.15; }
  .iso-unassigned { opacity: 0.25; cursor: default; }

  .iso-tower-label {
    font-size: 7px;
    font-weight: 700;
    fill: var(--text-muted);
    letter-spacing: 0.15em;
    font-family: var(--font-mono);
    text-transform: uppercase;
  }
  .iso-tower-count {
    font-size: 5px;
    font-weight: 600;
    fill: var(--text-muted);
    letter-spacing: 0.1em;
    font-family: var(--font-mono);
    opacity: 0.6;
  }

  .iso-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 300px;
  }

  .iso-footer { padding: 8px 16px; border-top: 1px solid var(--border-dim); display: flex; flex-direction: column; gap: 6px; }
  .iso-controls { display: flex; align-items: center; gap: 4px; }
  .zoom-btn {
    width: 22px; height: 22px; display: flex; align-items: center; justify-content: center;
    background: var(--bg-inset); border: 1px solid var(--border); color: var(--text-secondary);
    font-family: var(--font-mono); font-size: 12px; cursor: pointer; padding: 0;
    transition: all var(--transition);
  }
  .zoom-btn:hover { background: var(--bg-hover); color: var(--text-primary); border-color: var(--accent); }
  .zoom-reset { font-size: 11px; }
  .iso-grabbing, .iso-grabbing * { cursor: grabbing !important; }
  .iso-viewport { cursor: grab; }
  .iso-legend { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .iso-legend-item { display: flex; align-items: center; gap: 5px; }
  .iso-legend-dot { width: 8px; height: 8px; flex-shrink: 0; }
  .iso-legend-sep { width: 1px; height: 12px; background: var(--border-dim); }

  /* ── Floating Tooltip (from inspect) ───────── */
  .iso-tooltip {
    position: fixed;
    z-index: 1000;
    pointer-events: none;
    min-width: 160px;
    max-width: 240px;
    background: var(--bg-inset);
    border: 2px solid var(--accent);
    clip-path: polygon(
      0 4px, 4px 4px, 4px 0,
      calc(100% - 4px) 0, calc(100% - 4px) 4px, 100% 4px,
      100% calc(100% - 4px), calc(100% - 4px) calc(100% - 4px), calc(100% - 4px) 100%,
      4px 100%, 4px calc(100% - 4px), 0 calc(100% - 4px)
    );
  }
  .iso-tooltip-bar { height: 3px; }
  .iso-tooltip-header { display: flex; align-items: center; gap: 8px; padding: 6px 10px 4px; }
  .iso-tooltip-type { font-size: 8px; font-weight: 700; letter-spacing: 0.1em; color: var(--text-muted); padding: 1px 4px; border: 1px solid var(--border-dim); }
  .iso-tooltip-body { display: flex; flex-direction: column; gap: 3px; padding: 4px 10px 8px; }
  .iso-tooltip-row { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
  .iso-tooltip-row .code { font-size: 10px; color: var(--text-primary); }
  .iso-tooltip-sep { height: 1px; background: var(--border-dim); margin: 2px 0; }
  .iso-tooltip-cat {
    padding: 4px 6px;
    border: 1px solid;
    margin-bottom: 2px;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }
  .iso-tooltip-cat-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.12em;
    font-family: var(--font-mono);
  }
  .iso-tooltip-cat-desc {
    font-size: 7px;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
    font-family: var(--font-mono);
  }

  /* ── Tabs ───────────────────────────────────── */
  .dna-tabs-col {
    display: flex;
    flex-direction: column;
    min-height: 0;
    border: 1px solid var(--border);
    background: var(--bg-surface);
  }

  .tab-bar { display: flex; border-bottom: 1px solid var(--border); }
  .tab-btn {
    flex: 1;
    padding: 8px;
    background: var(--bg-inset);
    border: none;
    border-right: 1px solid var(--border);
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.12em;
    cursor: pointer;
    transition: all var(--transition);
  }
  .tab-btn:last-child { border-right: none; }
  .tab-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
  .tab-active {
    background: var(--bg-surface);
    color: var(--accent);
    box-shadow: inset 0 -2px 0 var(--accent);
  }

  .tab-body {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
  }

  .tab-content {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  /* ── Files Tab ─────────────────────────────── */
  .parent-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 8px;
  }
  .parent-card {
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    border-left: 3px solid var(--gene-color, var(--border));
  }
  .add-card {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: center;
    justify-content: center;
    border: 1px dashed var(--border);
    min-height: 100px;
  }
  .parent-header { display: flex; align-items: flex-start; gap: 6px; padding-bottom: 4px; border-bottom: 1px solid var(--border-dim); }
  .parent-id-row { display: flex; align-items: center; gap: 5px; margin-bottom: 2px; }
  .parent-name { font-size: 10px; font-weight: 700; letter-spacing: 0.06em; color: var(--text-primary); }
  .gene-dot { width: 12px; height: 12px; flex-shrink: 0; }
  .parent-info { display: flex; flex-direction: column; gap: 2px; }
  .info-row { display: flex; justify-content: space-between; align-items: center; }
  .code-sm { font-size: 10px; font-weight: 500; color: var(--text-secondary); letter-spacing: 0.04em; }
  .parent-actions { display: flex; gap: 4px; margin-top: auto; }

  .compat-section, .comp-section { display: flex; flex-direction: column; gap: 6px; }
  .compat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
  .compat-item { display: flex; align-items: center; gap: 5px; }
  .compat-error { padding: 4px 8px; background: var(--on-danger); border: 1px solid var(--danger); }
  .compat-warn { padding: 4px 8px; background: var(--on-accent); border: 1px solid var(--accent-dim); }

  .composition-bar { display: flex; height: 14px; border: 1px solid var(--border-dim); overflow: hidden; }
  .comp-seg { transition: width 200ms ease; min-width: 2px; }
  .comp-legend { display: flex; flex-wrap: wrap; gap: 6px; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot { width: 8px; height: 8px; flex-shrink: 0; }

  /* ── Layers Tab ────────────────────────────── */
  .layer-actions { display: flex; gap: 6px; align-items: center; padding-bottom: 6px; border-bottom: 1px solid var(--border-dim); }
  .layer-list { display: flex; flex-direction: column; gap: 1px; max-height: 360px; overflow-y: auto; }
  .layer-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 6px;
    background: var(--bg-inset);
    transition: background 80ms ease;
  }
  .layer-row:hover, .layer-row-highlight { background: var(--bg-hover); }
  .layer-row-empty .layer-source { opacity: 0.4; }
  .layer-idx-label { font-size: 8px; font-weight: 600; color: var(--text-muted); width: 22px; text-align: right; font-family: var(--font-mono); }
  .layer-source { display: flex; align-items: center; gap: 5px; flex: 1; min-width: 0; }
  .layer-dot { width: 8px; height: 8px; flex-shrink: 0; }
  .layer-pick { display: flex; gap: 3px; }
  .layer-pick-btn {
    width: 10px;
    height: 10px;
    border: 1px solid var(--border);
    cursor: pointer;
    padding: 0;
    transition: opacity 80ms ease, transform 80ms ease;
  }
  .layer-pick-btn:hover { opacity: 1 !important; transform: scale(1.3); }
  .layer-pick-active { outline: 1px solid var(--text-primary); outline-offset: 1px; }

  .layer-cat-badge {
    font-size: 7px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 1px 4px;
    border: 1px solid;
    font-family: var(--font-mono);
    flex-shrink: 0;
    min-width: 36px;
    text-align: center;
  }

  .profile-strip { display: flex; gap: 5px; overflow-x: auto; padding-bottom: 4px; }
  .profile-card { min-width: 85px; max-width: 85px; padding: 6px; display: flex; flex-direction: column; gap: 3px; flex-shrink: 0; }
  .profile-hdr { display: flex; justify-content: space-between; align-items: center; }
  .profile-bar { height: 3px; background: var(--border-dim); overflow: hidden; }
  .profile-fill { height: 100%; transition: width 200ms ease; }

  /* ── Settings Tab ──────────────────────────── */
  .mode-bar { display: flex; border: 1px solid var(--border); }
  .mode-btn {
    flex: 1; padding: 6px; background: var(--bg-inset); border: none; border-right: 1px solid var(--border);
    color: var(--text-secondary); font-family: var(--font-mono); font-size: 9px; font-weight: 600;
    letter-spacing: 0.12em; cursor: pointer; transition: all var(--transition);
  }
  .mode-btn:last-child { border-right: none; }
  .mode-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
  .mode-active { background: var(--on-accent); color: var(--accent); box-shadow: inset 0 -2px 0 var(--accent); }

  .preset-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 4px; }
  .preset-card {
    padding: 8px 6px; background: var(--bg-inset); border: 1px solid var(--border); color: var(--text-primary);
    cursor: pointer; transition: all var(--transition); display: flex; flex-direction: column; gap: 3px;
    align-items: center; text-align: center;
  }
  .preset-card:hover { border-color: var(--accent); background: var(--bg-hover); }
  .preset-active { border-color: var(--accent); background: var(--on-accent); box-shadow: inset 0 -2px 0 var(--accent); }
  .preset-name { font-size: 9px; font-weight: 700; letter-spacing: 0.08em; font-family: var(--font-mono); }
  .preset-desc { font-size: 7px; color: var(--text-muted); letter-spacing: 0.02em; line-height: 1.3; font-family: var(--font-mono); }

  .method-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 4px; }
  .method-btn {
    padding: 7px 5px; background: var(--bg-inset); border: 1px solid var(--border); color: var(--text-primary);
    cursor: pointer; transition: all var(--transition); display: flex; flex-direction: column; gap: 2px;
    align-items: center; text-align: center; position: relative;
  }
  .method-btn:hover { border-color: var(--accent); background: var(--bg-hover); }
  .method-active { border-color: var(--accent); background: var(--on-accent); box-shadow: inset 0 -2px 0 var(--accent); }
  .method-name { font-size: 9px; font-weight: 700; letter-spacing: 0.08em; font-family: var(--font-mono); }
  .method-desc { font-size: 7px; color: var(--text-muted); letter-spacing: 0.02em; line-height: 1.3; font-family: var(--font-mono); }
  .method-badge { position: absolute; top: 2px; right: 2px; font-size: 6px; padding: 1px 3px; color: var(--text-muted); border: 1px solid var(--border-dim); font-family: var(--font-mono); }

  .param-section { display: flex; flex-direction: column; gap: 8px; padding-top: 8px; border-top: 1px solid var(--border-dim); }
  .param-row { display: flex; align-items: center; gap: 8px; }
  .param-row .label-xs { min-width: 80px; }
  .range-input {
    flex: 1; height: 4px; -webkit-appearance: none; appearance: none; background: var(--border-dim); outline: none;
  }
  .range-input::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px; background: var(--accent); cursor: pointer; }
  .range-input::-moz-range-thumb { width: 12px; height: 12px; background: var(--accent); cursor: pointer; border: none; }

  .output-config { display: flex; flex-direction: column; gap: 8px; }
  .input-sm { flex: 1; padding: 4px 8px; background: var(--bg-inset); border: 1px solid var(--border); color: var(--text-primary); font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.04em; }
  .input-sm:focus { outline: none; border-color: var(--accent); }

  /* ── Common ────────────────────────────────── */
  .btn-xs { padding: 3px 6px; font-size: 8px; font-weight: 600; letter-spacing: 0.1em; }
  .btn-danger-ghost { background: transparent; color: var(--danger); border: 1px solid var(--danger); cursor: pointer; font-family: var(--font-mono); text-transform: uppercase; }
  .btn-danger-ghost:hover { background: var(--on-danger); }
  .badge-danger { background: var(--on-danger); color: var(--danger); border-color: var(--danger); }
  .badge-sm { padding: 1px 5px; font-size: 8px; font-weight: 600; letter-spacing: 0.08em; border: 1px solid var(--border); }

  /* ── Action Panel ──────────────────────────── */
  .action-panel { padding: 12px; display: flex; flex-direction: column; gap: 8px; }
  .action-buttons { display: flex; gap: 8px; flex-wrap: wrap; }
  .btn-compute {
    background: #a78bfa20;
    color: #a78bfa;
    border: 1px solid #a78bfa60;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 6px 14px;
    cursor: pointer;
    text-transform: uppercase;
    transition: all var(--transition);
  }
  .btn-compute:hover:not(:disabled) { background: #a78bfa30; border-color: #a78bfa; }
  .btn-compute:disabled { opacity: 0.4; cursor: not-allowed; }
  .analysis-progress { display: flex; flex-direction: column; gap: 4px; }
  .preview-row { display: flex; gap: 16px; padding-top: 6px; border-top: 1px solid var(--border-dim); }
  .error-banner { padding: 6px 12px; background: var(--on-danger); border: 1px solid var(--danger); text-align: center; }

  /* ── Progress ──────────────────────────────── */
  .progress-panel { padding: 12px; display: flex; flex-direction: column; gap: 6px; }
  .progress-header { display: flex; justify-content: space-between; align-items: center; }
  .progress-bar { height: 5px; background: var(--border-dim); overflow: hidden; }
  .progress-fill { height: 100%; background: var(--accent); transition: width 200ms ease; }
  .progress-detail { display: flex; flex-direction: column; gap: 2px; }

  /* ── Result ────────────────────────────────── */
  .result-panel { padding: 12px; display: flex; flex-direction: column; gap: 6px; border: 1px solid var(--success); }
  .result-header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 6px; border-bottom: 1px solid var(--border-dim); }

  /* ── Dimension Analysis ────────────────────── */
  .dim-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 4px;
  }
  .dim-item {
    padding: 4px 6px;
    border: 1px solid var(--border-dim);
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .dim-error { border-color: var(--danger); }
  .dim-warn { border-color: var(--accent); }
  .dim-values {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  .strat-item {
    padding: 4px 6px;
    border: 1px solid var(--border-dim);
    margin-bottom: 3px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .strat-header {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .badge-accent { background: var(--accent); color: var(--bg-primary); }

  /* ── Capability filter ──────────────────────── */
  .cap-filter-section {
    margin-top: 6px;
  }
  .cap-toggle-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 4px;
    margin-top: 4px;
  }
  .cap-toggle {
    display: flex;
    flex-direction: column;
    padding: 5px 8px;
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    cursor: pointer;
    text-align: left;
    transition: all 0.15s ease;
    font-family: var(--font-mono);
  }
  .cap-toggle:hover { border-color: var(--accent); }
  .cap-on { border-color: var(--accent); }
  .cap-on .cap-toggle-name { color: var(--accent); }
  .cap-off { opacity: 0.5; }
  .cap-off .cap-toggle-name { text-decoration: line-through; color: var(--text-muted); }
  .cap-toggle-name {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-primary);
  }
  .cap-toggle-meta {
    font-size: 8px;
    color: var(--text-muted);
    letter-spacing: 0.05em;
  }
  .cap-impact {
    margin-top: 4px;
    padding: 3px 6px;
    border: 1px solid var(--danger);
  }

  /* ── Responsive ────────────────────────────── */
  @media (max-width: 900px) {
    .dna-grid { grid-template-columns: 1fr; }
    .hero-specs { grid-template-columns: repeat(3, 1fr); }
    .action-buttons { flex-direction: column; }
  }
</style>
