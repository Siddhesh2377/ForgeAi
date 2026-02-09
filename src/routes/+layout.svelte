<script lang="ts">
  import "../app.css";
  import { page } from "$app/stores";
  import { theme } from "$lib/theme.svelte";
  import { model } from "$lib/model.svelte";
  import { dna } from "$lib/dna.svelte";

  let { children } = $props();

  const modules = [
    { code: "00", name: "DASHBOARD", href: "/" },
    { code: "01", name: "LOAD", href: "/load" },
    { code: "02", name: "INSPECT", href: "/inspect" },
    { code: "03", name: "OPTIMIZE", href: "/optimize" },
    { code: "04", name: "HUB", href: "/hub" },
    { code: "05", name: "CONVERT", href: "/convert" },
    { code: "08", name: "M-DNA", href: "/dna" },
    { code: "09", name: "TEST", href: "/test" },
  ];

  // Global task state for sidebar progress
  const taskState = $derived.by(() => {
    if (dna.merging) return { active: true, label: "MERGING", percent: dna.mergeProgress?.percent ?? 0, color: "info" } as const;
    if (dna.analyzing) return { active: true, label: "ANALYZING", percent: dna.analysisProgress?.percent ?? 0, color: "info" } as const;
    if (dna.profiling) return { active: true, label: "PROFILING", percent: 0, color: "info" } as const;
    if (dna.status === "complete") return { active: false, label: "COMPLETE", percent: 100, color: "success" } as const;
    if (dna.status === "error") return { active: false, label: "ERROR", percent: 100, color: "danger" } as const;
    if (model.status === "loading") return { active: true, label: "LOADING", percent: 0, color: "info" } as const;
    return { active: false, label: "", percent: 0, color: "accent" } as const;
  });

  function getTimestamp() {
    return new Date().toISOString().slice(0, 10).replace(/-/g, ".");
  }

  function isActive(href: string | null) {
    if (!href) return false;
    return $page.url.pathname === href;
  }
</script>

<div class="app" data-theme={theme.mode}>
  <!-- ── Header ──────────────────────────────────── -->
  <header class="header">
    <div class="header-left">
      <span class="header-brand">FORGEAI</span>
      <span class="header-sep"></span>
      <span class="label-xs">v0.1.0</span>
      <span class="header-sep"></span>
      <span class="label-xs">{getTimestamp()}</span>
    </div>
    <div class="header-right">
      <span class="badge badge-accent">
        <span class="dot dot-active"></span>
        SYS:READY
      </span>
      <button class="btn btn-ghost" onclick={() => theme.toggleMode()}>
        {theme.mode === "dark" ? "LIGHT" : "DARK"}
      </button>
    </div>
  </header>

  <!-- ── Body ────────────────────────────────────── -->
  <div class="body">
    <!-- Sidebar -->
    <nav class="sidebar">
      <div class="sidebar-label">
        <span class="divider-label">MODULES</span>
      </div>

      {#each modules as mod}
        {#if mod.href}
          <a
            class="nav-item"
            class:active={isActive(mod.href)}
            href={mod.href}
          >
            <span class="nav-code">{mod.code}</span>
            <span class="nav-name">{mod.name}</span>
            {#if isActive(mod.href)}
              <span class="nav-indicator"></span>
            {/if}
          </a>
        {:else}
          <button class="nav-item nav-disabled" disabled>
            <span class="nav-code">{mod.code}</span>
            <span class="nav-name">{mod.name}</span>
          </button>
        {/if}
      {/each}

      <div class="sidebar-spacer"></div>

      <div class="sidebar-label">
        <span class="divider-label">SYSTEM</span>
      </div>
      <a
        class="nav-item"
        class:active={$page.url.pathname === "/settings"}
        href="/settings"
      >
        <span class="nav-code">07</span>
        <span class="nav-name">SETTINGS</span>
        {#if $page.url.pathname === "/settings"}
          <span class="nav-indicator"></span>
        {/if}
      </a>

      <!-- Global task progress -->
      {#if taskState.label}
        <div class="sidebar-progress">
          <div class="sidebar-progress-header">
            <span class="dot" class:dot-working={taskState.active} class:dot-success={taskState.color === "success"} class:dot-danger={taskState.color === "danger"}></span>
            <span class="label-xs">{taskState.label}</span>
            {#if taskState.active}
              <span class="label-xs" style="margin-left:auto; color: var(--{taskState.color});">{Math.round(taskState.percent)}%</span>
            {/if}
          </div>
          <div class="sidebar-progress-bar">
            <div
              class="sidebar-progress-fill"
              style="width: {taskState.percent}%; background: var(--{taskState.color});"
              class:sidebar-progress-pulse={taskState.active && taskState.percent === 0}
            ></div>
          </div>
        </div>
      {/if}

      <!-- Sidebar barcode -->
      <div class="sidebar-footer">
        <div class="barcode"></div>
        <span class="label-xs" style="margin-top: 4px;">FRG-001-{getTimestamp()}</span>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="main" class:main-centered={theme.layout === "centered"}>
      {@render children()}
    </main>
  </div>

  <!-- ── Status Bar ──────────────────────────────── -->
  <footer class="statusbar" class:statusbar-working={taskState.active} class:statusbar-success={taskState.color === "success"} class:statusbar-error={taskState.color === "danger"}>
    <div class="statusbar-left">
      {#if taskState.active}
        <span class="dot dot-working"></span>
        <span style="color: var(--info);">{taskState.label}</span>
        <span class="statusbar-sep">|</span>
        <span style="color: var(--info);">{Math.round(taskState.percent)}%</span>
      {:else if model.status === "loading"}
        <span class="dot dot-working"></span>
        <span>LOADING</span>
      {:else if model.status === "loaded"}
        <span class="dot dot-success"></span>
        <span>READY</span>
      {:else if model.status === "error"}
        <span class="dot dot-danger"></span>
        <span>ERROR</span>
      {:else}
        <span class="dot dot-active"></span>
        <span>IDLE</span>
      {/if}
      <span class="statusbar-sep">|</span>
      <span>MODEL: {model.info ? model.info.file_name : "NONE"}</span>
      {#if model.info}
        <span class="statusbar-sep">|</span>
        <span>{model.formatDisplay}</span>
        <span class="statusbar-sep">|</span>
        <span>{model.info.parameter_count_display}</span>
      {/if}
    </div>
    <div class="statusbar-right">
      <span>CPU</span>
      <span class="statusbar-sep">|</span>
      <span>MEM: --</span>
      <span class="statusbar-sep">|</span>
      <span>FORGEAI v0.1.0</span>
    </div>
  </footer>
</div>

<style>
  /* ── App Shell ─────────────────────────────────── */
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    background: var(--bg-base);
    transition: background-color var(--theme-transition);
  }

  /* ── Header ────────────────────────────────────── */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 36px;
    padding: 0 12px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    user-select: none;
    transition:
      background-color var(--theme-transition),
      border-color var(--theme-transition);
  }

  .header-left,
  .header-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-brand {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.16em;
    color: var(--accent);
    transition: color var(--theme-transition);
  }

  .header-sep {
    width: 1px;
    height: 12px;
    background: var(--border);
    transition: background-color var(--theme-transition);
  }

  /* ── Body ──────────────────────────────────────── */
  .body {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── Sidebar ───────────────────────────────────── */
  .sidebar {
    display: flex;
    flex-direction: column;
    width: 160px;
    background: var(--bg-surface);
    border-right: 1px solid var(--border);
    flex-shrink: 0;
    overflow-y: auto;
    padding: 8px 0;
    user-select: none;
    transition:
      background-color var(--theme-transition),
      border-color var(--theme-transition);
  }

  .sidebar-label {
    padding: 8px 12px 4px;
  }

  .sidebar-spacer {
    flex: 1;
    min-height: 16px;
  }

  .sidebar-footer {
    padding: 12px 12px 8px;
    border-top: 1px solid var(--border-dim);
    margin-top: 8px;
  }

  /* ── Nav Item ──────────────────────────────────── */
  .nav-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 6px 12px;
    background: none;
    border: none;
    border-left: 2px solid transparent;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition);
    text-align: left;
    text-decoration: none;
    position: relative;
  }

  .nav-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .nav-item.active {
    border-left-color: var(--accent);
    color: var(--text-primary);
    background: var(--accent-bg);
  }

  .nav-disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }

  .nav-disabled:hover {
    background: none;
    color: var(--text-secondary);
  }

  .nav-code {
    font-size: 9px;
    color: var(--text-muted);
    font-weight: 600;
    min-width: 16px;
  }

  .nav-item.active .nav-code {
    color: var(--accent);
  }

  .nav-name {
    flex: 1;
  }

  .nav-indicator {
    width: 4px;
    height: 4px;
    background: var(--accent);
    transition: background-color var(--theme-transition);
  }

  /* ── Main Content ──────────────────────────────── */
  .main {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    background: var(--bg-base);
    transition: background-color var(--theme-transition);
  }

  .main-centered > :global(*) {
    max-width: 960px;
    margin-left: auto;
    margin-right: auto;
  }

  /* ── Status Bar ────────────────────────────────── */
  .statusbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 24px;
    padding: 0 12px;
    background: var(--bg-surface);
    border-top: 1px solid var(--border);
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    flex-shrink: 0;
    user-select: none;
    transition:
      background-color var(--theme-transition),
      border-color var(--theme-transition),
      color var(--theme-transition);
  }

  .statusbar-left,
  .statusbar-right {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .statusbar-sep {
    color: var(--border);
  }

  /* Status bar state colors */
  .statusbar-working {
    border-top-color: var(--info);
  }

  .statusbar-success {
    border-top-color: var(--success);
  }

  .statusbar-error {
    border-top-color: var(--danger);
  }

  /* ── Sidebar Progress ──────────────────────────── */
  .sidebar-progress {
    padding: 8px 12px;
    border-top: 1px solid var(--border-dim);
    margin-top: 4px;
  }

  .sidebar-progress-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
  }

  .sidebar-progress-bar {
    height: 3px;
    background: var(--bg-inset);
    width: 100%;
    overflow: hidden;
  }

  .sidebar-progress-fill {
    height: 100%;
    transition: width 300ms ease;
    image-rendering: pixelated;
  }

  .sidebar-progress-pulse {
    width: 100% !important;
    animation: progress-pulse 1.5s ease-in-out infinite;
    opacity: 0.6;
  }

  @keyframes progress-pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.8; }
  }
</style>
