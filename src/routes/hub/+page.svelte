<script lang="ts">
  import { goto } from "$app/navigation";
  import { open } from "@tauri-apps/plugin-dialog";
  import { hub, type HfFileInfo } from "$lib/hub.svelte";
  import { model } from "$lib/model.svelte";

  type View = "search" | "library";

  let view = $state<View>("search");
  let searchInput = $state("");
  let modelsOnly = $state(true);

  // Load library on mount
  $effect(() => {
    hub.loadLibrary();
  });

  function handleFetch() {
    const trimmed = searchInput.trim();
    if (!trimmed) return;
    hub.fetchRepo(trimmed);
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter") handleFetch();
  }

  function handleDownload(file: HfFileInfo) {
    if (!hub.repoInfo || hub.downloading) return;
    hub.downloadFile(hub.repoInfo.id, file.rfilename);
  }

  function handleDownloadRepo() {
    if (!hub.repoInfo || hub.downloading) return;
    hub.downloadRepo(hub.repoInfo.id);
  }

  async function handleLoad(filePath: string) {
    await model.load(filePath);
    goto("/load");
  }

  let displayedFiles = $derived.by(() => {
    if (!hub.repoInfo) return [];
    if (modelsOnly) return hub.repoInfo.files.filter((f) => f.format !== null);
    return hub.repoInfo.files;
  });

  let modelFileCount = $derived(
    hub.repoInfo?.files.filter((f) => f.format !== null).length ?? 0
  );

  let isRepoDownload = $derived(
    hub.downloadProgress?.files_total !== null && hub.downloadProgress?.files_total !== undefined
  );

  function formatDate(iso: string): string {
    try {
      return new Date(iso).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch {
      return iso;
    }
  }

  let importError = $state<string | null>(null);
  let importing = $state(false);

  async function importGgufFile() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "GGUF Models", extensions: ["gguf"] }],
    });
    if (selected) {
      const filePath = Array.isArray(selected) ? selected[0] : selected;
      if (filePath) await doImport(filePath);
    }
  }

  async function importSafetensorsFile() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "SafeTensors Models", extensions: ["safetensors"] }],
    });
    if (selected) {
      const filePath = Array.isArray(selected) ? selected[0] : selected;
      if (filePath) await doImport(filePath);
    }
  }

  async function importFolder() {
    const selected = await open({ directory: true, multiple: false });
    if (selected) {
      const dirPath = Array.isArray(selected) ? selected[0] : selected;
      if (dirPath) await doImport(dirPath);
    }
  }

  async function doImport(path: string) {
    importing = true;
    importError = null;
    try {
      await hub.importLocal(path);
    } catch (e) {
      importError = String(e);
    } finally {
      importing = false;
    }
  }

  function formatDownloaded(bytes: number): string {
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + " GB";
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + " MB";
    if (bytes >= 1024) return (bytes / 1024).toFixed(0) + " KB";
    return bytes + " B";
  }
</script>

<div class="hub fade-in">
  <!-- ── Hero Panel ──────────────────────────────── -->
  <div class="hero panel">
    <div class="hero-top">
      <span class="label-xs">FRG.04</span>
      <span class="label-xs" style="color: var(--text-muted);">HUB-ENGINE</span>
      <span class="badge badge-accent" style="margin-left: auto;">
        <span class="dot dot-active"></span>
        ONLINE
      </span>
    </div>

    <h1 class="hero-title">MODEL HUB</h1>
    <p class="hero-subtitle">Download &amp; manage models from HuggingFace</p>

    <div class="hero-specs">
      <div class="spec-cell">
        <span class="label-xs">LOCAL MODELS</span>
        <span class="spec-value">{hub.localModels.length}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">STORAGE</span>
        <span class="spec-value">{hub.totalStorageDisplay}</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">SOURCE</span>
        <span class="spec-value">HUGGINGFACE</span>
      </div>
      <div class="spec-cell">
        <span class="label-xs">FORMATS</span>
        <span class="spec-value">GGUF / ST</span>
      </div>
    </div>
  </div>

  <!-- ── View Toggle ─────────────────────────────── -->
  <div class="section">
    <div class="section-label">
      <span class="divider-label">BROWSE</span>
    </div>

    <div class="view-toggle">
      <button
        class="view-btn"
        class:view-btn-active={view === "search"}
        onclick={() => (view = "search")}
      >
        SEARCH
      </button>
      <button
        class="view-btn"
        class:view-btn-active={view === "library"}
        onclick={() => (view = "library")}
      >
        LIBRARY
      </button>
    </div>
  </div>

  <!-- ─── SEARCH VIEW ────────────────────────────── -->
  {#if view === "search"}
    <div class="section">
      <div class="section-label">
        <span class="divider-label">SEARCH REPOSITORY</span>
      </div>

      <div class="search-row">
        <input
          class="search-input"
          type="text"
          placeholder="owner/model-name"
          bind:value={searchInput}
          onkeydown={handleKeydown}
        />
        <button
          class="btn btn-accent"
          onclick={handleFetch}
          disabled={hub.repoLoading || !searchInput.trim()}
        >
          {hub.repoLoading ? "FETCHING..." : "FETCH"}
        </button>
      </div>
    </div>

    {#if hub.repoLoading}
      <div class="empty-state panel-flat">
        <span class="heading-sm" style="color: var(--info); animation: pulse 1.2s ease infinite;">
          FETCHING REPOSITORY...
        </span>
      </div>
    {:else if hub.repoError}
      <div class="empty-state panel-flat" style="border-color: var(--danger);">
        <div class="error-inner">
          <span class="dot dot-danger"></span>
          <span class="danger-text">{hub.repoError}</span>
        </div>
      </div>
    {:else if hub.repoInfo}
      <div class="section">
        <div class="section-label">
          <span class="divider-label">REPOSITORY FILES</span>
        </div>

        <div class="repo-header panel-flat">
          <div class="repo-header-left">
            <span class="heading-sm">{hub.repoInfo.id}</span>
            <span class="badge badge-dim">{hub.repoInfo.files.length} FILES</span>
            <span class="badge badge-accent">{modelFileCount} MODELS</span>
          </div>
          <div class="repo-header-actions">
            <button
              class="btn btn-sm btn-accent"
              onclick={handleDownloadRepo}
              disabled={hub.downloading}
            >
              DOWNLOAD REPO
            </button>
            <button
              class="btn btn-sm btn-secondary"
              onclick={() => (modelsOnly = !modelsOnly)}
            >
              {modelsOnly ? "SHOW ALL" : "MODELS ONLY"}
            </button>
          </div>
        </div>

        <div class="file-list panel-flat">
          {#each displayedFiles as file}
            <div class="file-row" class:file-row-model={file.format !== null}>
              <div class="file-name">
                <span class="file-name-text">{file.rfilename}</span>
              </div>
              <span class="file-size code">{file.size_display}</span>
              {#if file.format}
                <span class="badge badge-accent">{file.format.toUpperCase()}</span>
              {:else}
                <span class="badge badge-dim">OTHER</span>
              {/if}
              <button
                class="btn btn-sm {file.format ? 'btn-accent' : 'btn-secondary'}"
                onclick={() => handleDownload(file)}
                disabled={hub.downloading}
              >
                DOWNLOAD
              </button>
            </div>
          {:else}
            <div class="file-empty">
              <span class="label-xs">NO MODEL FILES FOUND</span>
            </div>
          {/each}
        </div>
      </div>

      <!-- Download Progress -->
      {#if hub.downloading && hub.downloadProgress}
        <div class="section">
          <div class="section-label">
            <span class="divider-label">DOWNLOAD PROGRESS</span>
          </div>

          <div class="progress-section panel">
            <div class="progress-header">
              <span class="heading-sm">{hub.downloadProgress.file_name}</span>
              <div class="progress-badges">
                {#if isRepoDownload && hub.downloadProgress.files_done !== null && hub.downloadProgress.files_total}
                  <span class="badge badge-dim">
                    {hub.downloadProgress.files_done}/{hub.downloadProgress.files_total} FILES
                  </span>
                {/if}
                <span class="badge badge-info">
                  <span class="dot dot-working" style="animation: pulse 1.2s ease infinite;"></span>
                  DOWNLOADING
                </span>
              </div>
            </div>

            <div class="progress-bar-row">
              <div class="progress-track">
                <div
                  class="progress-fill"
                  style="width: {hub.downloadProgress.percent}%;"
                ></div>
              </div>
            </div>

            <div class="progress-stats">
              <span class="code">{hub.downloadProgress.percent.toFixed(1)}%</span>
              <span class="label-xs">
                {formatDownloaded(hub.downloadProgress.bytes_downloaded)} / {formatDownloaded(hub.downloadProgress.bytes_total)}
              </span>
              <button class="btn btn-sm btn-danger" onclick={() => hub.cancelDownload()}>
                CANCEL
              </button>
            </div>
          </div>
        </div>
      {/if}

      {#if hub.downloadError}
        <div class="empty-state panel-flat" style="border-color: var(--danger);">
          <div class="error-inner">
            <span class="dot dot-danger"></span>
            <span class="danger-text">{hub.downloadError}</span>
          </div>
        </div>
      {/if}
    {/if}

  <!-- ─── LIBRARY VIEW ───────────────────────────── -->
  {:else}
    <!-- Import Section -->
    <div class="section">
      <div class="section-label">
        <span class="divider-label">IMPORT LOCAL MODEL</span>
      </div>

      <div class="import-grid">
        <button class="import-btn panel-flat" onclick={importGgufFile} disabled={importing}>
          <span class="import-btn-code">01</span>
          <span class="heading-sm">GGUF FILE</span>
          <span class="label-xs" style="color: var(--text-muted);">.gguf</span>
        </button>
        <button class="import-btn panel-flat" onclick={importSafetensorsFile} disabled={importing}>
          <span class="import-btn-code">02</span>
          <span class="heading-sm">SAFETENSORS FILE</span>
          <span class="label-xs" style="color: var(--text-muted);">.safetensors</span>
        </button>
        <button class="import-btn panel-flat" onclick={importFolder} disabled={importing}>
          <span class="import-btn-code">03</span>
          <span class="heading-sm">MODEL FOLDER</span>
          <span class="label-xs" style="color: var(--text-muted);">directory</span>
        </button>
      </div>

      {#if importing}
        <div class="import-status panel-flat">
          <span class="dot dot-working" style="animation: pulse 1.2s ease infinite;"></span>
          <span class="label-xs" style="color: var(--info);">COPYING TO LIBRARY...</span>
        </div>
      {/if}

      {#if importError}
        <div class="import-status panel-flat" style="border-color: var(--danger);">
          <span class="dot dot-danger"></span>
          <span class="danger-text">{importError}</span>
        </div>
      {/if}
    </div>

    <div class="section">
      <div class="section-label">
        <span class="divider-label">LOCAL MODELS</span>
      </div>

      {#if hub.libraryLoading}
        <div class="empty-state panel-flat">
          <span class="heading-sm" style="color: var(--info); animation: pulse 1.2s ease infinite;">
            LOADING LIBRARY...
          </span>
        </div>
      {:else if hub.localModels.length === 0}
        <div class="empty-state panel-flat" style="border-style: dashed;">
          <div class="empty-inner">
            <span class="heading-sm" style="color: var(--text-muted);">NO MODELS DOWNLOADED</span>
            <span class="label-xs" style="margin-top: 4px;">
              Use SEARCH to find and download models from HuggingFace
            </span>
            <button
              class="btn btn-accent"
              style="margin-top: 12px;"
              onclick={() => (view = "search")}
            >
              SEARCH MODELS
            </button>
          </div>
        </div>
      {:else}
        <div class="model-list">
          {#each hub.localModels as m}
            <div class="model-card panel-flat">
              <div class="model-card-header">
                <span class="heading-sm">{m.file_name}</span>
                <span class="badge {m.format === 'repo' ? 'badge-info' : 'badge-accent'}">{m.format.toUpperCase()}</span>
              </div>

              <div class="model-card-meta">
                <div class="model-meta-cell">
                  <span class="label-xs">SIZE</span>
                  <span class="code">{m.file_size_display}</span>
                </div>
                {#if m.source_repo}
                  <div class="model-meta-cell">
                    <span class="label-xs">SOURCE</span>
                    <span class="code" style="font-size: 9px;">{m.source_repo}</span>
                  </div>
                {/if}
                <div class="model-meta-cell">
                  <span class="label-xs">DOWNLOADED</span>
                  <span class="code" style="font-size: 9px;">{formatDate(m.downloaded_at)}</span>
                </div>
              </div>

              <div class="model-card-actions">
                {#if m.format !== "repo"}
                  <button class="btn btn-accent" onclick={() => handleLoad(m.file_path)}>
                    LOAD
                  </button>
                {:else}
                  <span class="label-xs" style="color: var(--text-muted); align-self: center;">
                    FULL REPO — USE CONVERT TO PRODUCE GGUF
                  </span>
                {/if}
                <button class="btn btn-sm btn-danger" onclick={() => hub.deleteModel(m.id)}>
                  DELETE
                </button>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <div class="section">
      <div class="section-label">
        <span class="divider-label">STORAGE</span>
      </div>

      <div class="storage-panel panel-flat">
        <div class="storage-grid">
          <div class="storage-cell">
            <span class="label-xs">MODELS</span>
            <span class="spec-value">{hub.localModels.length}</span>
          </div>
          <div class="storage-cell">
            <span class="label-xs">TOTAL SIZE</span>
            <span class="spec-value">{hub.totalStorageDisplay}</span>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  /* ── Page ──────────────────────────────────────── */
  .hub {
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

  /* ── View Toggle ───────────────────────────────── */
  .view-toggle {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .view-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    cursor: pointer;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-secondary);
    transition: all var(--transition);
  }

  .view-btn:hover {
    border-color: var(--border-strong);
    color: var(--text-primary);
  }

  .view-btn-active {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-bg);
  }

  .view-btn-active:hover {
    border-color: var(--accent);
  }

  /* ── Search ────────────────────────────────────── */
  .search-row {
    display: flex;
    gap: 8px;
  }

  .search-input {
    flex: 1;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    padding: 8px 12px;
    background: var(--bg-inset);
    border: 1px solid var(--border);
    color: var(--text-primary);
    transition: border-color var(--transition);
  }

  .search-input::placeholder {
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 10px;
  }

  .search-input:focus {
    outline: none;
    border-color: var(--accent);
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

  /* ── Repo Header ───────────────────────────────── */
  .repo-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    gap: 8px;
    flex-wrap: wrap;
  }

  .repo-header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .repo-header-actions {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  /* ── File List ─────────────────────────────────── */
  .file-list {
    padding: 0;
    overflow: hidden;
  }

  .file-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-dim);
    opacity: 0.5;
    transition: opacity var(--transition), background var(--transition);
  }

  .file-row:last-child {
    border-bottom: none;
  }

  .file-row:hover {
    background: var(--bg-hover);
    opacity: 0.8;
  }

  .file-row-model {
    opacity: 1;
  }

  .file-row-model:hover {
    opacity: 1;
  }

  .file-name {
    flex: 1;
    min-width: 0;
  }

  .file-name-text {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.03em;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: block;
  }

  .file-size {
    flex-shrink: 0;
    width: 80px;
    text-align: right;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .file-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
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

  .progress-badges {
    display: flex;
    align-items: center;
    gap: 6px;
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

  .progress-stats {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .progress-stats .code {
    font-size: 12px;
    font-weight: 700;
    color: var(--accent);
  }

  .progress-stats .label-xs {
    flex: 1;
  }

  /* ── Model Cards (Library) ─────────────────────── */
  .model-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .model-card {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .model-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .model-card-meta {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }

  .model-meta-cell {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .model-card-actions {
    display: flex;
    gap: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-dim);
  }

  /* ── Storage Panel ─────────────────────────────── */
  .storage-panel {
    padding: 12px;
  }

  .storage-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1px;
    background: var(--border-dim);
  }

  .storage-cell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    background: var(--bg-surface);
  }

  /* ── Import Grid ──────────────────────────────── */
  .import-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .import-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 14px 12px;
    cursor: pointer;
    transition: all var(--transition);
    font-family: var(--font-mono);
    text-align: center;
    border: 1px solid var(--border);
    background: var(--bg-surface);
  }

  .import-btn:hover:not(:disabled) {
    border-color: var(--accent);
    background: var(--accent-bg);
  }

  .import-btn:disabled {
    opacity: 0.35;
    cursor: wait;
  }

  .import-btn-code {
    font-size: 9px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.1em;
  }

  .import-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
  }

  /* ── Responsive ────────────────────────────────── */
  @media (max-width: 600px) {
    .hero-specs {
      grid-template-columns: repeat(2, 1fr);
    }

    .import-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
