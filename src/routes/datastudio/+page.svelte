<script lang="ts">
  import { onMount } from "svelte";
  import { goto } from "$app/navigation";
  import { datastudio, type ColumnAnalysis } from "$lib/datastudio.svelte";

  onMount(() => {
    if (datastudio.pendingPath) {
      const path = datastudio.pendingPath;
      datastudio.pendingPath = null;
      datastudio.loadDataset(path);
    }
  });

  function formatNumber(n: number): string {
    return n.toLocaleString();
  }

  function truncate(s: string, len: number): string {
    if (s.length <= len) return s;
    return s.slice(0, len) + "...";
  }

  function cellDisplay(val: any, maxLen = 80): string {
    if (val === null || val === undefined) return "";
    if (typeof val === "string") {
      return val.length > maxLen ? val.slice(0, maxLen) + "..." : val;
    }
    const s = JSON.stringify(val);
    return s.length > maxLen ? s.slice(0, maxLen) + "..." : s;
  }

  function dtypeLabel(dtype: string): string {
    return dtype.toUpperCase();
  }

  function handleHfFetch() {
    const q = datastudio.hfQuery.trim();
    if (q) datastudio.fetchHfDataset(q);
  }

  function handleHfKeydown(e: KeyboardEvent) {
    if (e.key === "Enter") handleHfFetch();
  }
</script>

<div class="ds-page">
  <!-- ── Hero Panel ── -->
  <div class="panel hero">
    <div class="hero-top">
      <div>
        <span class="label-xs" style="color: var(--text-muted);">MODULE 10</span>
        <h1 class="heading-lg">DATA STUDIO</h1>
        <p class="label" style="color: var(--text-secondary);">DATASET VIEWER & PREPARATION</p>
      </div>
      <div class="hero-actions">
        <button class="btn btn-ghost" onclick={() => goto("/training")}>TRAINING</button>
      </div>
    </div>
  </div>

  <!-- ── Source Toggle ── -->
  <div class="panel source-bar">
    <div class="source-toggle">
      <button
        class="source-btn"
        class:source-active={datastudio.source === "local"}
        onclick={() => datastudio.source = "local"}
      >LOCAL</button>
      <button
        class="source-btn"
        class:source-active={datastudio.source === "huggingface"}
        onclick={() => datastudio.source = "huggingface"}
      >HUGGINGFACE</button>
    </div>

    {#if datastudio.source === "local"}
      <button class="btn btn-accent" onclick={() => datastudio.browseDataset()}>BROWSE FILE...</button>
      {#if datastudio.dataset}
        <span class="code browse-path">{datastudio.dataset.path}</span>
      {/if}
      {#if datastudio.loading}
        <span class="label-xs" style="color: var(--info); margin-left: auto;">LOADING...</span>
      {/if}
    {:else}
      <div class="hf-search">
        <input
          class="input"
          type="text"
          placeholder="e.g. tatsu-lab/alpaca"
          bind:value={datastudio.hfQuery}
          onkeydown={handleHfKeydown}
        />
        <button
          class="btn btn-accent"
          onclick={handleHfFetch}
          disabled={datastudio.hfLoading || !datastudio.hfQuery.trim()}
        >{datastudio.hfLoading ? "FETCHING..." : "FETCH"}</button>
      </div>
    {/if}
  </div>

  <!-- ── HuggingFace Results ── -->
  {#if datastudio.source === "huggingface"}
    {#if datastudio.hfError}
      <div class="panel" style="border-color: var(--danger);">
        <div class="danger-text">{datastudio.hfError}</div>
      </div>
    {/if}

    {#if datastudio.hfDownloading && datastudio.hfDownloadProgress}
      <div class="panel hf-progress">
        <div class="hf-progress-header">
          <span class="dot dot-working"></span>
          <span class="label-xs" style="color: var(--info);">
            DOWNLOADING {datastudio.hfDownloadProgress.file_name}
          </span>
          <span class="label-xs" style="margin-left: auto; color: var(--info);">
            {Math.round(datastudio.hfDownloadProgress.percent)}%
          </span>
        </div>
        <div class="hf-progress-bar">
          <div class="hf-progress-fill" style="width: {datastudio.hfDownloadProgress.percent}%;"></div>
        </div>
      </div>
    {/if}

    {#if datastudio.hfRepo}
      <div class="panel">
        <div class="divider-label">
          {datastudio.hfRepo.id} — {datastudio.hfRepo.files.length} DATASET FILE{datastudio.hfRepo.files.length !== 1 ? "S" : ""}
        </div>
        {#if datastudio.hfRepo.files.length === 0}
          <div class="label-xs" style="padding: 12px; color: var(--text-muted);">
            NO DATASET FILES FOUND (JSON, JSONL, CSV, PARQUET)
          </div>
        {:else}
          <div class="hf-file-list">
            {#each datastudio.hfRepo.files as file}
              <div class="hf-file-row">
                <div class="hf-file-info">
                  <span class="code hf-file-name">{file.rfilename}</span>
                  <div class="hf-file-meta">
                    {#if file.format}
                      <span class="badge badge-accent">{file.format.toUpperCase()}</span>
                    {/if}
                    <span class="label-xs">{file.size_display}</span>
                  </div>
                </div>
                <button
                  class="btn btn-ghost"
                  disabled={datastudio.hfDownloading}
                  onclick={() => datastudio.downloadHfFile(datastudio.hfRepo!.id, file.rfilename)}
                >DOWNLOAD</button>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {:else if !datastudio.hfLoading && !datastudio.hfError}
      <div class="panel empty-state">
        <span class="label-xs" style="color: var(--text-muted);">ENTER A DATASET REPO ID TO SEARCH HUGGINGFACE</span>
      </div>
    {/if}
  {/if}

  <!-- ── Local Dataset View ── -->
  {#if datastudio.source === "local"}
    {#if datastudio.error}
      <div class="panel" style="border-color: var(--danger);">
        <div class="danger-text">{datastudio.error}</div>
      </div>
    {/if}

    {#if datastudio.dataset}
      <!-- ── Top Grid: Metadata + Column Analysis ── -->
      <div class="ds-top-grid">
        <!-- Metadata -->
        <div class="panel">
          <div class="divider-label">METADATA</div>
          <div class="meta-list">
            <div class="meta-row">
              <span class="label-xs">PATH</span>
              <span class="code meta-val" title={datastudio.dataset.path}>{truncate(datastudio.dataset.path, 50)}</span>
            </div>
            <div class="meta-row">
              <span class="label-xs">FORMAT</span>
              <span class="code meta-val">{datastudio.dataset.format.toUpperCase()}</span>
            </div>
            <div class="meta-row">
              <span class="label-xs">ROWS</span>
              <span class="code meta-val">{formatNumber(datastudio.dataset.rows)}</span>
            </div>
            <div class="meta-row">
              <span class="label-xs">SIZE</span>
              <span class="code meta-val">{datastudio.dataset.size_display}</span>
            </div>
            {#if datastudio.dataset.detected_template}
              <div class="meta-row">
                <span class="label-xs">TEMPLATE</span>
                <span class="badge badge-accent">{datastudio.dataset.detected_template.toUpperCase()}</span>
              </div>
            {/if}
            <div class="meta-row">
              <span class="label-xs">COLUMNS</span>
              <span class="code meta-val">{datastudio.dataset.columns.length}</span>
            </div>
          </div>
        </div>

        <!-- Column Analysis -->
        <div class="panel">
          <div class="divider-label">COLUMN ANALYSIS</div>
          <div class="col-table-wrap">
            <table class="col-table">
              <thead>
                <tr>
                  <th class="label-xs">COLUMN</th>
                  <th class="label-xs">TYPE</th>
                  <th class="label-xs">VALID</th>
                  <th class="label-xs">NULL</th>
                  <th class="label-xs">AVG LEN</th>
                </tr>
              </thead>
              <tbody>
                {#each datastudio.dataset.column_analysis as col}
                  <tr>
                    <td class="code">{col.name}</td>
                    <td class="code col-type">{dtypeLabel(col.dtype)}</td>
                    <td class="code">{formatNumber(col.non_null_count)}</td>
                    <td class="code" class:null-warn={col.null_count > 0}>{formatNumber(col.null_count)}</td>
                    <td class="code">{col.avg_length !== null ? `${Math.round(col.avg_length)} ch` : "--"}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- ── Data Preview ── -->
      <div class="panel">
        <div class="divider-label">DATA PREVIEW — {datastudio.dataset.preview.length} OF {formatNumber(datastudio.dataset.rows)} ROWS</div>
        <div class="data-preview-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th class="label-xs row-num">#</th>
                {#each datastudio.dataset.columns as col}
                  <th class="label-xs">{col.toUpperCase()}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each datastudio.dataset.preview as row, idx}
                <tr>
                  <td class="code row-num">{idx + 1}</td>
                  {#each datastudio.dataset.columns as col}
                    <td class="code">{cellDisplay(row[col])}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>

    {:else if !datastudio.loading && !datastudio.error}
      <div class="panel empty-state">
        <span class="label-xs" style="color: var(--text-muted);">SELECT A DATASET FILE TO BEGIN</span>
      </div>
    {/if}
  {/if}
</div>

<style>
  .ds-page {
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
  .hero-actions {
    flex-shrink: 0;
  }

  /* ── Source Toggle ── */
  .source-bar {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .source-toggle {
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
    flex-shrink: 0;
  }
  .source-btn {
    padding: 5px 12px;
    font-family: var(--font-mono);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.1em;
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition);
  }
  .source-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .source-active {
    background: var(--accent-bg);
    color: var(--accent);
  }

  .browse-path {
    font-size: 9px;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-secondary);
  }

  /* ── HF Search ── */
  .hf-search {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
  }
  .hf-search .input {
    flex: 1;
  }

  /* ── HF File List ── */
  .hf-file-list {
    display: flex;
    flex-direction: column;
  }
  .hf-file-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-dim);
  }
  .hf-file-row:last-child {
    border-bottom: none;
  }
  .hf-file-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .hf-file-name {
    font-size: 10px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .hf-file-meta {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ── HF Download Progress ── */
  .hf-progress {
    border-color: var(--info);
  }
  .hf-progress-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .hf-progress-bar {
    height: 3px;
    background: var(--bg-inset);
    overflow: hidden;
  }
  .hf-progress-fill {
    height: 100%;
    background: var(--info);
    transition: width 300ms ease;
  }

  /* ── Top Grid ── */
  .ds-top-grid {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 14px;
    align-items: start;
  }

  /* ── Metadata ── */
  .meta-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .meta-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 0;
    border-bottom: 1px solid var(--border-dim);
  }
  .meta-row:last-child {
    border-bottom: none;
  }
  .meta-val {
    margin-left: auto;
    font-size: 10px;
    color: var(--text-secondary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 180px;
  }

  /* ── Column Analysis ── */
  .col-table-wrap {
    overflow-x: auto;
  }
  .col-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 10px;
  }
  .col-table th {
    padding: 4px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border-dim);
    background: var(--bg-surface);
    position: sticky;
    top: 0;
  }
  .col-table td {
    padding: 3px 10px;
    border-bottom: 1px solid var(--border-dim);
  }
  .col-type {
    color: var(--accent);
    font-weight: 700;
    font-size: 9px;
  }
  .null-warn {
    color: var(--danger);
  }

  /* ── Data Preview ── */
  .data-preview-wrap {
    max-height: 600px;
    overflow: auto;
    border: 1px solid var(--border-dim);
    background: var(--bg-inset);
  }
  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9px;
  }
  .data-table th {
    padding: 5px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border-dim);
    background: var(--bg-surface);
    position: sticky;
    top: 0;
    z-index: 1;
    white-space: nowrap;
  }
  .data-table td {
    padding: 3px 10px;
    border-bottom: 1px solid var(--border-dim);
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-num {
    width: 36px;
    text-align: center;
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px;
  }

  @media (max-width: 700px) {
    .ds-top-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
