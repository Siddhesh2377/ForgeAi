# ForgeAI

A cross-platform desktop tool for loading, inspecting, optimizing, and exporting AI models — built for developers who run models locally.

> Android Studio, but for AI models.

---

## Problem

Running AI models locally means juggling CLI tools, manual quantization scripts, opaque file formats, and no easy way to compare before/after performance. There is no unified workbench for the local-AI workflow.

## Solution

ForgeAI provides a single desktop app where you can:

1. **Load** a model from disk (ONNX, GGUF, SafeTensors)
2. **Inspect** its architecture, layers, metadata, and tensor shapes
3. **Optimize** it with one-click quantization (INT8 / INT4)
4. **Benchmark** size, latency, and memory usage before and after
5. **Visualize** tokenizer behavior and token flow
6. **Export** the optimized model ready for deployment
7. **Merge** models with genetic engineering techniques (SLERP, TIES, DARE, LoRA)

All offline. No cloud. No accounts.

---

## Tech Stack

| Layer         | Technology                | Role                                   |
|---------------|---------------------------|----------------------------------------|
| Shell         | Tauri v2                  | Lightweight native desktop window      |
| Frontend      | SvelteKit 5 (Svelte 5)   | Reactive UI with runes (`$state`)      |
| Backend       | Rust                      | Model loading, optimization, benchmarks|
| Async         | Tokio                     | Non-blocking task execution            |
| Formats       | ONNX / GGML / SafeTensors | Model file parsing and export          |
| Serialization | Serde                     | Rust <-> Frontend data exchange        |

---

## Architecture

```
┌──────────────────────────────────────┐
│            SvelteKit UI              │
│  (panels, visualizers, controls)     │
│                                      │
│  ┌──────────┐  ┌──────────────────┐  │
│  │ Settings │  │ M-DNA Forge      │  │
│  │ (07)     │  │ (08) merge/LoRA  │  │
│  └──────────┘  └──────────────────┘  │
└──────────────┬───────────────────────┘
               │ Tauri IPC (invoke/events)
┌──────────────▼───────────────────────┐
│           Rust Core                  │
│                                      │
│  ┌────────────┐ ┌──────────────────┐ │
│  │ Model      │ │ Optimizer        │ │
│  │ Loader     │ │ Engine           │ │
│  └────────────┘ └──────────────────┘ │
│  ┌────────────┐ ┌──────────────────┐ │
│  │ Benchmark  │ │ Tokenizer        │ │
│  │ Runner     │ │ Visualizer       │ │
│  └────────────┘ └──────────────────┘ │
│  ┌────────────┐ ┌──────────────────┐ │
│  │ Export     │ │ M-DNA Merge      │ │
│  │ Pipeline   │ │ Engine           │ │
│  └────────────┘ └──────────────────┘ │
└──────────────────────────────────────┘
```

---

## Roadmap

Development is organized into milestones. Each milestone has concrete checkpoints that must all pass before moving to the next.

### Milestone 0 — Project Scaffold *(complete)*

Set up the project skeleton and verify the build pipeline works end-to-end.

- [x] Initialize Tauri v2 + SvelteKit + Rust project
- [x] Verify `tauri dev` boots the app window
- [x] Replace default boilerplate UI with ForgeAI shell layout (header, sidebar, statusbar)
- [x] Set up dark/light theme toggle (CSS custom properties)
- [x] Add sidebar navigation skeleton (Load, Inspect, Optimize, Benchmark, Tokenize, Export, M-DNA, Settings)
- [x] Industrial label design system (monospace fonts, corner marks, barcode decorations)
- [ ] Confirm cross-platform build (Linux `.AppImage`, macOS `.dmg`, Windows `.exe`)

### Milestone 1 — Model Loader

Load model files from disk and surface basic metadata.

- [ ] File picker dialog (drag-and-drop + browse) via Tauri file dialog
- [ ] Rust: parse ONNX protobuf and extract graph metadata
- [ ] Rust: parse GGUF header and extract model metadata
- [ ] Rust: parse SafeTensors header and extract tensor info
- [ ] Display loaded model info in the UI (name, format, size, layer count, parameter count)
- [ ] Error handling: clear messages for unsupported or corrupt files

### Milestone 2 — Model Inspector

Visualize the internal structure of a loaded model.

- [ ] Render layer list with type, shape, and parameter count per layer
- [ ] Expandable tensor detail view (dtype, dimensions, memory footprint)
- [ ] Search/filter layers by name or type
- [ ] Display model-level metadata (author, license, training config if available)
- [ ] Layer activation probing and logit lens visualization

### Milestone 3 — Quantization & Optimization

Reduce model size and improve inference speed.

- [ ] Rust: INT8 quantization for ONNX models
- [ ] Rust: INT4 quantization (GPTQ / AWQ style) for GGUF models
- [ ] Progress reporting from Rust to UI via Tauri events
- [ ] Side-by-side comparison view: original vs. quantized (size, dtype distribution)
- [ ] Save optimized model to disk

### Milestone 4 — Benchmark Runner

Measure real performance differences.

- [ ] Rust: measure model file size on disk
- [ ] Rust: measure peak memory usage during dummy inference
- [ ] Rust: measure inference latency (avg over N runs, with warmup)
- [ ] UI: benchmark results table with before/after columns
- [ ] UI: bar chart visualization for latency and memory comparison

### Milestone 5 — Tokenizer Visualizer

Help users understand how text is processed.

- [ ] Rust: load tokenizer from model or standalone tokenizer file
- [ ] Tokenize input text and return token IDs + token strings
- [ ] UI: interactive text input with color-coded token spans
- [ ] Display token count, vocabulary size, and special tokens

### Milestone 6 — Export Pipeline

Package optimized models for deployment targets.

- [ ] Export quantized ONNX with metadata preserved
- [ ] Export GGUF with selected quantization level
- [ ] Bundle model + tokenizer into a single output directory
- [ ] Configurable export: choose target format, precision, and output path

### Milestone 7 — M-DNA Forge

Genetic model engineering — merge, splice, and evolve models.

- [x] UI: M-DNA Forge page with parent model grid, merge controls, DNA visualization
- [ ] Multi-model merge methods: AVERAGE, SLERP, TIES, DARE
- [ ] LoRA adapter loading and merge (strength slider, enable/disable per adapter)
- [ ] Layer-selective merge: choose weight ranges per parent model (attention, MLP, embeddings)
- [ ] Gene color visualization: map layer types to DNA strand display
- [ ] Weight validation: ensure parent model weights total 100%
- [ ] Rust: execute SLERP interpolation across model tensors
- [ ] Rust: execute TIES (trim, elect, merge) with density threshold
- [ ] Rust: execute DARE (drop and rescale) with probability control
- [ ] Rust: LoRA merge into base model with configurable alpha
- [ ] Progress reporting during merge operations
- [ ] Export merged model to SafeTensors / GGUF

### Future (Post-MVP)

Not in scope for the initial release, but planned:

- GPU / NPU acceleration for benchmarks and inference
- Plugin system for community-contributed optimizers and visualizers
- Advanced compression (pruning, distillation, knowledge transfer)
- Model diffing (compare two model versions)
- Layer Inspector: activation probing, logit lens, causal tracing
- Mobile companion app for on-device testing

---

## Constraints

These apply for the entire MVP and should not be relaxed:

- **Local only** — no network calls, no telemetry, no cloud
- **CPU first** — GPU support is deferred to post-MVP
- **Offline** — the app must work without an internet connection

---

## Design Principles

- **Industrial label aesthetic** — monospace fonts, 0px border-radius, 1px borders, corner marks, barcode decorations, dense grid layouts
- **Color semantics** — colors represent application state, not user preference:

  | Color  | Variable    | Meaning             |
  |--------|-------------|---------------------|
  | Amber  | `--accent`  | Idle / Default      |
  | Blue   | `--info`    | Working / Processing|
  | Green  | `--success` | Success / Complete  |
  | Red    | `--danger`  | Danger / Error      |
  | Gray   | `--gray`    | Paused / Inactive   |

- **Keyboard-first** — all primary actions accessible without a mouse
- **Real-time feedback** — progress bars and streaming logs for long operations
- **Fail clearly** — every error state has a user-readable message and suggested action

---

## Development

### Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [Node.js](https://nodejs.org/) (v18+)
- [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/) for your OS

### Run locally

```bash
npm install
npm run tauri dev
```

### Build for production

```bash
npm run tauri build
```

---

## Status

**Milestone 0 — complete.** Shell layout, theme toggle, sidebar navigation, settings page, and industrial design system are implemented. M-DNA Forge UI (Milestone 7) is in progress. Next up: Milestone 1 (Model Loader).

---

## License

MIT
