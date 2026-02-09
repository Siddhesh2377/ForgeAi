# ForgeAI

A local-first desktop tool for loading, inspecting, quantizing, merging, and testing AI models — entirely offline, entirely yours.

> Your local model workshop.

---

## What It Does

| Module | Code | What |
|--------|:----:|------|
| **Load** | 01 | Import GGUF files, SafeTensors files, or sharded HuggingFace model folders |
| **Inspect** | 02 | Memory breakdown, quantization distribution, isometric 3D architecture viz, runtime compatibility matrix, SHA-256 verification |
| **Optimize** | 03 | Quantize GGUF models across 7 levels (Q2_K → Q8_0) with real-time size/quality preview |
| **Hub** | 04 | Download models from HuggingFace, manage a local library |
| **Convert** | 05 | SafeTensors → GGUF conversion with configurable output types (F16, BF16, F32, Q8_0) |
| **M-DNA Forge** | 08 | Merge 2–5 models using SLERP, TIES, DARE, DeLLa, Frankenmerge, Task Arithmetic, Passthrough, or Average |
| **Test** | 09 | Run inference with real-time token streaming — llama.cpp (GGUF) or HuggingFace Transformers (SafeTensors) |
| **Settings** | 07 | Theme (dark/light), layout (stretched/centered), GPU detection, llama.cpp tools management |

All offline. No cloud. No accounts. No telemetry.

---

## Screenshots

<!-- Add screenshots here -->

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Shell | Tauri v2 | Native desktop window (Linux, macOS, Windows) |
| Frontend | SvelteKit 5 (Svelte 5 runes) | Reactive UI with `$state`, `$derived`, `$props` |
| Backend | Rust | Model parsing, tensor operations, merge execution |
| Tensors | Candle | Rust ML framework for tensor math (SLERP, TIES, DARE) |
| GGUF Inference | llama.cpp | Quantized model inference with GPU support |
| ST Inference | HuggingFace Transformers | SafeTensors inference via Python |
| Model Hub | HuggingFace API | Model discovery and download |
| Async | Tokio | Non-blocking task execution |
| Serialization | Serde | Rust ↔ Frontend data exchange |

---

## Architecture

```
┌───────────────────────────────────────────┐
│              SvelteKit 5 UI               │
│   Dashboard │ Load │ Inspect │ Optimize   │
│   Hub │ Convert │ M-DNA │ Test │ Settings │
└──────────────────┬────────────────────────┘
                   │ Tauri IPC (invoke / events)
┌──────────────────▼────────────────────────┐
│              Rust Backend                 │
│                                           │
│  ┌─────────────┐  ┌────────────────────┐  │
│  │ Model       │  │ Merge Engine       │  │
│  │ Parser      │  │ (SLERP/TIES/DARE)  │  │
│  │ (GGUF/ST)   │  │ Candle tensors     │  │
│  └─────────────┘  └────────────────────┘  │
│  ┌─────────────┐  ┌────────────────────┐  │
│  │ Quantizer   │  │ Profiler           │  │
│  │ (llama.cpp) │  │ Layer analysis     │  │
│  └─────────────┘  └────────────────────┘  │
│  ┌─────────────┐  ┌────────────────────┐  │
│  │ HF Hub      │  │ Converter          │  │
│  │ Downloader  │  │ (ST → GGUF)        │  │
│  └─────────────┘  └────────────────────┘  │
└───────────────────────────────────────────┘
```

---

## Supported Formats

| Format | Load | Inspect | Optimize | Convert | Merge | Test |
|--------|:----:|:-------:|:--------:|:-------:|:-----:|:----:|
| GGUF | ✓ | ✓ | ✓ | output | ✓ | ✓ |
| SafeTensors | ✓ | ✓ | — | input | ✓ | ✓ |
| Sharded Folders | ✓ | ✓ | — | input | ✓ | ✓ |

---

## M-DNA Forge — Model Merging

Merge 2–5 parent models into hybrid offspring with full control over merge strategy and layer composition.

| Method | Description |
|--------|-------------|
| **SLERP** | Spherical interpolation — best for 2-model merges |
| **TIES** | Trim, elect sign, merge — resolves task vector interference |
| **DARE** | Drop and rescale — prunes delta parameters |
| **DeLLa** | Density-based layer-level adaptive merging |
| **Frankenmerge** | Cherry-pick specific layers from specific parents |
| **Task Arithmetic** | Add task vectors from multiple finetunes |
| **Passthrough** | Stack layers sequentially |
| **Average** | Weighted mean of tensors |

Features:
- Isometric 3D visualization of parent and offspring towers
- Layer specialization analysis (Syntactic / Semantic / Reasoning)
- Auto-assign layers (Split / Interleave)
- Output to SafeTensors (HF-compatible directory) or GGUF (with embedded tokenizer)
- Background execution — navigate freely while merge runs

---

## Design

Industrial label / technical spec sheet aesthetic:

- **Font**: JetBrains Mono (system monospace stack)
- **Theme**: Dark default, light available
- **Corners**: 0px border-radius (sharp everywhere)
- **Borders**: 1px everywhere, grid-based layouts
- **Decorations**: Corner marks on panels, barcode patterns, serial identifiers
- **Color semantics**:

| Color | Variable | Meaning |
|-------|----------|---------|
| Amber | `--accent` | Idle / Default |
| Blue | `--info` | Working / Processing |
| Green | `--success` | Success / Complete |
| Red | `--danger` | Error / Failure |
| Gray | `--gray` | Paused / Inactive |

---

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [Node.js](https://nodejs.org/) (v20+)
- [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/) for your OS
- Python 3.10+ (optional — for SafeTensors conversion)

### Run locally

```bash
npm install
npm run tauri dev
```

### Build for production

```bash
npm run tauri build
```

### Quick start

1. Launch ForgeAI
2. Go to **Load** (01) → import a GGUF or SafeTensors model
3. Go to **Inspect** (02) → explore architecture, memory layout, compatibility
4. Go to **Optimize** (03) → quantize to a smaller size
5. Go to **Test** (09) → run inference and see token output

---

## System Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| **OS** | Linux, macOS, Windows | — |
| **RAM** | 8 GB | 16 GB+ |
| **Disk** | 2 GB + model storage | SSD with 50 GB+ free |
| **GPU** | Not required | NVIDIA (CUDA) / AMD (Vulkan) / Apple Silicon (Metal) |

---

## Project Structure

```
src/                          # SvelteKit frontend
├── routes/                   # Pages (dashboard, load, inspect, optimize, hub, convert, dna, test, settings)
├── lib/                      # Stores (model.svelte.ts, dna.svelte.ts, hub.svelte.ts, etc.)
└── app.css                   # Global design system

src-tauri/                    # Rust backend
├── src/
│   ├── lib.rs                # Tauri app setup + command registration
│   ├── commands.rs           # Load, inspect, optimize, convert, test commands
│   ├── merge_commands.rs     # M-DNA merge commands
│   ├── model/                # Model parsing (GGUF, SafeTensors), state, errors
│   └── merge/                # Merge engine (methods, planner, executor, profiler, registry)
└── tauri.conf.json           # Tauri config (1100x720 window)

docs/                         # Mintlify documentation site
```

---

## Documentation

Full documentation available at the [docs site](docs/) (Mintlify).

Covers every module with feature descriptions, workflows, and guides for:
- [Supported Formats](docs/guides/supported-formats.mdx)
- [Merge Methods](docs/guides/merge-methods.mdx)
- [Quantization Levels](docs/guides/quantization-levels.mdx)
- [GPU Setup](docs/guides/gpu-setup.mdx)

---

## License

MIT
