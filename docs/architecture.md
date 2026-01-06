# OhMyGPU Architecture

## Vision

**Stop juggling 5 different AI tools. One daemon, all workloads.**

ohmygpu is unified local AI infrastructure - replacing the fragmented landscape of single-purpose tools with one daemon, one model repository, and one API surface.

### The Problem: Fragmentation

| Category | Current Tools | Problems |
|----------|---------------|----------|
| **LLM inference** | ollama, LM Studio, LocalAI | Each has own model cache, own API, own format |
| **Image generation** | ComfyUI, Automatic1111, Fooocus | Separate downloads, separate processes |
| **Audio transcription** | whisper.cpp, faster-whisper | Yet another binary to manage |
| **Model downloads** | huggingface-cli, civitai manual | No unified cache, no deduplication |
| **GPU monitoring** | nvidia-smi, gpustat | Disconnected from workloads |

### The Solution: Unified Infrastructure

```
Before:
├── ollama          (LLMs, own model cache)
├── ComfyUI         (images, own model cache)
├── whisper.cpp     (audio, own model cache)
└── 3 different UIs, 3 different APIs, duplicate 70GB models

After:
└── ohmygpu
    ├── daemon      (one API, one model cache, one GPU scheduler)
    ├── runtimes    (candle, llamacpp, comfyui, whisper)
    └── UI          (TUI/Desktop, or Open WebUI via API)
```

### What ohmygpu Replaces

| Category | Replaces | With Runtime |
|----------|----------|--------------|
| **LLM inference** | ollama, LM Studio | `runtime_candle`, `runtime_llamacpp` |
| **Image generation** | ComfyUI, Automatic1111 | `runtime_comfyui` |
| **Audio transcription** | whisper.cpp | `runtime_candle` (has Whisper support) |
| **Model management** | huggingface-cli, manual downloads | `ohmygpu_core` |
| **GPU scheduling** | nothing (users manage manually) | `ohmygpu_daemon` |

### Key Differentiators

| Aspect | ollama | ohmygpu |
|--------|--------|---------|
| Architecture | Monolithic | Modular (core + pluggable runtimes) |
| Model formats | Own format, converts on import | HuggingFace ecosystem directly |
| Workload types | LLMs only | LLMs, images, audio, video (via runtimes) |
| Scope | Local CLI/API | Local + remote GPU boxes + fleet |
| GPU scheduling | None (single model) | VRAM budget, concurrent models, queue |

### Roadmap: Beyond AI Inference

The name "ohmygpu" and the daemon + runtime architecture support GPU workloads beyond AI inference:

| Phase | Focus | Runtimes |
|-------|-------|----------|
| **v1** | AI Inference | `runtime_candle` (LLMs + Whisper) |
| **v2** | Wider AI + Training | `runtime_llamacpp`, `runtime_comfyui`, `runtime_trainer` (LoRA/fine-tune) |
| **v3** | General GPU | `runtime_ffmpeg` (video transcode), `runtime_blender` (3D render) |

**What fits the ohmygpu model:**

Any workload that:
- Needs GPU scheduling (avoid VRAM OOM)
- Is a "job" with start/progress/end lifecycle
- Benefits from unified queue and management

| Category | Fits? | Why |
|----------|-------|-----|
| **Training/fine-tuning** | Yes | Same users, same models, natural extension |
| **Video transcoding** | Yes | FFmpeg + NVENC, job-based, needs GPU scheduling |
| **Video upscaling** | Yes | Real-ESRGAN etc., similar to image gen |
| **3D rendering** | Maybe | Blender render jobs, different user base |
| **Scientific compute** | No | Too specialized, custom CUDA kernels |

**v1 principle:** Ship AI inference first. The architecture supports expansion, but focus wins.

---

## Workspace Crates

### ohmygpu_core (Lib, Pure Core)

Only handles stable, cross-platform concerns:

- HuggingFace API client
- Downloader (resume, verification, concurrency, rate limiting)
- Local model repository (manifest, indexing, deduplication, GC)
- Run definition (RunSpec): model, parameters, VRAM budget, ports, etc.
- Event stream: progress/log/events (for UI consumption)

**Core Principle:** No direct GPU inference framework bindings. No libtorch, no CUDA.

---

### ohmygpu_runtime_* (Pluggable Backends, Lib or Bin)

Inference backends as plugins/adapters rather than embedded in core:

- `ohmygpu_runtime_candle` (Rust-native, first runtime for PoC - LLMs + Whisper)
- `ohmygpu_runtime_llamacpp` (wider model/quantization support, second priority)
- `ohmygpu_runtime_comfyui` (future, image generation workflows)
- `ohmygpu_runtime_vllm` (future, production server deployments)

**Runtime responsibilities:**

- Start/stop backend processes (or call libraries) based on RunSpec
- Provide unified trait: `start()` / `stop()` / `health()` / `openai_endpoint()` etc.
- Forward logs/status to core's event bus

---

### ohmygpu_daemon (Bin, Strongly Recommended)

The "moat container" for future capabilities:

- Scheduling (concurrent models/tasks, VRAM strategy)
- Model lifecycle (hot-swap, idle unloading)
- External API (OpenAI-compatible + management API)

With daemon, CLI/Tauri become mere "clients" - truly achieving "one capability set, multi-endpoint access".

---

### ohmygpu_cli (Bin)

- TUI (ratatui)
- Calls daemon exclusively
- Includes MCP server (`omg mcp` command)

**MCP Server** (via `omg mcp`):

MCP (Model Context Protocol) server for Claude Desktop and other MCP clients.

- Exposes local models as MCP tools
- Connects to daemon via HTTP
- Runs as stdio server for MCP protocol

Available tools: `chat`, `list_models`, `status`

---

### ohmygpu_desktop (Tauri Bin, Future)

- GUI (React/Vue)
- Communicates with daemon via HTTP/WebSocket (cleaner and more extensible than direct Tauri command → Rust function calls)

---

## Why GUI → Daemon Instead of Direct Tauri Commands?

Direct "Tauri Command → core function" calls are convenient short-term but have two long-term problems:

1. **Desktop App coupled to core:** Future remote management (another GPU machine on LAN) or Web UI support would require refactoring.

2. **Complex process/task/log management:** Inference tasks are long-running, need streaming logs and cancellation; cramming these into Tauri commands becomes awkward.

**Better structure:**

- Desktop app includes daemon (starts one locally)
- UI connects to daemon via websocket/http
- CLI also connects to daemon

This naturally supports: local / remote GPU box / multi-machine cluster (even "fleet" management later).

---

## File Tree

```
ohmygpu/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── core/                     # ohmygpu_core
│   ├── registry/                 # (optional) pure repository index/manifest/GC
│   └── runtimes/
│       ├── runtime_api/          # runtime trait + common types
│       ├── runtime_candle/       # Rust-native (PoC - LLMs + Whisper)
│       ├── runtime_llamacpp/     # (future, wider model support)
│       ├── runtime_comfyui/      # (future, image generation)
│       └── runtime_vllm/         # (future, production servers)
├── daemon/                       # ohmygpu_daemon (lib)
├── cli/                          # ohmygpu_cli (bin) - includes MCP server
├── desktop/                      # ohmygpu_desktop (tauri bin, future)
├── assets/
└── docs/
```

---

## Crate Responsibilities

### 1. ohmygpu_core (Cross-platform Pure Core)

Only handles stable "model asset layer + run specification". Does not touch GPU inference implementation.

**Includes:**

- HuggingFace API interaction (model/file list, commit, etag, etc.)
- Download management: concurrency, resume, verification, rate limiting, retry
- Local model repository: directory structure, indexing, deduplication strategy, GC
- `ModelRef` / `ModelRevision` / `Artifact` definitions ("model asset graph")
- `RunSpec`: all declarative parameters for a single run (model, backend preference, VRAM budget, port, concurrency, etc.)
- Events and progress: `EventStream` (download/run/error/log)

**Outputs:**

- Pure Rust API (sync/async both supported)
- No HTTP output, no UI output, no process spawning

---

### 2. runtimes/runtime_api (Backend Unified Interface Layer)

The contract for "pluggable runtimes".

**Includes:**

- `Runtime` trait (lifecycle management)
- `RuntimeCaps` (capability description: supports chat? images? video? streaming?)
- `RuntimeConfig` (backend common params + extension point for backend-specific params)
- Standardized status: `RuntimeStatus` (starting/running/degraded/stopped)
- Standardized log/event model (for daemon to forward to clients)

**Core principles:**

- Runtime only implements "how to start, stop, health-check, and expose endpoint for a given backend"
- Runtime does not do HF downloads (core handles that); runtime only receives core's output paths/manifest

---

### 3. runtimes/runtime_* (Specific Backend Adapters)

Each runtime is an independent crate:

- Transforms `RunSpec` + local model path → backend startup method (CLI/config file/HTTP)
- Handles start/stop/health check/log capture
- (Optional) Supports "model hot-swap" or "multi-model concurrency", unified via `runtime_api`

**Recommendation:** Start with `runtime_candle` for PoC (Rust-native, no subprocess management). Add `runtime_llamacpp` later for wider model/quantization support.

---

### 4. ohmygpu_daemon (System Hub, Strongly Recommended)

All UIs connect to it. This is your moat.

**Responsibilities:**

- Local/remote unified entry (unchanged for future multi-machine)
- Persistent config: installed models, recent usage, default runtime, token/ACL
- Scheduling: which models/tasks run concurrently; resource strategy (VRAM budget/concurrency/queue)
- Lifecycle: start/stop/restart runtime; crash recovery; health patrol

**External APIs:**

- OpenAI-compatible (at minimum: chat/completions + streaming)
- Management API (models list/install/remove/status/logs)
- Event channel (WebSocket/SSE)

---

### 5. ohmygpu_cli (TUI / Power Users)

**Responsibilities:**

- "k9s/nvidia-smi style": view downloads, tasks, runtime status
- Model management: `model list` / `model pull` / `model rm` / `model info` / `model gc`
- Daemon control: `serve` / `serve status` / `serve stop`
- Content generation: `gen image` / `gen video`
- Interactive: `chat <model>`
- Utilities: `search` / `config` / `mcp` / `update`
- Connects to daemon via HTTP (localhost:11434)

---

### 6. ohmygpu_desktop (General Users)

**Responsibilities:**

- UI only (model management, run, logs, doctor)
- Default starts embedded daemon with app (or prompts to install daemon)
- Communicates with daemon via HTTP/WebSocket

Not recommended: Using Tauri command to directly call core (works short-term, blocks remote/multi-endpoint long-term).

---

## Key Data Structures

### Model Asset Layer

| Structure | Fields |
|-----------|--------|
| `ModelRef` | `provider: HF \| Local \| URL`, `repo: "org/name"`, `kind: LLM \| Diffusion \| Video \| ...` |
| `Revision` | `branch/tag/commit`, `resolved_commit_hash` |
| `Artifact` | `filename`, `size`, `sha256?`, `etag?`, `format: gguf \| safetensors \| ...` |
| `InstallRecord` | `model_ref`, `revision`, `artifacts[]`, `local_paths`, `installed_at` |

### RunSpec (Declarative Run Specification)

- `run_id`
- `model_ref` + `revision`
- `preferred_runtime` (optional; empty lets scheduler choose)
- `mode`: chat | image | video | embeddings | ...
- `resources`: `{ gpu_id?, vram_budget_mb?, max_concurrency?, cpu_threads? }`
- `network`: `{ bind_addr, port, auth_token?, cors? }`
- `params`: `{ temperature, top_p, steps, cfg_scale, ... }` (common + extension fields)

### Runtime Contract

| Method | Description |
|--------|-------------|
| `caps()` | What the runtime can do |
| `prepare(run_spec, install_record)` | Generate runtime plan for backend startup (CLI/config/port) |
| `start(plan)` | Start the runtime |
| `stop(run_id)` | Stop a running instance |
| `status(run_id)` | Get current status |
| `logs(run_id)` | Get logs |
| `endpoint(run_id)` | Return OpenAI-compatible base_url or backend native endpoint |

---

## API Skeleton (Daemon External)

### Management API

| Endpoint | Description |
|----------|-------------|
| `GET /v1/health` | Health check |
| `GET /v1/models` | Installed models list (with usage, revision) |
| `POST /v1/models/install` | HF repo + revision → start download (returns job_id) |
| `GET /v1/jobs/{id}` | Download/install progress |
| `POST /v1/runs` | Submit RunSpec → returns run_id |
| `GET /v1/runs` | All run statuses |
| `POST /v1/runs/{id}/stop` | Stop a run |
| `GET /v1/runs/{id}/logs` | Get run logs |
| `GET /v1/events` (WS/SSE) | Unified event stream (download, run, error, alert) |

### OpenAI-Compatible API

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming preferred) |
| `POST /v1/images/generations` | (Future) Image generation |
| `POST /v1/embeddings` | (Future) Embeddings |
| `GET /v1/models` | Model list |

### Ollama-Compatible API (drop-in replacement)

For users migrating from ollama, we support the native Ollama API format:

| Endpoint | Description |
|----------|-------------|
| `POST /api/chat` | Chat with a model |
| `POST /api/generate` | Generate completion |
| `GET /api/tags` | List local models |
| `POST /api/show` | Show model info |
| `GET /api/version` | Version info |

This allows existing ollama scripts and integrations to work without modification.

**Note:** Start with the most ecosystem-valuable endpoints. OpenAI chat streaming + Ollama chat are the priorities.

---

## Design Principles (Anti-patterns to Avoid)

### 1. Don't bind inference frameworks in core

> No tch/libtorch/CUDA in core.
> Make these runtime plugins. Core only handles assets/specifications.

### 2. Don't let desktop UI directly call core

> UI only connects to daemon.
> This naturally supports remote GPU box, CLI sharing, WebUI sharing.

### 3. Don't hardcode model download directory structure

> Define manifest/index first. Directory is just implementation detail.
> Future deduplication/shared cache/cross-machine sync will be much easier.
