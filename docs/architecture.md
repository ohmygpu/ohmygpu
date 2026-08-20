# OhMyGPU Runtime — Architecture

> **OhMyGPU Runtime is the product. Everything else is a client.**

This document has three parts:

1. The engineering assessment of the code that existed before the v0.1 refocus
   (what was kept, simplified, replaced, deferred, removed).
2. The backend decision (llama.cpp vs. the existing Candle implementation).
3. The v0.1 architecture as implemented.

---

## Part 1 — Assessment of the pre-v0.1 `runtime` branch

Snapshot assessed: commit `23b3356` (workspace of ~4.5k lines of Rust across
`crates/core`, `crates/runtimes/{runtime_api,runtime_candle,runtime_diffusion}`,
`daemon`, `cli`).

| # | Area | Findings | Verdict |
|---|------|----------|---------|
| 1 | **Workspace structure** | Sensible split (core / runtime_api / runtimes / daemon / cli). Core was GPU‑agnostic as intended. But `metal`/`cuda` cargo features leaked from Candle all the way into the CLI (`cli/build.rs` refused to build without a GPU feature), and the whole workspace pulled the Candle git dependency tree (slow builds, unstable git pin). | **SIMPLIFY** — keep the crate split; remove Candle from the active workspace; no GPU cargo features anywhere. |
| 2 | **Daemon** (`daemon/`) | axum server, CORS, tracing; `AppState` held one `CandleRuntime` and auto‑loaded a model on first request. Bound to `0.0.0.0` from the CLI. No lifecycle model, no management API, no graceful shutdown, no readiness. The `routes()` / `AppState` shape was reasonable. | **KEEP the skeleton, REPLACE the internals** — same crate, new orchestrator, explicit lifecycle, `127.0.0.1` default, management API, shutdown handling. |
| 3 | **CLI** (`cli/`) | Mixed thin‑client commands (`chat`, `serve status`) with fat local logic (`model pull/list/rm` operated on the registry directly, `gen image` ran diffusion in‑process, `mcp` server, `update`, `search`, `config`, GPU gate with interactive prompt via `dialoguer`, `ratatui`/`crossterm` deps unused). | **SIMPLIFY** — rewrite as a thin HTTP client of the Management API. Remove `gen`, `chat`, `mcp`, `search`, `update`, TUI deps, GPU gate. |
| 4 | **OpenAI API implementation** (`daemon/src/api/chat.rs`) | Working `/v1/chat/completions` (+SSE streaming) and `/v1/models`. Request/response types were hand‑rolled and mapped straight onto `runtime_api::ChatRequest` (so the OpenAI schema *was* the internal model). `prompt_tokens` hard‑coded to 0; no tools; no error object consistency. | **REPLACE** — keep the endpoint, re‑implement as a protocol adapter over the new internal inference model; add `/v1/responses`. |
| 5 | **Ollama compatibility** (`daemon/src/api/ollama.rs`, 356 lines) | Five endpoints, partly stubbed (fake digests, `size: 0`, "unknown" details), and coupled to the old runtime types. Not free to keep once the runtime types change. | **REMOVE FROM ACTIVE WORKSPACE** — no longer a product requirement (brief §26). |
| 6 | **Model management** (`core/registry.rs`, `core/models.rs`) | Simple JSON registry keyed by name; `ModelInfo` covered LLM/embedding/image/audio types; no lifecycle, no download state, no capabilities. Registry paths were global (hard to test). | **SIMPLIFY** — keep the JSON registry idea, make paths injectable, narrow to installed GGUF LLMs, add capabilities. Add a curated catalog and an explicit lifecycle state machine in the daemon. |
| 7 | **Hugging Face integration** (`core/downloaders/huggingface.rs`) | Working single‑file streaming download with a "prefer Q4_K_M GGUF" heuristic; but progress went to stdout via `indicatif` inside the *library*, no resume, no token support, no cancellation, model dir naming `owner--repo`. `search` API only used by the CLI. | **SIMPLIFY/REUSE** — keep the HTTP download core, add resume (`Range`), `.part` files, HF token, progress callbacks; drop stdout printing and search. |
| 8 | **Runtime abstraction** (`crates/runtimes/runtime_api`) | `Runtime` trait with `load/unload/chat/chat_stream` and `ChatRequest/ChatResponse` — protocol‑shaped and single‑model. No `available/prepare/start/stop/status` lifecycle, no process model. | **REPLACE** — new `RuntimeBackend` + `ModelInstance` traits operating on protocol‑independent inference types. |
| 9 | **Candle inference** (`runtime_candle`, ~600 lines) | In‑process; safetensors only (no GGUF); two architectures (llama, phi) via `candle-transformers`; hard‑coded Llama‑2 prompt template; hand‑rolled sampler; the decode loop re‑fed the whole sequence each step; F32 on Metal; a backend crash would take the daemon down. Would require OhMyGPU to maintain per‑architecture model code and chat templates. | **DEFER** — archived under `archive/runtime_candle` (not built). See Part 2. |
| 10 | **Diffusion** (`runtime_diffusion`, Z‑Image) | Working Z‑Image pipeline, but image generation is out of v0.1 scope. | **DEFER** — archived under `archive/runtime_diffusion` (not built). |
| 11 | **Hardware detection** (`cli/src/gpu.rs`) | Metal detection via `sysctl`, CUDA via `nvidia-smi`; lived in the CLI and gated CLI startup with an interactive prompt. Not exposed over HTTP. | **REPLACE/MOVE** — moved into `core::hardware`, exposed via `GET /ohmygpu/v1/hardware`; no CLI gate. |
| 12 | **Process / lifecycle management** | Only a PID file for the daemon (`cli/src/daemon.rs`) plus `pkill` fallback. No model lifecycle at all (a model was either loaded in‑process or not). | **REPLACE** — explicit per‑model state machine, backend subprocess supervision, crash detection, graceful shutdown. |
| 13 | **Tests** | None. | **ADD** — unit + API tests with a mock backend; real llama.cpp end‑to‑end tests isolated behind `OHMYGPU_E2E=1`. |

Summary of the classification:

```text
KEEP       axum daemon skeleton, JSON registry concept, HF download core, config layout (~/.config/ohmygpu)
SIMPLIFY   workspace/features, CLI (thin client), registry, downloader
REPLACE    runtime trait, OpenAI adapter, hardware detection, lifecycle/process mgmt
DEFER      runtime_candle, runtime_diffusion (archived), Ollama API, MCP, TUI, self-update
REMOVE     ollama.rs, gen/chat/mcp/search/update commands, GPU cargo features, ratatui/crossterm/dialoguer/rmcp/self_update deps
```

---

## Part 2 — Backend decision: llama.cpp (subprocess) over Candle

Criteria from the brief, applied honestly:

| Criterion | Existing Candle backend | llama.cpp (`llama-server` subprocess) |
|-----------|-------------------------|----------------------------------------|
| Model compatibility | safetensors only; llama + phi architectures; every new family needs Rust code | Any GGUF; hundreds of architectures maintained upstream; quantized by default |
| Maintenance burden | We own model code, chat templates, sampling, KV cache | We own an adapter (~800 lines) and a binary version pin |
| Metal support | Yes (F32) | Yes, mature (prebuilt `macos-arm64` release) |
| CUDA support | Yes (build‑time feature) | Yes (prebuilt Windows CUDA; Linux via user‑supplied build or Vulkan prebuilt) |
| Streaming | Yes | Yes (SSE, OpenAI‑style) |
| Tool calling | Not implemented; would need template + parser work per model | Built in (`--jinja`): Qwen 2.5/3, Llama 3.x, Mistral, Functionary, generic fallback |
| Stability | Experimental | Widely deployed |
| Process isolation | None (in‑process; crash kills daemon) | Full (separate OS process per running model) |
| Embeddability | Ties the daemon binary to a specific GPU toolchain at build time | Daemon is a small pure‑Rust binary; backend binary is downloaded per platform at runtime |
| Lifecycle management | Load/unload only | Start/stop/health per process; crash detection via exit status |
| Speed to reliable v0.1 | Slow (many gaps) | Fast (adapter + orchestration only) |

**Decision: llama.cpp is the v0.1 backend, driven as a supervised `llama-server` subprocess.**
OhMyGPU orchestrates a proven runtime instead of maintaining an inference engine.

Why a subprocess (HTTP to `llama-server` on `127.0.0.1:<ephemeral port>`) rather
than FFI bindings (`llama-cpp-2`): process isolation, no C++/CMake/CUDA
toolchain in our build, upstream chat‑template + tool‑call parsing comes with the
server, and prebuilt release binaries make "hide the llama.cpp binary" achievable —
OhMyGPU downloads the right release asset for the platform into
`~/.config/ohmygpu/runtimes/llamacpp/<tag>/` on first use.

The Candle code is archived (not deleted) under `archive/`. It could return as an
experimental adapter behind the same `RuntimeBackend` trait, outside the v0.1
critical path.

---

## Part 3 — v0.1 architecture

```text
Third-party application (Electron / Swift / Python / Tauri / ...)
                │  HTTP (127.0.0.1:10692)
                ▼
┌───────────────────────────────────────────────────────────────┐
│  ohmygpu-runtime (daemon)                                     │
│                                                               │
│   /v1/responses ─────▶ Responses adapter ──┐                  │
│   /v1/chat/completions ▶ ChatCompletions ──┼─▶ InferenceRequest│
│   /v1/models                       adapter │        │         │
│                                            │        ▼         │
│   /ohmygpu/v1/* ─▶ Management API ─▶ ModelManager (lifecycle) │
│                                            │        │         │
│                                            ▼        ▼         │
│                                     RuntimeBackend / ModelInstance
│                                       (runtime_llamacpp)      │
└─────────────────────────────────────────────┬─────────────────┘
                                              │ spawn + HTTP
                                              ▼
                                       llama-server (one per running model)
                                              │
                                        Metal / CUDA / Vulkan / CPU
```

### Workspace

```text
crates/
├── core/               ohmygpu_core        config, paths, hardware, catalog, registry, HF downloader, lifecycle state
├── inference/          ohmygpu_inference   protocol-independent InferenceRequest/Response/StreamEvent, tools, errors
├── runtime_api/        ohmygpu_runtime_api RuntimeBackend + ModelInstance traits
└── runtime_llamacpp/   ohmygpu_runtime_llamacpp  llama-server locate/install/spawn/supervise + wire translation
daemon/                 ohmygpu_daemon      axum server, ModelManager, protocol adapters, management API, `ohmygpu-runtime` bin
cli/                    ohmygpu_cli         thin HTTP client (`ohmygpu` / `omg`)
archive/                deferred code (candle, diffusion) — not part of the workspace
```

### One inference pipeline

Both `/v1/responses` and `/v1/chat/completions` are *protocol adapters*:

```text
external JSON ──parse──▶ InferenceRequest ──▶ ModelManager.instance(model) ──▶ ModelInstance.infer_stream()
                                                                                       │
external JSON ◀─serialize── InferenceResponse / StreamEvent ◀───────────────────────────┘
```

`InferenceRequest` (crate `ohmygpu_inference`) carries: model id, ordered
`InputItem`s (`Message{role,content}`, `ToolCall`, `ToolResult`), `ToolDefinition`s,
`ToolChoice`, and `GenerationOptions` (max tokens, temperature, top_p, stop, seed,
penalties). Runtime adapters never see OpenAI schemas. Non‑streaming inference is
implemented by collecting the streaming events, so both endpoints and both modes
share exactly one path through the backend.

### Explicit model lifecycle

```text
not_installed ─pull─▶ downloading ─▶ installed ─start─▶ starting ─▶ running ─stop─▶ stopping ─▶ stopped
                          │                                │           │
                          └──▶ error ◀─────────────────────┴───────────┘ (crash / failed start)
```

`GET /ohmygpu/v1/models/{id}` always reports the state plus `download` progress,
`error.message`, and `runtime` details (backend, pid, port) when running.

### Backend contract (`ohmygpu_runtime_api`)

```rust
trait RuntimeBackend {  id(); available(); prepare(); start(spec) -> ModelInstance }
trait ModelInstance  {  status(); infer(); infer_stream(); wait(); stop() }
```

`prepare()` for llama.cpp resolves the binary in this order: explicit config path →
`OHMYGPU_LLAMA_SERVER` env → managed install dir → `PATH`; if none is found and
`auto_install` is on (default), it downloads the matching GitHub release asset.

### Storage / configuration

```text
$OHMYGPU_HOME (default ~/.config/ohmygpu)/
├── config.toml
├── registry.json            installed models
├── models/<model-id>/*.gguf
├── runtimes/llamacpp/<tag>/llama-server (+ libs)
└── daemon.json              pid/port of the running daemon
```

Everything the daemon writes lives under one directory, so an application that
bundles the runtime can point it at its own data dir with `--data-dir` /
`OHMYGPU_HOME`.

### Security defaults

Binds `127.0.0.1` only. No auth, no accounts. Configurable host for advanced users.

### Known limitations (v0.1)

- One `llama-server` process per running model; concurrent requests to the same
  model are queued by llama.cpp (`-np` auto).
- If the daemon is SIGKILLed, child `llama-server` processes may be orphaned
  (normal shutdown, SIGTERM and Ctrl‑C stop them).
- Managed backend install covers macOS (arm64/x64, Metal), Linux x64/arm64 (CPU, or
  Vulkan when a GPU is detected), Windows x64 (CPU/Vulkan). CUDA builds and other
  targets: point `backend.llamacpp.server_path` at your own build.
- No response persistence, conversations, or hosted tools (by design).
- Vision: image input only (no image output, no audio); remote image URLs are
  fetched by the daemon with a 20 s timeout and a 20 MB cap; a model accepts
  images only when a projector (`mmproj`) is installed next to it.
