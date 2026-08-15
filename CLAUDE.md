# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Product

**OhMyGPU Runtime** — an open-source, embeddable, headless local AI runtime for application developers.
It runs GGUF models through a supervised `llama-server` (llama.cpp) subprocess and exposes:

- `POST /v1/responses` (canonical) and `POST /v1/chat/completions` (compatibility) — an OpenAI-compatible **subset**, both feeding **one** internal inference pipeline
- `GET /v1/models`
- `/ohmygpu/v1/*` — the Management API (health, status, hardware, catalog, model pull/delete/start/stop, backend install, shutdown)

The runtime is the product. The CLI (`ohmygpu`, alias `omg`), any future GUI, and third-party apps are clients. No UI technology may be required by the runtime. Local only: binds `127.0.0.1` by default.

Out of scope for v0.1 (do not add to the critical path): GUI/Tauri, chat app, image/audio/video, MCP, Ollama API, `/v1/completions`, response persistence, hosted tools, cloud. Deferred Candle/diffusion code lives in `archive/` (excluded from the workspace).

## Build & test

```bash
cargo build                                   # debug; no GPU cargo features exist any more
make build                                    # release: target/release/{ohmygpu-runtime,ohmygpu}
cargo test --workspace                        # fast: unit + API tests with a mock backend
OHMYGPU_E2E=1 cargo test -p ohmygpu_daemon --test e2e_llamacpp -- --ignored --nocapture   # real llama.cpp + ~470 MB model
cargo run --bin ohmygpu-runtime -- --port 10692 --data-dir /tmp/omg      # standalone daemon
cargo run --bin ohmygpu -- serve              # same daemon via the CLI
cargo run --bin ohmygpu -- model catalog | model pull <id> | run <id> | stop <id> | status | hardware
```

llama.cpp is **not** linked; the runtime downloads the official release binary per platform on first model start (`backend.llamacpp.auto_install`), or uses `OHMYGPU_LLAMA_SERVER` / `backend.llamacpp.server_path`.

## Workspace

```text
crates/core/              ohmygpu_core        paths, config (+env), hardware detection, catalog + ModelRef parsing,
                                              registry (registry.json), resumable downloader, lifecycle ModelState
crates/inference/         ohmygpu_inference   InferenceRequest/Response, InputItem/OutputItem, ToolDefinition/ToolCall,
                                              GenerationOptions, StreamEvent, ResponseAccumulator, InferenceError
crates/runtime_api/       ohmygpu_runtime_api RuntimeBackend { available, prepare, start } / ModelInstance { status, infer(_stream), wait, stop }
crates/runtime_llamacpp/  ohmygpu_runtime_llamacpp  install.rs (locate/managed install), process.rs (spawn/supervise/stop),
                                              wire.rs (internal ⇄ llama-server JSON/SSE), lib.rs (backend + instance)
daemon/                   ohmygpu_daemon      manager.rs (ModelManager: lifecycle orchestrator), api/{responses,chat_completions,models,management}.rs,
                                              error.rs (OpenAI error envelope), server.rs (bind/graceful shutdown), main.rs (`ohmygpu-runtime` bin),
                                              testing.rs (MockBackend, `testing` feature), tests/api.rs, tests/e2e_llamacpp.rs
cli/                      ohmygpu_cli         thin HTTP client of the Management API (`omg`)
archive/                  deferred code, not built
docs/architecture.md      assessment of the old code, backend decision, architecture
```

## Rules

1. **One inference pipeline.** New API features go through protocol adapters → `ohmygpu_inference` types → `ModelInstance`. Never let OpenAI schemas leak into `runtime_*` crates, and never let llama-server's wire format leak into `daemon/`.
2. **Explicit lifecycle.** All state changes go through `ModelManager`; background tasks must check the record `generation` before applying results.
3. **CLI stays thin.** No model/runtime business logic in `cli/`; use the Management API (read-only offline fallbacks only).
4. **Do not claim unimplemented API fields.** Unsupported request features return `400 unsupported`; document supported subsets in the README.
5. Keep the default test suite fast and offline (mock backend); real-model tests stay behind `OHMYGPU_E2E=1`.
6. MUST NOT use `ln` during testing — call binaries by path.
