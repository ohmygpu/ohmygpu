# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Product

**OhMyGPU Runtime** — an open-source, embeddable, headless local AI runtime for application developers.
It runs GGUF models (text, and vision models with a multimodal projector for image input) through a supervised `llama-server` (llama.cpp) subprocess, and whisper.cpp ggml models through a supervised `whisper-server` subprocess, and exposes:

- `POST /v1/responses` (canonical) and `POST /v1/chat/completions` (compatibility) — an OpenAI-compatible **subset**, both feeding **one** internal inference pipeline
- `POST /v1/audio/transcriptions` — OpenAI-compatible speech to text (multipart upload, decoded in the daemon, whisper.cpp backend)
- `GET /v1/models`
- `/ohmygpu/v1/*` — the Management API (health, status, hardware, catalog, model pull/delete/start/stop, backend install, shutdown)

The runtime is the product. The CLI (`ohmygpu`, alias `omg`), any future GUI, and third-party apps are clients. No UI technology may be required by the runtime. Local only: binds `127.0.0.1` by default.

Out of scope for v0.1 (do not add to the critical path): GUI/Tauri, chat app, image/audio/video, MCP, Ollama API, `/v1/completions`, response persistence, hosted tools, cloud. Deferred Candle/diffusion code lives in `archive/` (excluded from the workspace).

**Direction after v0.1 (recorded 2026-08-20, see `docs/vision.md`):** OhMyGPU is meant to be a *base* you can drop onto any GPU box (rented VPS included), pull any open model — including one released today — and be serving it within 10 minutes. That implies: a vLLM backend behind the same `RuntimeBackend` trait, per-model recipes as data (YAML, spec in `docs/recipes.md`, types in `crates/core/src/recipe.rs`) instead of the static Rust catalog, multi-GPU hardware resolution, HF snapshot downloads, an authenticated remote mode, and a pre-baked CUDA image + installer. Local-first stays the default; remote mode must be opt-in and require an API key.

## Build & test

```bash
cargo build                                   # debug; no GPU cargo features exist any more
make build                                    # release: target/release/{ohmygpu-runtime,ohmygpu}
cargo test --workspace                        # fast: unit + API tests with a mock backend
OHMYGPU_E2E=1 cargo test -p ohmygpu_daemon --test e2e_llamacpp -- --ignored --nocapture   # real llama.cpp (+ whisper.cpp) + small models; on macOS set OHMYGPU_WHISPER_SERVER to a local whisper-server build until a release carries one
cargo run --bin ohmygpu-runtime -- --port 10692 --data-dir /tmp/omg      # standalone daemon
cargo run --bin ohmygpu -- serve              # same daemon via the CLI
cargo run --bin ohmygpu -- model catalog | model pull <id> | run <id> | stop <id> | status | hardware
cargo run --bin ohmygpu -- upgrade [VERSION] [--check] [--force]   # self-update: replaces the *running* binaries' files — test on a copy
make recipe-schema                            # regenerate schemas/recipe-v1.json after editing crates/core/src/recipe.rs
```

llama.cpp is **not** linked; the runtime downloads the official release binary per platform on first model start (`backend.llamacpp.auto_install`), or uses `OHMYGPU_LLAMA_SERVER` / `backend.llamacpp.server_path`.

## Workspace

```text
crates/core/              ohmygpu_core        paths, config (+env), hardware detection, catalog + ModelRef parsing,
                                              registry (registry.json; capabilities, modalities, native context_length),
                                              gguf.rs (GGUF header reader: architecture + context_length, read at install /
                                              backfilled on load), resumable downloader, lifecycle ModelState,
                                              recipe.rs (recipe schema v1: YAML/JSON loader, validation, JSON Schema)
crates/inference/         ohmygpu_inference   InferenceRequest/Response, InputItem/OutputItem, ToolDefinition/ToolCall,
                                              GenerationOptions, StreamEvent, ResponseAccumulator, InferenceError
crates/runtime_api/       ohmygpu_runtime_api RuntimeBackend { available, prepare, start } / ModelInstance { status, infer(_stream), transcribe, wait, stop }
crates/runtime_common/    ohmygpu_runtime_common  process.rs (supervised child server: spawn/logs/exit/stop), install.rs (locate, install records,
                                              download + extract release archives) — shared by every subprocess backend
crates/runtime_llamacpp/  ohmygpu_runtime_llamacpp  install.rs (asset choice + managed install), process.rs (llama-server args),
                                              wire.rs (internal ⇄ llama-server JSON/SSE), lib.rs (backend + instance)
crates/runtime_whisper/   ohmygpu_runtime_whisper   install.rs (official Linux/Windows assets; macOS build from our release), wire.rs
                                              (TranscriptionRequest ⇄ whisper-server /inference multipart + verbose_json), lib.rs
daemon/                   ohmygpu_daemon      manager.rs (ModelManager + Backends: lifecycle orchestrator, one backend per ModelKind),
                                              api/{responses,chat_completions,audio,images,models,management}.rs, audio.rs (decode + resample to 16 kHz),
                                              error.rs (OpenAI error envelope), server.rs (bind/graceful shutdown), main.rs (`ohmygpu-runtime` bin),
                                              testing.rs (MockBackend, `testing` feature), tests/api.rs, tests/e2e_llamacpp.rs
cli/                      ohmygpu_cli         thin HTTP client of the Management API (`omg`); upgrade.rs = `omg upgrade`
                                              (self-update of both binaries from a GitHub release, SHA256SUMS verified)
install.sh                `curl | sh` installer for macOS/Linux: latest release → existing install dir, else /usr/local/bin,
                          else ~/.local/bin; refuses Homebrew installs (→ brew upgrade); attached to every release; shellcheck-clean
install.ps1               `irm | iex` installer for Windows x64: %LOCALAPPDATA%\Programs\ohmygpu + user PATH; omg.exe is a copy
packaging/homebrew/       render-formula.sh <tag> <SHA256SUMS.txt> → Formula/ohmygpu.rb for github.com/ohmygpu/homebrew-tap
                          (`brew install ohmygpu/tap/ohmygpu`); the release workflow renders + pushes it (secret HOMEBREW_TAP_TOKEN)
.cargo/config.toml        static CRT for x86_64-pc-windows-msvc so the Windows binaries run without the VC++ redistributable
.github/workflows/        release.yml (tag → test, build matrix, whisper-server macOS, GitHub release, Homebrew tap bump);
                          installers.yml (real runs of install.sh on ubuntu/macos + brew install of the rendered formula,
                          install.ps1 on windows with PowerShell 5.1 and 7 — the only place install.ps1 is tested)
recipes/                  example per-model recipes (YAML; loaded by core tests) — see docs/recipes.md
schemas/recipe-v1.json    generated JSON Schema for recipes (`make recipe-schema`; a test fails when stale)
archive/                  deferred code, not built
docs/architecture.md      assessment of the old code, backend decision, architecture
docs/vision.md            the post-v0.1 direction (base for any GPU box, any model in 10 minutes)
docs/recipes.md           recipe format spec: fields, merge rules, resolver contract, conventions
```

## Rules

1. **One inference pipeline.** New API features go through protocol adapters → `ohmygpu_inference` types → `ModelInstance`. Never let OpenAI schemas leak into `runtime_*` crates, and never let llama-server's / whisper-server's wire format leak into `daemon/`. Images follow the same rule: adapters turn `input_image`/`image_url` into `ContentPart::Image` (`daemon/src/api/images.rs` validates data: URLs and inlines http(s) URLs); backends only ever see `data:` URLs. Audio too: `daemon/src/audio.rs` decodes uploads to 16 kHz mono PCM (`AudioInput`); backends never see containers or codecs.
2. **Explicit lifecycle.** All state changes go through `ModelManager`; background tasks must check the record `generation` before applying results. Every model has a `ModelKind` (`llm` | `whisper`) that picks the backend (`Backends::for_kind`) and the API that serves it.
3. **CLI stays thin.** No model/runtime business logic in `cli/`; use the Management API (read-only offline fallbacks only).
4. **Do not claim unimplemented API fields.** Unsupported request features return `400 unsupported`; document supported subsets in the README.
5. Keep the default test suite fast and offline (mock backend); real-model tests stay behind `OHMYGPU_E2E=1`.
6. MUST NOT use `ln` during testing — call binaries by path.
