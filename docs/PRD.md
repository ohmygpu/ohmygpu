# OhMyGPU Runtime v0.1 — Product Definition

## One sentence

**OhMyGPU Runtime is an open-source, embeddable, headless local AI runtime for
application developers.** It runs open-source models behind OpenAI-compatible
APIs and owns the GPU runtime, model files and model lifecycle so applications
don't have to.

## What it is / is not

| Is | Is not |
|----|--------|
| Local AI infrastructure applications embed or depend on | A desktop app, ChatGPT clone, chat UI, TUI |
| One headless daemon with a stable local HTTP API | An image/audio/video generator (deferred) |
| An orchestrator of proven runtimes (llama.cpp) | An inference framework of its own |
| Local-first: one machine, `127.0.0.1` | A cloud GPU platform; agents; MCP |

**The runtime is the product. Everything else — CLI, future GUI, third-party
apps — is a client.**

## Primary user

An application developer (Electron, Tauri, Swift, Python, .NET, Java, Go,
Flutter, …) who wants local open-source models without learning CUDA, Metal,
GGUF internals, llama.cpp flags, download mechanics or process management.

Their code: *ensure model exists → start model → use `/v1/responses` or
`/v1/chat/completions` with the OpenAI client they already have.*

## v0.1 scope

Required:

- headless daemon (`ohmygpu-runtime`), binds `127.0.0.1:10692`, configurable port/data dir
- `POST /v1/responses` (canonical, Responses-API-compatible subset, streaming)
- `POST /v1/chat/completions` (ecosystem compatibility, streaming)
- both through **one** internal inference pipeline; protocol adapters at the boundary
- `GET /v1/models`
- Management API `/ohmygpu/v1/*`: health, status, hardware, catalog, models (pull/delete/start/stop), backend install, shutdown
- explicit model lifecycle: `not_installed → downloading → installed → starting → running → stopping → stopped`, plus `error` with reason
- one reliable LLM backend: llama.cpp (`llama-server` subprocess), auto-installed per platform
- hardware detection (platform, arch, CPU, memory, GPU, backend)
- curated model catalog + Hugging Face / URL GGUF pulls with resumable downloads
- minimal tool/function calling (the application executes tools)
- thin CLI (`omg`) over the Management API
- tests: fast unit/API suite with a mock backend; isolated real-llama.cpp e2e

Explicitly out of scope for v0.1: GUI, chat application, image/audio/video,
MCP, Ollama compatibility, legacy `/v1/completions`, response persistence and
`GET/DELETE /v1/responses/{id}`, hosted tools, cloud/multi-node, SDKs.

## Definition of done

The refactor is done when an Electron developer can bundle or install the
runtime, tell it *ensure model exists / start model* over HTTP, and then use
either inference endpoint from existing code — and the whole workflow is
**simple, predictable, reliable and boring**.
