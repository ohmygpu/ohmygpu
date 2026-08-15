# OhMyGPU Runtime

**OhMyGPU Runtime is open-source local AI infrastructure for application developers.** It runs open-source models behind OpenAI-compatible APIs and manages the GPU runtime, model files, and model lifecycle so your application doesn't have to.

```text
Your application
(Electron / Swift / Python / Tauri / .NET / Go / …)
        │  HTTP on 127.0.0.1:10692
        ▼
OhMyGPU Runtime  ──  headless daemon
        │  /v1/responses · /v1/chat/completions · /v1/models
        │      (inference, OpenAI-compatible subset)
        │  /ohmygpu/v1/*
        │      (model & runtime management)
        ▼
llama.cpp  ──  supervised subprocess per running model
        │
        ▼
Metal / CUDA / Vulkan / CPU
```

| Component | Role |
|-----------|------|
| **OhMyGPU Runtime** (`ohmygpu-runtime`) | **The product.** Headless daemon: inference APIs + Management API + model lifecycle + backend supervision. |
| **OhMyGPU CLI** (`ohmygpu`, alias `omg`) | An administrative client of the runtime. Thin by design. |
| Future GUI | An optional client. Nothing in the runtime depends on a UI. |
| **Third-party applications** | The primary consumers. They talk to the runtime over HTTP. |

OhMyGPU is **not** a chat app, a desktop app, an agent framework, an image generator, or a cloud platform. It is the boring, reliable local runtime underneath those things.

## What your app no longer has to manage

CUDA · Metal · GPU/VRAM detection · model formats · downloads · storage · inference binaries · processes · ports · crashes · runtime configuration.

Your app tells the runtime *ensure model exists → start model*, then uses the OpenAI client library it already has.

## Quick start

Requires: macOS (Apple Silicon or Intel), Linux x86_64/arm64, or Windows x86_64. A GPU is used when present (Metal / Vulkan / CUDA); CPU works too.

```bash
# build (no GPU toolchain needed — llama.cpp is fetched at runtime)
make build            # → target/release/ohmygpu-runtime and target/release/ohmygpu
alias omg=$PWD/target/release/ohmygpu
```

Start the runtime:

```bash
omg serve                      # or: ohmygpu-runtime --port 10692
```

Download a model (see the curated catalog with `omg model catalog`):

```bash
omg model pull qwen2.5-1.5b-instruct
```

Start the model (first start also downloads the matching llama.cpp release for your platform):

```bash
omg run qwen2.5-1.5b-instruct
```

### Modern API — `POST /v1/responses`

```bash
curl http://127.0.0.1:10692/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "input": "Explain why the sky is blue."
  }'
```

### Ecosystem compatibility — `POST /v1/chat/completions`

```bash
curl http://127.0.0.1:10692/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [{"role": "user", "content": "Explain why the sky is blue."}]
  }'
```

Both endpoints reach the **same** local model through the **same** internal inference pipeline; pick whichever your libraries already speak.

### From an existing OpenAI client

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:10692/v1", api_key="not-needed")

r = client.responses.create(model="qwen2.5-1.5b-instruct", input="Explain why the sky is blue.")
print(r.output_text)

c = client.chat.completions.create(model="qwen2.5-1.5b-instruct",
                                   messages=[{"role": "user", "content": "Hello!"}], stream=True)
for chunk in c:
    print(chunk.choices[0].delta.content or "", end="")
```

Verified against the official `openai` Python SDK (models list, Responses create/stream/tools, Chat Completions create/stream/tools, error types).

## The workflow from an application

```text
GET  /ohmygpu/v1/models/{id}          → state: not_installed | downloading | installed | starting | running | stopping | stopped | error
POST /ohmygpu/v1/models/pull          {"model": "qwen2.5-1.5b-instruct"}       → 202, poll state/download progress
POST /ohmygpu/v1/models/{id}/start?wait=true                                   → 200 running (or 502 with the reason)
POST /v1/responses  /  POST /v1/chat/completions                              → inference
POST /ohmygpu/v1/models/{id}/stop
POST /ohmygpu/v1/shutdown
```

Every state, download percentage, and failure reason is visible over HTTP. Nothing requires the CLI.

## API reference (v0.1)

### Inference — OpenAI-compatible **subset** (`/v1`)

Only what is listed here is implemented. Unknown fields are ignored; unsupported features return `400` with `"code": "unsupported"`.

**`POST /v1/responses`**

| Supported | Notes |
|-----------|-------|
| `model`, `input` | `input` is a string or an array of items: `message` (roles `user`/`assistant`/`system`/`developer`, string or `input_text`/`output_text` parts), `function_call`, `function_call_output` |
| `instructions` | becomes the system message |
| `tools` (`type: "function"`), `tool_choice` (`auto` / `none` / `required` / `{"type":"function","name":…}`) | tool calls are returned as `function_call` output items; **your application executes them** |
| `temperature`, `top_p`, `max_output_tokens`, `metadata` | |
| `stream: true` | Responses-style SSE: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta/.done`, `response.content_part.done`, `response.function_call_arguments.delta/.done`, `response.output_item.done`, `response.completed` / `response.incomplete` / `response.failed`, `error` |
| Response object | `id`, `object: "response"`, `status` (`completed` / `incomplete` / `failed`), `output` (`message` with `output_text`, `function_call`), `usage`, `incomplete_details`, `error`, echoed request fields; `store` is always `false` |

Not supported (400): `previous_response_id`, `background`, hosted tools (web search, file search, code interpreter, computer use), image/file inputs, non-text `text.format`. Not implemented: `GET/DELETE /v1/responses/{id}`, `/cancel`, `/compact`, conversations, response storage.

**`POST /v1/chat/completions`**

| Supported | Notes |
|-----------|-------|
| `model`, `messages` | roles `system` / `developer` / `user` / `assistant` (with `tool_calls`) / `tool`; content as string or text parts |
| `tools` (`function`), `tool_choice` | |
| `temperature`, `top_p`, `max_tokens` / `max_completion_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty` | |
| `stream`, `stream_options.include_usage` | standard `chat.completion.chunk` SSE, `data: [DONE]` terminator |
| Response object | `id`, `object: "chat.completion"`, `choices[0].message` (`content`, `tool_calls`), `finish_reason` (`stop` / `length` / `tool_calls`), `usage` |

Not supported (400): `n > 1`, image/audio content parts, non-text `response_format`. Ignored: `logprobs`, `user`, `parallel_tool_calls`.

**`GET /v1/models`**, **`GET /v1/models/{id}`** — installed models, OpenAI list shape (`created` = install time, extra `state` field).

**`POST /v1/completions`** (legacy) is intentionally not implemented.

**Errors** use the OpenAI envelope on every endpoint:

```json
{"error": {"message": "…", "type": "invalid_request_error", "code": "model_not_running", "param": "model"}}
```

| HTTP | `code` | When |
|------|--------|------|
| 400 | `invalid_request`, `invalid_json`, `unsupported` | malformed or unsupported request |
| 404 | `model_not_found` | not installed / unknown |
| 409 | `model_not_running` | installed but not running (message tells you how to start it) |
| 502 | `backend_error`, `model_start_failed` | llama.cpp failed |
| 503 | `backend_unavailable` | llama.cpp went away mid-request |

By default a request for a stopped model is a `409`; set `inference.auto_start = true` in the config to start it on demand instead.

### Management — `/ohmygpu/v1`

| Method & path | Purpose |
|---------------|---------|
| `GET  /ohmygpu/v1/health` | liveness (`{"status":"ok","version":…}`); also `GET /health` |
| `GET  /ohmygpu/v1/status` | version, uptime, pid, bind address, data dir, backend availability, installed/running/downloading models |
| `GET  /ohmygpu/v1/hardware` | platform, architecture, CPU, system memory, GPU (vendor, name, memory) and the acceleration backend (`metal` / `cuda` / `vulkan` / `cpu`) |
| `GET  /ohmygpu/v1/backend`, `POST /ohmygpu/v1/backend/install` | llama.cpp availability; install it now instead of on first start |
| `GET  /ohmygpu/v1/catalog` | curated supported models with `installed`/`state` |
| `GET  /ohmygpu/v1/models[?installed=true]` | all known models with lifecycle state |
| `GET  /ohmygpu/v1/models/{id}` | one model: `state`, `download` progress, `message` (while starting), `error`, `runtime` (backend, pid, port) |
| `POST /ohmygpu/v1/models/pull` `{"model": "<catalog id \| hf:owner/repo/file.gguf \| https://…/file.gguf>", "id"?: "…"}` | 202 downloading (idempotent) |
| `DELETE /ohmygpu/v1/models/{id}` | stop if needed, delete files, forget |
| `POST /ohmygpu/v1/models/{id}/start[?wait=true&timeout=600]` `{"context_length"?, "gpu_layers"?, "threads"?}` | 202 starting, or with `wait` 200 running / 502 error |
| `POST /ohmygpu/v1/models/{id}/stop` | 200 stopped |
| `POST /ohmygpu/v1/shutdown` | graceful shutdown (stops all models) |

### Model lifecycle

```text
not_installed ─pull─▶ downloading ─▶ installed ─start─▶ starting ─▶ running ─stop─▶ stopping ─▶ stopped
                          │                                │           │
                          └──▶ error ◀─────────────────────┴───────────┘   (failed download/start, or crash)
```

`installed`, `stopped` and `error` are all startable; `error.message` says why it failed (including the tail of the llama.cpp log).

## Supported models

v0.1 ships a small, verified catalog of single-file GGUF instruct models (Qwen2.5 0.5B–7B, Qwen3 4B, Llama 3.2 1B/3B, Llama 3.1 8B, Phi-4 mini, Gemma 3 1B–12B, SmolLM2 135M). `omg model catalog` lists them with sizes and whether native tool calling is supported. Any other GGUF can be pulled with `hf:owner/repo/file.gguf` or a direct URL, unsupported but usually fine.

## CLI

```text
omg serve [--host H] [--port P]        run the runtime in the foreground
omg status                              runtime + backend + models summary
omg hardware                            detected hardware
omg model list | catalog | pull <ref> [--id X] | rm <id> | info <id>
omg run <id> [--context-length N] [--gpu-layers N] [--threads N]
omg stop <id>
omg shutdown
omg config [key [value]]
```

Every command except `serve`/`config` uses the Management API; add `--json` for machine-readable output. `--url` / `OHMYGPU_URL` point it at a non-default runtime.

## Configuration & storage

Everything lives under one directory — `~/.config/ohmygpu` by default, or `$OHMYGPU_HOME` / `--data-dir` (so a bundling application can give the runtime a private data dir):

```text
config.toml, registry.json, models/<id>/*.gguf, runtimes/llamacpp/<tag>/, daemon.json
```

```toml
[daemon]
host = "127.0.0.1"          # local only by default; no auth, so keep it that way
port = 10692

[inference]
auto_start = false          # start an installed model on first request instead of 409

[backend.llamacpp]
auto_install = true         # download the official release if llama-server isn't found
release = "latest"          # or a tag like "b10437" for reproducible installs
context_length = 8192
# server_path = "/path/to/llama-server"   # or OHMYGPU_LLAMA_SERVER
# gpu_layers = 999          # only needed for older/custom llama.cpp builds that default to CPU
# threads = 8
startup_timeout_secs = 600
```

Environment overrides: `OHMYGPU_HOME`, `OHMYGPU_HOST`, `OHMYGPU_PORT`, `OHMYGPU_LLAMA_SERVER`, `HF_TOKEN`, `OHMYGPU_LOG` (e.g. `info,llamacpp=debug` to see llama.cpp's own logs).

Managed llama.cpp downloads cover macOS arm64/x64 (Metal), Linux x64/arm64 (CPU, or Vulkan when a GPU is detected), Windows x64 (CPU/Vulkan). For CUDA on Linux, point `server_path` at your own llama.cpp build.

## Embedding the runtime in your app

Ship `ohmygpu-runtime` next to your application, launch it with `--data-dir <your app data> --port <port>`, wait for `GET /ohmygpu/v1/health`, and use the APIs above. Shut it down with `POST /ohmygpu/v1/shutdown` (or SIGTERM); it stops its llama.cpp children. Structured logs go to stderr.

## Development

```bash
make test        # unit + API tests with a mock backend (fast, no GPU/network)
make test-e2e    # real llama.cpp + real model (downloads ~480 MB)
make check       # fmt + clippy
```

Architecture, the engineering assessment of the previous codebase, and the backend decision (llama.cpp vs Candle) are in [`docs/architecture.md`](docs/architecture.md). Deferred code (Candle LLM runtime, Z-Image diffusion) is kept, unbuilt, under [`archive/`](archive/).

## License

Apache-2.0
