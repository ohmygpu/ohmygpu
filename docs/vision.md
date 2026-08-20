# Vision: OhMyGPU as a base — any model, any GPU box, in 10 minutes

_Recorded 2026-08-20. This is the direction after v0.1; it does not change the v0.1 scope rules in
`CLAUDE.md`, it tells us what the runtime must grow into._

## The goal

> OhMyGPU is a **base** (基座): drop it onto any machine with a GPU — a rented VPS, a cloud
> instance, a box under the desk — point it at a model, and have an OpenAI-compatible endpoint
> serving that model **within 10 minutes**, for **any** open model, including one released today.

Why: new open models ship every day and most of them never get a hosted API. Anyone who wants to
use them has to self-deploy, and today that means hand-assembling GPU image + inference engine +
weights download + engine flags + reverse proxy, differently for every model. OhMyGPU should make
that one command.

Per-model adaptation (engine flags, quant variants, chat/tool templates) is accepted as
hard-and-boring work. What must be *excellent* is the base underneath it, so that adapting a model
is editing a data file, not writing code.

## What "10 minutes" actually means

The budget, on a fresh GPU VPS with ~1 Gbps (~7 GB/min) network:

| Step | Target | Who controls it |
|---|---|---|
| Install the base (`curl … \| sh` or `docker run`) | ≤ 1 min | us — static binary / one image |
| Install the inference backend (vLLM is several GB of Python+CUDA) | ≈ 0 if pre-baked in the image; 3–5 min via `uv` on bare metal | us — **must be pre-baked for the 10-min path** |
| Download weights | bandwidth-bound: 7B FP16 ≈ 2 min, 32B FP16 ≈ 9 min, 70B AWQ ≈ 6 min, 70B FP16 ≈ 20 min | us only via parallel chunks, Xet, mirrors, persistent cache |
| Start + warm-up (CUDA graphs, KV allocation) | llama.cpp seconds; vLLM 1–3 min | us via good defaults (`--max-model-len`, eager mode for huge models) |

So the honest promise is: **≤ ~50 GB of weights on a ≥1 Gbps box with the pre-baked image → under
10 minutes.** Bigger weights are a network fact, not a runtime defect; the runtime must show a
correct ETA instead of hiding it.

Non-goals of the 10-minute promise: ≥200B MoE models that need 4–8×H100/H200 multi-GPU tuning
(wait for a hosted API or use a quantized variant), and architectures that no engine supports yet
(the base must fail fast with a clear "unsupported by llama.cpp/vLLM vX" rather than hang).

## What the base must provide (and where we stand)

| Pillar | What it means | State in the code today (2026-08) | Gap |
|---|---|---|---|
| **1. Backend drivers** | Supervised engine subprocesses behind `RuntimeBackend` / `ModelInstance` | `crates/runtime_llamacpp` only; `ModelManager` holds **one** `Arc<dyn RuntimeBackend>`; `StartSpec` is GGUF-shaped (`model_path`, `gpu_layers`, `threads`) | Add `crates/runtime_vllm` (vLLM is what makes "any model" true: Transformers fallback gives day-0 coverage, GGUF lags). Generalize `StartSpec` (model dir/snapshot + backend id + engine args). Backend registry, per-model backend choice. SGLang later behind the same trait. |
| **2. Recipes as data** ([spec](recipes.md)) | A YAML file per model: repo, backend, variants (quant ↔ VRAM), engine args (`tp`, `max-model-len`, tool/reasoning parser, chat template), known-good engine version, smoke-test prompt | `crates/core/src/catalog.rs` is a **static Rust array** of GGUF entries | Recipe format + loader; a **default recipe** (`vllm --trust-remote-code`, Transformers fallback) so most models need no recipe at all; migrate the catalog into `recipes/*.yaml`; community-contributable. This is how the 体力活 stays boring. Format decision (2026-08-20): **YAML for hand-authored recipes** (comments, multi-line chat templates, what ML contributors already use), **JSON on the wire** (Management API, `omg recipe … --json`, machine-generated skeletons), one serde struct behind both; the loader accepts `.yaml`/`.yml`/`.json`. Not TOML (nested lists read badly), not JSON5/JSONC (a third format nobody asked for). |
| **3. Hardware resolver** | Detect every GPU (count, VRAM, free VRAM, CUDA/driver) and pick the recipe variant that fits — or say exactly what hardware is needed | `hardware.rs` reads only the **first** `nvidia-smi` line; no CUDA version, no free memory | Multi-GPU + CUDA version + free VRAM; resolver `recipe × hardware → variant (quant, tp, ctx)`; "will not fit, needs X" errors before any download starts. |
| **4. HF snapshot downloader** | Multi-file safetensors repos, parallel chunks, Xet, token, mirror (hf-mirror), checksums, resume | `download.rs` resumable; `ModelSource::HuggingFace { repo, file }` is **single-file** | Repo snapshot mode; parallelism; mirror config; keep the resumable core. |
| **5. Remote-safe mode** | Bind non-loopback **only** with an API key; key on `/v1/*` **and** `/ohmygpu/v1/*` (management can delete/shutdown); TLS via reverse proxy or built-in rustls | `daemon.host` exists; `server.rs` only **warns** when non-loopback; no auth | `--bind`, `--api-key` / `OHMYGPU_API_KEY`, refuse non-loopback without a key; CLI `--host` / `OHMYGPU_HOST` (the CLI is already a thin HTTP client, so `omg --host vps model pull … && omg run …` is nearly free). |
| **6. Distribution** | One CUDA container image (vLLM + llama-server + runtime pre-baked) + `install.sh` for bare VPS + Linux static binaries + systemd unit | `make build` only; no Dockerfile, no installer | Dockerfile, GitHub Release artifacts (linux x86_64/aarch64, macOS arm64), `install.sh`, unit template. |

Cross-cutting: multi-model across multiple GPUs (GPU assignment in `ModelManager`), health/auto-restart
(supervision exists), `/metrics`, structured logs.

## What does not change

The v0.1 rules stay the foundation:

1. **One inference pipeline** — vLLM's OpenAI-ish wire format stays inside `runtime_vllm/wire.rs`; the
   daemon still only sees `ohmygpu_inference` types.
2. **Explicit lifecycle** through `ModelManager`, generation checks on background tasks.
3. **CLI stays thin** — remote mode is the CLI talking to a remote Management API, nothing more.
4. **No unimplemented API fields** claimed.
5. **Fast offline tests** with the mock backend; real GPU runs behind `OHMYGPU_E2E=1`.

Local-first remains: a laptop with `127.0.0.1` and llama.cpp is still the default experience; the
base just has to run identically on a rented GPU box.

## Proposed order (each step ships on its own)

1. **Remote mode** — bind + API key + CLI `--host`. Smallest change; immediately lets today's
   llama.cpp runtime be used on a GPU VPS.
2. **Linux packaging** — Release binaries, `install.sh`, Dockerfile (llama.cpp only at first).
3. **Hardware: multi-GPU detection** + **HF snapshot download**.
4. **`crates/runtime_vllm`** — generalize `StartSpec`, backend registry, vLLM wire adapter.
5. **Recipes** — format, loader, default recipe, migrate the catalog to `recipes/*.yaml`. _Started 2026-08-20: schema v1 types + validation + JSON Schema + example recipes are in (`docs/recipes.md`); the resolver and the catalog migration are next._
6. **Pre-baked CUDA image** + a **10-minute SLA test matrix**: nightly, N models on a real GPU,
   `pull → start → smoke test`, timed, published. The matrix is both the quality gate and the proof
   of the promise.

## Landscape (why the base, not another wrapper)

GPUStack, Xinference, RamaLama and Ollama all do some of "multi-backend + model management".
OhMyGPU's edge has to be: a single static Rust binary with no Python in the control plane,
embeddable/headless, recipes anyone can contribute, the Responses API as the canonical surface, and
a **measured** 10-minute promise.
