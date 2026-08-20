# Recipes — schema v1

A **recipe** is the declarative, per-model knowledge that turns "deploy any model" into data
instead of code: where the weights live, which backend runs them, which *variants* exist
(precision / quantization / file format), what hardware each needs, the engine settings that are
required, how the model talks (chat template, tool/reasoning parsers), and how to prove it works
(smoke tests).

Recipes are the "hard-and-boring" part of the [base vision](vision.md) made boring on purpose:
adapting a new model is editing a YAML file, reviewable in a PR, contributable by anyone.

- **Types and validation:** `crates/core/src/recipe.rs` (`ohmygpu_core::recipe::Recipe`)
- **JSON Schema (generated):** `schemas/recipe-v1.json`
- **Examples / test fixtures:** `recipes/*.yaml`
- **Format policy:** hand-authored in **YAML**; **JSON** on the wire (Management API,
  machine-generated skeletons). Both load into the same struct; the loader accepts `.yaml`,
  `.yml`, `.json`.

Status (2026-08-20): the format, loader and validation exist and are tested. The hardware-aware
resolver, the HF-metadata default recipe, and the backends' use of `ResolvedVariant` are the next
steps (see [Roadmap](#roadmap-hooks)).

---

## 1. The minimal recipe

```yaml
# yaml-language-server: $schema=../schemas/recipe-v1.json
schema_version: 1
id: smollm2-135m-instruct
source:
  hf: bartowski/SmolLM2-135M-Instruct-GGUF
  file: SmolLM2-135M-Instruct-Q8_0.gguf
```

Three things are mandatory: `schema_version`, `id`, and *some way to find weights* (a recipe-level
`source`, or a `source` on every variant). Everything else is optional. Here the backend
(`llamacpp`) and format (`gguf`) are inferred from the `.gguf` file, and the recipe yields one
implicit variant named `default`.

A safetensors repo with nothing else — `source: { hf: Qwen/Qwen3-8B }` — yields one `vllm`
variant. That is the shape the runtime will generate on its own for models without a recipe.

## 2. A full recipe

See [`recipes/qwen3-32b.yaml`](../recipes/qwen3-32b.yaml): one model, five variants across two
backends (`bf16`, `bf16-tp2`, `fp8`, `awq` on vLLM; `q4_k_m` on llama.cpp), shared engine
settings, parsers, sampling defaults and smoke tests. Condensed:

```yaml
schema_version: 1
id: qwen3-32b
family: qwen3
context_length: 40960
capabilities: { tools: true, reasoning: true }
source: { hf: Qwen/Qwen3-32B }                 # inherited unless a variant names a new repo
chat: { tool_call_parser: hermes, reasoning_parser: qwen3 }
engine:                                         # shared; variants merge on top
  context_length: 32768
  vllm: { min_version: "0.8.5", enable_prefix_caching: true }
variants:                                       # preference order
  - { name: bf16,  backend: vllm, quantization: bf16, size_gb: 65.5,
      requires: { vram_gb: 80, min_compute_capability: "8.0" } }
  - { name: fp8,   backend: vllm, source: { hf: Qwen/Qwen3-32B-FP8 }, quantization: fp8,
      size_gb: 34, requires: { vram_gb: 48, min_compute_capability: "8.9" } }
  - { name: q4_k_m, backend: llamacpp,
      source: { hf: unsloth/Qwen3-32B-GGUF, file: Qwen3-32B-Q4_K_M.gguf },
      size_gb: 19.8, requires: { vram_gb: 24, ram_gb: 24 },
      engine: { context_length: 16384, llamacpp: { flash_attention: true } } }
tests:
  - { prompt: "Reply with exactly: pong", expect_contains: pong }
  - { prompt: "What is the weather in Paris? Use the tool.", expect_tool_call: true }
```

---

## 3. Field reference

All keys are `snake_case`. Unknown keys are an error everywhere except inside `metadata`.

### 3.1 Top level

| Field | Type | Req. | Meaning |
|---|---|---|---|
| `schema_version` | int | **yes** | Always `1`. Bumped only for breaking changes. |
| `id` | string | **yes** | Stable model id: `^[a-z0-9][a-z0-9._-]*$`. What clients send as `model`. |
| `display_name` | string | | Human name; defaults to `id`. |
| `family` | string | | Architecture line (`qwen3`, `llama-3.1`, `gemma-3`). Grouping / family defaults. |
| `description` | string | | One or two sentences for `omg model show`. |
| `license` | string | | SPDX where possible (`apache-2.0`, `llama3.1`). |
| `homepage` | url | | Model card / project page. |
| `tags` | [string] | | Discovery tags (`reasoning`, `tools`, `coder`, `tiny`). |
| `kind` | enum | | `llm` (default). Embedding / reranker kinds later, without a schema bump. |
| `context_length` | int | | Native max context (`config.json` → `max_position_embeddings`). Upper bound for `engine.context_length`. |
| `capabilities` | object | | `tools`, `vision`, `reasoning` — all default `false`. Drives API validation (tool definitions are only accepted when `tools: true`). **Never claim what was not verified.** |
| `source` | Source | | Default weights location, inherited by variants. |
| `backend` | enum | | Default backend (`vllm` \| `llamacpp`), inherited by variants. |
| `chat` | Chat | | Template override, parsers, stop strings. |
| `generation_defaults` | object | | Sampling defaults applied when a request omits the parameter. |
| `engine` | Engine | | Shared engine settings; variants merge on top. |
| `variants` | [Variant] | | Preference-ordered concrete ways to run the model. Empty = one implicit `default` variant. |
| `tests` | [SmokeTest] | | Run after start and by the nightly SLA matrix. |
| `verified` | [Verification] | | Provenance: real runs that proved a variant works. |
| `metadata` | map | | Free-form extension data for tooling; ignored by the runtime. |

### 3.2 `source`

Exactly one of `hf` or `url`.

| Field | Meaning |
|---|---|
| `hf` | Hugging Face repo id `owner/name`. |
| `revision` | Hub git revision (branch / tag / commit). Default: the repo's default branch. |
| `file` | One file in the repo — the GGUF to load. Sharded GGUF: name the first shard (`…-00001-of-00005.gguf`). |
| `include` | Globs of repo files to download. Empty = downloader default (every weight / tokenizer / config file; no `original/`, no legacy `.bin`/`.pth` when safetensors exist). |
| `exclude` | Globs to skip, applied after `include`. |
| `url` | Direct http(s) URL to one file (self-hosted GGUF). Cannot be combined with any `hf` field. |

**Merge rule** (recipe `source` → variant `source`): a variant that names a new *location*
(`hf` or `url`) **replaces** the recipe source entirely; a variant that only sets
`file` / `include` / `exclude` / `revision` **refines** it and inherits the rest. This makes the
common GGUF layout tidy — repo on the recipe, one `file:` per quant variant — while a variant that
points at a different repo (`-AWQ`, `-FP8`) starts clean.

### 3.3 `chat`

| Field | Meaning |
|---|---|
| `template.file` / `template.inline` | Jinja chat template override (exactly one). `file` is relative to the recipe. |
| `tool_call_parser` | How tool calls appear in output. Vocabulary = vLLM `--tool-call-parser` names (`hermes`, `llama3_json`, `mistral`, `qwen3_coder`, `deepseek_v3`, `glm45`, `kimi_k2`, …). llama.cpp derives this from the template and ignores the field. |
| `reasoning_parser` | Vocabulary = vLLM `--reasoning-parser` names (`deepseek_r1`, `qwen3`, `glm45`, …). |
| `stop` | Extra stop strings applied to every request. |

We deliberately reuse vLLM's parser vocabulary instead of inventing one: it is the widest-known
naming in the ecosystem, and a recipe field with a private vocabulary would be a new thing for
contributors to learn.

### 3.4 `generation_defaults`

`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`,
`frequency_penalty`, `max_output_tokens`. Applied only when the request does not set the value.
Model cards usually state these (Qwen3 thinking mode: `0.6 / 0.95 / 20`).

### 3.5 `engine`

Backend-agnostic scalars every backend maps, plus one typed section per backend holding its
*stable, high-value* flags and an escape hatch. New exotic flags go into `extra_args` first and are
promoted to typed fields once common.

| Field | vLLM | llama.cpp |
|---|---|---|
| `context_length` | `--max-model-len` | `--ctx-size` |
| `max_concurrency` | `--max-num-seqs` | `--parallel` |
| `tensor_parallel` | `--tensor-parallel-size` | ignored (layer split is automatic) |

`engine.vllm`: `min_version` (semver), `dtype`, `quantization` (engine override — rarely needed,
vLLM reads `quantization_config` from `config.json`), `model_impl` (`auto` \| `vllm` \|
`transformers` — the day-0 fallback), `trust_remote_code`, `gpu_memory_utilization` (0,1],
`max_num_batched_tokens`, `enable_prefix_caching`, `kv_cache_dtype`, `enforce_eager`, `tokenizer`,
`extra_args`, `env`.

`engine.llamacpp`: `min_release` (`b6000`), `gpu_layers`, `threads`, `batch_size`,
`ubatch_size`, `flash_attention`, `cache_type_k`, `cache_type_v`, `mmap`, `mlock`,
`extra_args`, `env`.

**Merge rule** (recipe `engine` → variant `engine`): scalars override when set, `extra_args`
**append** (recipe first), `env` overlays (variant wins).

### 3.6 `variants[]`

| Field | Meaning |
|---|---|
| `name` | Unique in the recipe; id character rules (`bf16`, `fp8`, `awq`, `q4_k_m`, `bf16-tp2`). Users can force one: `omg run qwen3-32b@awq`. |
| `description` | Why it exists / when to pick it. |
| `backend` | Default: recipe `backend`, else inferred (GGUF source → `llamacpp`, otherwise `vllm`). |
| `format` | `safetensors` \| `gguf`. Default: inferred from source / backend. |
| `source` | Merged over the recipe source (§3.2). |
| `quantization` | Human label (`bf16`, `fp16`, `fp8`, `awq`, `gptq`, `int4`, `q4_k_m`, `q8_0`). Display and `--quant` filters; engines read the real method from the weights. |
| `size_gb` | Download size (10⁹ bytes) for ETAs and disk checks. |
| `requires` | Hardware needs (below). Missing numbers mean *unknown* — not filtered on, with a warning. |
| `engine` | Merged over the recipe engine (§3.5). |

`requires`:

| Field | Meaning |
|---|---|
| `vram_gb` | **Total** GPU memory across the GPUs used (weights + KV at the variant's context length). Unified memory on Apple Silicon. |
| `ram_gb` | System RAM (CPU inference, mmap'd GGUF). |
| `gpus` | GPUs used (default 1). Must equal `engine.tensor_parallel` for vLLM. |
| `accelerators` | Subset of `cuda`, `metal`, `rocm`, `cpu`. Empty = whatever the backend supports. |
| `platforms` | Subset of `linux`, `macos`, `windows`. Empty = any the backend supports. |
| `min_compute_capability` | NVIDIA `major.minor` as a **string** (`"7.5"` Turing, `"8.0"` Ampere, `"8.9"` Ada, `"9.0"` Hopper). bf16 needs 8.0, fp8 8.9. The loader also accepts the number `8.0` and normalises it, but quote it. |

### 3.7 `tests[]`

`prompt` (required), `name`, `system`, `max_tokens` (default 32), and one expectation:
`expect_contains` (case-insensitive substring), `expect_regex`, or `expect_tool_call: true` (the
runtime sends a fixed `get_weather(city)` tool definition and requires a tool call — use on recipes
that claim `capabilities.tools`). No expectation = "200 and non-empty text".

### 3.8 `verified[]`

`date` (`YYYY-MM-DD`), `variant` (must exist), `hardware` (`1x RTX 4090 24GB`, `Apple M3 Max
64GB`), `backend_version` (`vllm 0.10.1`, `llama.cpp b6234`), `by`, `notes`. Only real runs go
here — the SLA matrix will append entries automatically; hand-written ones should say who ran it.

---

## 4. Semantics

### 4.1 Resolution (`Recipe::resolved_variants()`)

For each variant, in recipe order:

1. `source` = merge(recipe.source, variant.source) — error if neither exists.
2. `backend` = variant.backend → recipe.backend → inferred (GGUF source → `llamacpp`, else `vllm`).
3. `format` = variant.format → inferred (GGUF source → `gguf`, else the backend's default).
4. `engine` = merge(recipe.engine, variant.engine).

No variants → one variant named `default` from the recipe-level fields. The result is a
`ResolvedVariant` — what the hardware resolver and the backends consume. The daemon never reads
raw `Variant`s.

### 4.2 Validation (what `from_yaml` / `from_json` enforce)

- `schema_version == 1`; `id` matches the id rules; variant names unique and id-shaped.
- `source`: exactly one of `hf`/`url`; `hf` is `owner/name`; `url` is http(s) and carries no
  `hf`-only fields; `file` is a relative in-repo path.
- GGUF variants (`format: gguf`, and therefore every `llamacpp` variant) must name the file to
  load (`source.file`, a `.gguf` `url`, or an `include` pattern mentioning `.gguf`);
  `llamacpp` with non-GGUF weights is an error; a `.gguf` file with `format: safetensors` is an
  error.
- `engine.context_length ≤ context_length`; vLLM `tensor_parallel == requires.gpus` when both set;
  `vllm.min_version` is semver; `llamacpp.min_release` is `b<digits>`;
  `gpu_memory_utilization ∈ (0,1]`; `model_impl ∈ {auto, vllm, transformers}`; positive sizes.
- `chat.template`: exactly one of `file`/`inline`. `generation_defaults` in range.
- `tests`: non-empty prompt. `verified`: valid date, existing variant.
- Unknown keys anywhere (except `metadata`) → parse error naming the key.

### 4.3 Hardware resolver (contract for the next step)

Given the detected hardware (`HardwareInfo`: accelerator, GPUs with total/free VRAM, compute
capability, RAM, platform) and a list of `ResolvedVariant`s, plus optional user constraints
(`@variant`, `--backend`, `--quant`):

1. Drop variants whose `platforms` / `accelerators` exclude the machine.
2. Drop variants whose backend is neither available nor installable here.
3. Drop variants needing more `gpus` than present, or `min_compute_capability` above the lowest
   GPU used.
4. Drop variants whose `vram_gb` exceeds the sum of free VRAM of the best `gpus` GPUs (and whose
   per-GPU share exceeds any single GPU), or whose `ram_gb` exceeds free RAM. Unknown numbers do
   not filter; they warn.
5. Apply user constraints.
6. Pick the **first remaining in recipe order** — author preference wins, not "biggest that fits".
7. If nothing remains, fail **before downloading anything** with one line per variant saying
   why (`bf16: needs 80 GB VRAM, you have 24 GB · awq: needs compute capability 7.5, you have
   7.0 · …`).

### 4.4 The default recipe (models with no recipe file)

The runtime will synthesise a `Recipe` from Hub metadata when asked to run a bare repo id:

- `id` ← slug of the repo name; `family` ← `config.json` `model_type`; `context_length` ←
  `max_position_embeddings`.
- Repo has `.gguf` files → one `llamacpp` variant per quant file (`name` = quant label from the
  filename), `size_gb` from file sizes.
- Otherwise one `vllm` variant: `format: safetensors`, `quantization` from
  `quantization_config.quant_method` if present, `size_gb` from file sizes, `vram_gb` ≈ size × 1.2
  as a first estimate, `trust_remote_code` only when `auto_map` is present in `config.json`.
- `capabilities` all `false`, no parsers — the honest default. The `tests` list gets the `pong`
  test.

Because every field of that synthesis exists in the schema, "give this model a recipe" is a
diff on top of what the runtime already inferred (`omg recipe init <hf-repo>` will emit it).

---

## 5. Conventions

- **ids**: lowercase, `.` `_` `-`; mirror the upstream name (`qwen3-32b`, `llama-3.1-8b-instruct`,
  `gemma-3-12b-it`). No vendor prefix unless needed to disambiguate.
- **variant names**: the precision / quant label, plus a suffix for topology (`bf16-tp2`) or
  context (`q4_k_m-ctx128k`) when that is the distinguishing thing.
- **quantization labels**: `bf16`, `fp16`, `fp8`, `awq`, `gptq`, `int4`, `int8`, `nvfp4`,
  `q2_k` … `q8_0`, `iq4_xs` (llama.cpp names lowercased).
- **sizes**: decimal GB (10⁹), one decimal place, measured from the Hub file list.
- **VRAM numbers**: measured with the variant's `engine.context_length`, rounded *up* to the next
  card size people actually buy (`6`, `8`, `12`, `16`, `24`, `48`, `80`, `96`).
- **parsers**: vLLM vocabulary (§3.3).
- **comments** explain the *why* (`# tp=2 because 72 GB does not fit one A100`).

## 6. Versioning and compatibility

- `schema_version` is bumped only for **breaking** changes. Adding optional fields, new enum
  values (backends, accelerators, kinds) or new typed engine flags does **not** bump it.
- A runtime refuses recipes with a newer `schema_version` with a clear message; a newer recipe
  with a new *optional* field on an old runtime fails with "unknown field X" — that is the
  intended trade-off (typo safety over silent ignoring). `metadata` is the place for anything the
  runtime may not know.
- The JSON Schema is generated from the Rust types and checked in; a unit test fails when it is
  stale. Regenerate with `make recipe-schema`.

## 7. Tooling

- Editor validation / completion: first line of every recipe
  `# yaml-language-server: $schema=../schemas/recipe-v1.json` (or the published URL).
- Rust: `Recipe::from_path / from_yaml / from_json`, `to_yaml / to_json_pretty`,
  `resolved_variants()`, `validate()`, `Recipe::json_schema()`.
- Tests: `cargo test -p ohmygpu_core recipe` — every file in `recipes/` must load, resolve,
  and round-trip through JSON and YAML.

## 8. Roadmap hooks

What consumes recipes next (see [vision.md](vision.md), step 5):

1. `ModelManager` loads `recipes/` (bundled + `$OHMYGPU_HOME/recipes/` + API-pushed) and
   replaces the static `catalog.rs` array; `GET /ohmygpu/v1/catalog` returns recipes.
2. The hardware resolver (§4.3) picks a `ResolvedVariant`; `StartSpec` is generalised to carry it.
3. `runtime_llamacpp` maps `engine` + `engine.llamacpp` to `llama-server` flags; `runtime_vllm`
   maps `engine` + `engine.vllm` + `chat` parsers to `vllm serve` flags.
4. `omg recipe init|show|validate`, `POST /ohmygpu/v1/recipes`.
5. The nightly SLA matrix runs `tests[]` per variant on real GPUs and appends `verified[]`.

Open questions for v2 (not blocking): LoRA adapters as a variant dimension; multi-node
topologies; embedding / reranker kinds; recipe inheritance across files (`extends:`) — so far a
copy is simpler than a dependency.
