# Models in OhMyGPU Runtime

## Format

v0.1 runs **GGUF** files through llama.cpp. One model = one GGUF file (split/sharded
GGUFs are not supported yet).

## Storage layout

```text
$OHMYGPU_HOME (default ~/.config/ohmygpu)/
├── registry.json                       # installed models: id → source, path, size, capabilities
└── models/
    └── <model-id>/
        └── <file>.gguf                 # e.g. qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Downloads stream to `<file>.gguf.part` and are renamed on completion; interrupted
downloads resume with HTTP `Range` on the next pull.

## Model ids and references

| Reference | Example | Installed id |
|-----------|---------|--------------|
| Catalog id | `qwen2.5-1.5b-instruct` | same |
| Hugging Face file | `hf:bartowski/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct-Q4_K_M.gguf` | derived from the file name (`smollm2-360m-instruct-q4_k_m`) unless `id` is given |
| Hugging Face URL | `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf` | derived / `id` |
| Direct URL | `https://models.example.com/my-model.gguf` | derived / `id` |

Ids are lowercase `[a-z0-9._-]`. Gated Hugging Face repos need a token
(`HF_TOKEN` env or `models.hf_token` in `config.toml`).

## Curated catalog (v0.1)

`omg model catalog` / `GET /ohmygpu/v1/catalog`. All entries are ungated,
single-file GGUFs verified against the Hugging Face API; the `tools` flag means
llama.cpp has a native tool-call parser for the model family.

| id | repo | quant | approx size | tools |
|----|------|-------|-------------|-------|
| smollm2-135m-instruct | bartowski/SmolLM2-135M-Instruct-GGUF | Q8_0 | 0.14 GB | no |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct-GGUF | Q4_K_M | 0.49 GB | yes |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct-GGUF | Q4_K_M | 1.1 GB | yes |
| qwen2.5-3b-instruct | Qwen/Qwen2.5-3B-Instruct-GGUF | Q4_K_M | 2.1 GB | yes |
| qwen2.5-7b-instruct | bartowski/Qwen2.5-7B-Instruct-GGUF | Q4_K_M | 4.7 GB | yes |
| qwen3-4b-instruct | unsloth/Qwen3-4B-Instruct-2507-GGUF | Q4_K_M | 2.5 GB | yes |
| llama-3.2-1b-instruct | bartowski/Llama-3.2-1B-Instruct-GGUF | Q4_K_M | 0.8 GB | yes |
| llama-3.2-3b-instruct | bartowski/Llama-3.2-3B-Instruct-GGUF | Q4_K_M | 2.0 GB | yes |
| llama-3.1-8b-instruct | bartowski/Meta-Llama-3.1-8B-Instruct-GGUF | Q4_K_M | 4.9 GB | yes |
| phi-4-mini-instruct | bartowski/microsoft_Phi-4-mini-instruct-GGUF | Q4_K_M | 2.5 GB | no |
| gemma-3-1b-it | ggml-org/gemma-3-1b-it-GGUF | Q4_K_M | 0.8 GB | no |
| gemma-3-4b-it | ggml-org/gemma-3-4b-it-GGUF | Q4_K_M | 2.5 GB | no |
| gemma-3-12b-it | ggml-org/gemma-3-12b-it-GGUF | Q4_K_M | 7.3 GB | no |

The catalog is compiled into the runtime (`crates/core/src/catalog.rs`).

## Lifecycle

See `docs/architecture.md`. States: `not_installed`, `downloading`, `installed`,
`starting`, `running`, `stopping`, `stopped`, `error`.
