# Model Storage in ohmygpu

## Storage Location

Models are stored in `~/.config/ohmygpu/models/` by default. This can be configured in `~/.config/ohmygpu/config.toml`:

```toml
[models]
storage_path = "~/.config/ohmygpu/models"
```

## Directory Structure

Models downloaded from HuggingFace use this structure:

```
~/.config/ohmygpu/models/
├── Tongyi-MAI--Z-Image-Turbo/
│   ├── blobs/                    # Actual file content (deduplicated)
│   ├── refs/
│   └── snapshots/
│       └── <commit_hash>/
│           ├── tokenizer/
│           ├── text_encoder/
│           ├── transformer/
│           └── vae/
└── other-org--other-model/
```

## HuggingFace `models--` Prefix

The `hf-hub` Rust crate automatically adds a `models--` prefix when downloading (e.g., `models--Tongyi-MAI--Z-Image-Turbo`).

**ohmygpu handles this automatically:**
- After download, the directory is renamed to remove the prefix
- `models--Tongyi-MAI--Z-Image-Turbo` → `Tongyi-MAI--Z-Image-Turbo`

This keeps the models directory clean and consistent.

## Model Naming Convention

| HuggingFace Repo | Local Directory |
|------------------|-----------------|
| `Tongyi-MAI/Z-Image-Turbo` | `Tongyi-MAI--Z-Image-Turbo` |
| `microsoft/phi-2` | `microsoft--phi-2` |
| `meta-llama/Llama-2-7b` | `meta-llama--Llama-2-7b` |

The `/` is replaced with `--` for filesystem compatibility.

## Required Files

### Diffusion Models (Z-Image, FLUX)

```
model-dir/snapshots/<hash>/
├── transformer/
│   ├── config.json              # Required for model detection
│   └── *.safetensors            # Model weights
├── text_encoder/
│   ├── config.json
│   └── *.safetensors
├── vae/
│   ├── config.json
│   └── *.safetensors
└── tokenizer/
    └── tokenizer.json
```

### LLM Models

```
model-dir/snapshots/<hash>/
├── config.json
├── tokenizer.json
└── *.safetensors
```

## Model Detection

Models are detected by examining `transformer/config.json`:

| Config Contains | Model Type |
|-----------------|------------|
| `"ZImage"` | Z-Image |
| `"Flux"` | FLUX |

For FLUX, presence of `flux1-dev.safetensors` or `ae.safetensors` also indicates FLUX.
