# Z-Image Support in ohmygpu

Z-Image is Alibaba's 6.15B parameter text-to-image model using Flow Matching. ohmygpu supports Z-Image-Turbo for fast 8-9 step inference.

## Quick Start

```bash
# Build with Metal (Apple Silicon)
cargo build --release --features metal

# Generate an image (auto-downloads model on first run)
./target/release/ohmygpu generate "A cat sitting on a windowsill"

# With options
./target/release/ohmygpu generate "A cyberpunk city at night" \
    --width 1024 --height 1024 \
    --steps 9 --guidance-scale 5.0 \
    --output cyberpunk.png \
    --seed 42
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model, -m` | `Tongyi-MAI/Z-Image-Turbo` | Model to use |
| `--output, -o` | `~/Documents/ohmygpu/image_<timestamp>.png` | Output file |
| `--width` | 1024 | Image width (must be divisible by 16) |
| `--height` | 1024 | Image height (must be divisible by 16) |
| `--steps, -s` | 9 | Inference steps (8-9 recommended for Turbo) |
| `--guidance-scale, -g` | 5.0 | CFG guidance scale |
| `--negative-prompt` | None | Negative prompt for CFG |
| `--seed` | Random | Random seed for reproducibility |
| `--cpu` | False | Run on CPU (slow) |

## Model Storage

Models are downloaded to `~/.config/ohmygpu/models/`:

```
~/.config/ohmygpu/models/
└── Tongyi-MAI--Z-Image-Turbo/
    └── snapshots/<hash>/
        ├── tokenizer/
        ├── text_encoder/   (~11GB)
        ├── transformer/    (~23GB)
        └── vae/            (~160MB)
```

Total size: ~30GB

## Implementation

ohmygpu uses the Z-Image implementation from `candle-transformers` (added in [PR #3261](https://github.com/huggingface/candle/pull/3261)).

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Z-Image Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Text Encoding (Qwen3-based)                                 │
│     prompt → tokenize → Qwen3 → text embeddings                 │
│                                                                 │
│  2. Noise Initialization                                        │
│     random noise (B, 16, H/8, W/8)                              │
│                                                                 │
│  3. Denoising Loop (9 steps)                                    │
│     ┌─────────────────────────────────────────────┐             │
│     │  Noise Refiner (2 blocks, with modulation)  │             │
│     │  Context Refiner (2 blocks, text only)      │             │
│     │  Main Transformer (30 S3-DiT blocks)        │             │
│     │  → velocity prediction                       │             │
│     │  → Euler step                                │             │
│     └─────────────────────────────────────────────┘             │
│                                                                 │
│  4. VAE Decode                                                  │
│     latents → AutoEncoderKL → RGB image                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **ZImageTransformer2DModel** | 30-layer S3-DiT backbone |
| **ZImageTextEncoder** | Qwen3-based text encoder |
| **AutoEncoderKL** | VAE for latent ↔ image conversion |
| **FlowMatchEulerDiscreteScheduler** | 8-9 step sampling |
| **RopeEmbedder** | 3D rotary position embeddings |

### S3-DiT vs FLUX

| Aspect | FLUX | Z-Image S3-DiT |
|--------|------|----------------|
| **Parameters** | 12B | 6.15B |
| **Stream Type** | Dual → Single | Pure Single |
| **Blocks** | 19 dual + 38 single | 30 single |
| **Hidden Dim** | 3072 | 3840 |
| **Text Encoder** | T5-XXL | Qwen3-4B |
| **Inference Steps** | 4-50 | 8-9 (Turbo) |

## Hardware Requirements

| Platform | VRAM | Notes |
|----------|------|-------|
| Apple Silicon | 32GB+ unified | Recommended for macOS |
| NVIDIA GPU | 24GB+ | A100, RTX 4090 |
| CPU | 64GB+ RAM | Very slow, not recommended |

## Troubleshooting

### "Could not detect model type"

Ensure `transformer/config.json` exists in the model directory. If missing:

```bash
curl -sL "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/transformer/config.json" \
  -o ~/.config/ohmygpu/models/Tongyi-MAI--Z-Image-Turbo/snapshots/*/transformer/config.json
```

### Out of Memory

- Reduce image size: `--width 512 --height 512`
- Close other applications
- Use a machine with more VRAM

### Slow Generation

- Ensure GPU acceleration is enabled (`--features metal` or `--features cuda`)
- Check you're using release build (`--release`)

## References

- [Z-Image Paper](https://arxiv.org/abs/2511.22699)
- [Z-Image HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Candle PR #3261](https://github.com/huggingface/candle/pull/3261)
- [Decoupled-DMD Paper](https://arxiv.org/abs/2511.22677)
