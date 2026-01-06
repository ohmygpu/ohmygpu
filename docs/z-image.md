# Z-Image Support in ohmygpu

This document outlines the architecture analysis and implementation plan for adding Z-Image support to ohmygpu using native Rust/Candle.

## Table of Contents

1. [FLUX Architecture (Reference)](#flux-architecture-reference)
2. [Z-Image S3-DiT Architecture](#z-image-s3-dit-architecture)
3. [Architecture Comparison](#architecture-comparison)
4. [Porting Plan](#porting-plan)
5. [runtime_diffusion Crate Structure](#runtime_diffusion-crate-structure)
6. [Implementation Phases](#implementation-phases)

---

## FLUX Architecture (Reference)

FLUX is a 12B rectified flow transformer implemented in candle. Understanding it is key to porting Z-Image.

### Model Configuration

```
Model: FLUX.1-dev / FLUX.1-schnell
Parameters: 12B
Hidden Size: 3072
Attention Heads: 24
Double Stream Blocks: 19
Single Stream Blocks: 38
Context Dim: 4096 (T5)
Vec Dim: 768 (CLIP)
Axes Dim: [16, 56, 56] (RoPE)
```

### FLUX Block Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUX Forward Pass                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Inputs:                                                        │
│    img: (batch, seq_len, 64)      - latent patches              │
│    txt: (batch, seq_len, 4096)    - T5 embeddings               │
│    img_ids, txt_ids: (batch, seq_len, 3) - position indices     │
│    timesteps: (batch,)            - diffusion time              │
│    y: (batch, 768)                - CLIP embedding              │
│                                                                 │
│  1. Positional Embeddings (RoPE)                                │
│     pe = rope(concat(txt_ids, img_ids))                         │
│                                                                 │
│  2. Input Projections                                           │
│     img = img_in(img)    # 64 → 3072                            │
│     txt = txt_in(txt)    # 4096 → 3072                          │
│                                                                 │
│  3. Condition Vector                                            │
│     vec = time_in(timestep_emb) + vector_in(y) + guidance_in(g) │
│                                                                 │
│  4. Double Stream Blocks (×19)  ← DUAL STREAM                   │
│     ┌─────────────────────────────────────────────┐             │
│     │  img_stream    │    txt_stream              │             │
│     │  ────────────  │    ────────────            │             │
│     │  Modulation    │    Modulation              │             │
│     │  LayerNorm     │    LayerNorm               │             │
│     │  Self-Attn ────┼──→ Cross-Attn (concat Q,K,V)             │
│     │  MLP           │    MLP                     │             │
│     └─────────────────────────────────────────────┘             │
│                                                                 │
│  5. Concatenate: x = [txt; img]                                 │
│                                                                 │
│  6. Single Stream Blocks (×38)  ← SINGLE STREAM                 │
│     ┌─────────────────────────────────────────────┐             │
│     │  Modulation → LayerNorm → QKV+MLP           │             │
│     │  Self-Attention → Concat(attn, mlp) → Proj  │             │
│     └─────────────────────────────────────────────┘             │
│                                                                 │
│  7. Extract image portion: img = x[txt_len:]                    │
│                                                                 │
│  8. Final Layer (AdaLN + Linear)                                │
│                                                                 │
│  Output: noise prediction (batch, seq_len, 64)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key FLUX Components in Candle

| Component | Location | Purpose |
|-----------|----------|---------|
| `DoubleStreamBlock` | `model.rs:356-440` | Parallel text/image processing with cross-attention |
| `SingleStreamBlock` | `model.rs:442-493` | Unified stream processing |
| `SelfAttention` | `model.rs:180-230` | Attention with QK-Norm and RoPE |
| `Modulation1/2` | `model.rs:120-160` | AdaLN conditioning (shift, scale, gate) |
| `MlpEmbedder` | `model.rs:95-115` | Timestep/CLIP embedding |
| `EmbedNd` | `model.rs:60-90` | Multi-axis RoPE |
| `AutoEncoder` | `autoencoder.rs` | VAE encode/decode |

---

## Z-Image S3-DiT Architecture

Z-Image uses a **Scalable Single-Stream DiT (S3-DiT)** - fundamentally different from FLUX's dual-stream approach.

### Model Configuration

```
Model: Z-Image-Turbo / Z-Image-Base / Z-Image-Edit
Parameters: 6.15B
Hidden Size: 3840
Attention Heads: 32
Layers: 30 (all single-stream!)
FFN Intermediate: 10240
Spatial Resolution: (dt, dh, dw) = (32, 48, 48)
Text Encoder: Qwen3-4B (bilingual)
VAE: Flux VAE (reused)
```

### S3-DiT Block Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    S3-DiT Forward Pass                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Inputs:                                                        │
│    All tokens concatenated into SINGLE sequence:                │
│    x = [text_tokens | image_vae_tokens | (optional: ref_img)]   │
│                                                                 │
│  1. Modality-Specific Processors (2 transformer blocks each)    │
│     text_tokens  → TextProcessor  → aligned text                │
│     image_tokens → ImageProcessor → aligned image               │
│                                                                 │
│  2. Token Concatenation                                         │
│     x = concat(text, image) at sequence level                   │
│                                                                 │
│  3. 3D Unified RoPE                                             │
│     - Image tokens: expand across (height, width)               │
│     - Text tokens: increment temporally                         │
│     - For editing: ref/target offset by 1 temporal unit         │
│                                                                 │
│  4. Single-Stream Backbone (×30)  ← ALL SINGLE STREAM           │
│     ┌─────────────────────────────────────────────┐             │
│     │  [text | image] → unified sequence          │             │
│     │                                             │             │
│     │  QK-Norm (attention stability)              │             │
│     │  Sandwich-Norm (signal amplitude control)   │             │
│     │  RMSNorm (throughout)                       │             │
│     │                                             │             │
│     │  Attention (all tokens attend to all)       │             │
│     │  FFN with GELU                              │             │
│     │                                             │             │
│     │  Conditioning: scale/gate via projection    │             │
│     │    - Shared down-projection                 │             │
│     │    - Layer-specific up-projections          │             │
│     └─────────────────────────────────────────────┘             │
│                                                                 │
│  5. Extract image portion                                       │
│                                                                 │
│  6. Final projection                                            │
│                                                                 │
│  Output: velocity prediction v_t = x_1 - x_0                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key S3-DiT Innovations

1. **Pure Single-Stream**: No dual-stream blocks at all - text and image fuse immediately
2. **Modality Processors**: Lightweight 2-block transformers align modalities before fusion
3. **Shared Conditioning Projection**: Reduces parameters vs. per-layer conditioning
4. **Sandwich-Norm**: Additional normalization at attention/FFN boundaries
5. **Qwen3-4B Text Encoder**: Better bilingual (EN/CN) support than T5

---

## Architecture Comparison

| Aspect | FLUX | Z-Image S3-DiT |
|--------|------|----------------|
| **Parameters** | 12B | 6.15B |
| **Stream Type** | Dual → Single | Pure Single |
| **Double Blocks** | 19 | 0 |
| **Single Blocks** | 38 | 30 |
| **Hidden Dim** | 3072 | 3840 |
| **Heads** | 24 | 32 |
| **Text Encoder** | T5-XXL (4096) | Qwen3-4B |
| **CLIP** | Yes (768) | No |
| **VAE** | Flux VAE | Flux VAE (same!) |
| **Normalization** | RMSNorm + QK-Norm | RMSNorm + QK-Norm + Sandwich-Norm |
| **Conditioning** | Per-block modulation | Shared down-proj + per-layer up-proj |
| **Positional** | 3D RoPE | 3D Unified RoPE |
| **Inference Steps** | 4 (schnell) / 50 (dev) | 8 (turbo) |

### What Can Be Reused from FLUX

| Component | Reusability | Notes |
|-----------|-------------|-------|
| **AutoEncoder (VAE)** | ✅ 100% | Z-Image uses exact same Flux VAE |
| **RoPE implementation** | ✅ 90% | Same concept, minor 3D unification changes |
| **QK-Norm** | ✅ 100% | Identical implementation |
| **RMSNorm** | ✅ 100% | Standard implementation |
| **Sampling/Denoising** | ✅ 80% | Flow matching is same, schedule differs |
| **SingleStreamBlock** | 🔄 70% | Need to add Sandwich-Norm |
| **DoubleStreamBlock** | ❌ 0% | Not used in Z-Image |
| **Modulation** | 🔄 50% | Different projection structure |
| **Text Encoder** | ❌ 0% | Need Qwen3-4B instead of T5 |

---

## Porting Plan

### Phase 1: Core S3-DiT Block

```rust
// New struct for Z-Image
pub struct S3DiTBlock {
    // Pre-attention
    norm1: RMSNorm,
    sandwich_norm1_in: RMSNorm,   // NEW: Sandwich norm
    sandwich_norm1_out: RMSNorm,  // NEW: Sandwich norm

    // Attention
    qkv: Linear,
    q_norm: RMSNorm,  // QK-Norm
    k_norm: RMSNorm,  // QK-Norm
    proj: Linear,

    // Pre-FFN
    norm2: RMSNorm,
    sandwich_norm2_in: RMSNorm,   // NEW
    sandwich_norm2_out: RMSNorm,  // NEW

    // FFN
    ffn_up: Linear,      // hidden → intermediate (10240)
    ffn_down: Linear,    // intermediate → hidden

    // Conditioning (shared structure)
    cond_up: Linear,     // Layer-specific up-projection

    num_heads: usize,    // 32
    hidden_size: usize,  // 3840
}

impl S3DiTBlock {
    pub fn forward(
        &self,
        x: &Tensor,           // (batch, seq_len, 3840) - unified stream
        cond: &Tensor,        // (batch, cond_dim) - timestep + other conditions
        rope: &Tensor,        // Positional encoding
        shared_down: &Tensor, // Shared down-projected conditioning
    ) -> Result<Tensor> {
        // 1. Get scale/gate from conditioning
        let (scale1, gate1, scale2, gate2) = self.get_modulation(shared_down)?;

        // 2. Pre-attention norm + sandwich
        let x_norm = self.sandwich_norm1_in.forward(&self.norm1.forward(x)?)?;
        let x_scaled = (x_norm * (1.0 + scale1))?;

        // 3. Attention with QK-Norm
        let qkv = self.qkv.forward(&x_scaled)?;
        let (q, k, v) = split_qkv(qkv, self.num_heads)?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = apply_rope(q, k, rope)?;
        let attn = scaled_dot_product_attention(&q, &k, &v)?;
        let attn = self.sandwich_norm1_out.forward(&self.proj.forward(&attn)?)?;

        // 4. Residual with gate
        let x = (x + attn * gate1)?;

        // 5. FFN path (similar pattern)
        let x_norm = self.sandwich_norm2_in.forward(&self.norm2.forward(&x)?)?;
        let x_scaled = (x_norm * (1.0 + scale2))?;
        let ffn = self.ffn_down.forward(&self.ffn_up.forward(&x_scaled)?.gelu()?)?;
        let ffn = self.sandwich_norm2_out.forward(&ffn)?;

        // 6. Residual with gate
        (x + ffn * gate2)
    }
}
```

### Phase 2: Modality Processors

```rust
pub struct ModalityProcessor {
    blocks: Vec<S3DiTBlock>,  // 2 blocks
}

impl ModalityProcessor {
    pub fn forward(&self, tokens: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let mut x = tokens.clone();
        for block in &self.blocks {
            x = block.forward(&x, cond, /* ... */)?;
        }
        Ok(x)
    }
}
```

### Phase 3: Main Z-Image Model

```rust
pub struct ZImageModel {
    // Modality processors
    text_processor: ModalityProcessor,
    image_processor: ModalityProcessor,

    // Input projections
    text_in: Linear,
    image_in: Linear,

    // Conditioning
    time_embed: MlpEmbedder,
    cond_down: Linear,  // Shared down-projection

    // Main backbone
    blocks: Vec<S3DiTBlock>,  // 30 blocks

    // Output
    final_norm: RMSNorm,
    final_proj: Linear,

    config: ZImageConfig,
}

pub struct ZImageConfig {
    pub hidden_size: usize,        // 3840
    pub num_heads: usize,          // 32
    pub num_layers: usize,         // 30
    pub ffn_intermediate: usize,   // 10240
    pub text_dim: usize,           // Qwen3 output dim
    pub vae_channels: usize,       // 16 (Flux VAE)
}
```

### Phase 4: Text Encoder (Qwen3-4B)

Need to add Qwen3 support to candle or use existing implementation:

```rust
// Option A: Use candle's Qwen implementation if available
use candle_transformers::models::qwen2;

// Option B: Implement minimal Qwen3 encoder
pub struct Qwen3Encoder {
    // ... transformer layers
}
```

---

## runtime_diffusion Crate Structure

```
crates/runtimes/runtime_diffusion/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Runtime trait implementation
│   ├── models/
│   │   ├── mod.rs
│   │   ├── flux/           # Existing FLUX (copy from candle)
│   │   │   ├── mod.rs
│   │   │   ├── model.rs
│   │   │   ├── autoencoder.rs
│   │   │   └── sampling.rs
│   │   └── zimage/         # NEW: Z-Image S3-DiT
│   │       ├── mod.rs
│   │       ├── model.rs        # S3DiTBlock, ZImageModel
│   │       ├── config.rs       # ZImageConfig
│   │       ├── processors.rs   # ModalityProcessors
│   │       └── sampling.rs     # 8-step turbo sampling
│   ├── encoders/
│   │   ├── mod.rs
│   │   ├── t5.rs           # For FLUX
│   │   ├── clip.rs         # For FLUX
│   │   └── qwen3.rs        # NEW: For Z-Image
│   ├── vae/
│   │   ├── mod.rs
│   │   └── flux_vae.rs     # Shared between FLUX and Z-Image
│   └── pipeline.rs         # High-level generation API
```

### Cargo.toml

```toml
[package]
name = "ohmygpu_runtime_diffusion"
description = "Diffusion model runtime for ohmygpu (FLUX, Z-Image)"
version.workspace = true
edition.workspace = true
license.workspace = true

[features]
default = []
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal"]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
flash-attn = ["candle-transformers/flash-attn"]
# Model features
flux = []
zimage = []

[dependencies]
ohmygpu_core.workspace = true
ohmygpu_runtime_api.workspace = true
tokio.workspace = true
anyhow.workspace = true
tracing.workspace = true

candle-core.workspace = true
candle-nn.workspace = true
candle-transformers.workspace = true
tokenizers.workspace = true
hf-hub.workspace = true

image = "0.25"  # For image I/O
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create `runtime_diffusion` crate structure
- [ ] Copy FLUX implementation from candle as reference
- [ ] Implement shared VAE (already in candle)
- [ ] Add basic pipeline structure

### Phase 2: S3-DiT Core (Week 2)
- [ ] Implement `S3DiTBlock` with Sandwich-Norm
- [ ] Implement shared conditioning projection
- [ ] Implement `ModalityProcessor`
- [ ] Implement `ZImageModel` skeleton

### Phase 3: Text Encoder (Week 3)
- [ ] Add Qwen3-4B encoder (or find existing)
- [ ] Implement tokenization for bilingual support
- [ ] Wire up text encoding pipeline

### Phase 4: Integration (Week 4)
- [ ] Implement 8-step turbo sampling schedule
- [ ] Load Z-Image-Turbo weights from HuggingFace
- [ ] End-to-end text-to-image generation
- [ ] Test on Metal (macOS)

### Phase 5: Optimization
- [ ] Add quantization support (GGUF)
- [ ] Memory optimization for 16GB VRAM
- [ ] Benchmark against Python diffusers

---

## References

1. **Z-Image Paper**: [arxiv.org/abs/2511.22699](https://arxiv.org/abs/2511.22699)
2. **Decoupled-DMD Paper**: [arxiv.org/abs/2511.22677](https://arxiv.org/abs/2511.22677)
3. **Z-Image Repo**: [github.com/Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image)
4. **Candle FLUX**: [github.com/huggingface/candle/tree/main/candle-transformers/src/models/flux](https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/flux)
5. **HuggingFace Model**: [huggingface.co/Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)

---

## Notes

### Why Port to Rust/Candle?

1. **Unified runtime**: Same codebase for LLMs and diffusion
2. **No Python dependency**: Simpler deployment
3. **Metal optimization**: Better macOS performance
4. **Memory efficiency**: Rust's ownership model

### Risks

1. **Qwen3-4B encoder**: May need significant work if not in candle
2. **Weight compatibility**: Need to verify safetensor format matches
3. **Sandwich-Norm**: Novel normalization, need to verify correctness
4. **8-step schedule**: Decoupled-DMD specifics may be tricky

### Fallback Plan

If native Rust port takes too long:
1. Use Python subprocess with `diffusers` for Z-Image
2. Keep FLUX native in Rust (already works)
3. Gradually port Z-Image components
