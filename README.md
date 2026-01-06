# ohmygpu

**Stop juggling 5 different AI tools. One daemon, all workloads.**

ohmygpu is unified local AI infrastructure - replacing the fragmented landscape of single-purpose tools (ollama, ComfyUI, whisper.cpp) with one daemon, one model repository, and one OpenAI-compatible API.

## Install

```bash
curl -sSL https://get.ohmygpu.com | sh
```

Or build from source:
```bash
# macOS (Apple Silicon)
make build-metal

# Linux (NVIDIA)
make build-cuda
```

## Quick Start

```bash
# Download a model from HuggingFace
omg pull microsoft/phi-2

# Run the model
omg run microsoft/phi-2

# Chat via OpenAI-compatible API
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "microsoft/phi-2", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Requirements

GPU acceleration is required: **CUDA** (NVIDIA) or **Metal** (macOS).

CPU-only mode is not supported.

**Recommended GPU memory:** 8GB+

Devices with less than 8GB may run with warnings and can be unstable (OOM / slowdowns).

## CLI Commands

| Command | Description |
|---------|-------------|
| `omg pull <model>` | Download model from HuggingFace |
| `omg run <model>` | Load and serve model |
| `omg stop [model]` | Stop running model(s) |
| `omg models` | List installed models |
| `omg rm <model>` | Remove model |
| `omg status` | Show daemon and GPU status |
| `omg serve` | Start daemon server |
| `omg search <query>` | Search HuggingFace models |
| `omg generate <prompt>` | Generate images (diffusion models) |
| `omg update` | Self-update to latest version |

## API Endpoints

ohmygpu exposes an OpenAI-compatible API on port `11434`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/models` | GET | List installed models |
| `/health` | GET | Health check |

Works with Open WebUI, LangChain, and any OpenAI-compatible client.

## Why ohmygpu?

### The Problem: Fragmentation

| Category | Current Tools | Problems |
|----------|---------------|----------|
| LLM inference | ollama, LM Studio, LocalAI | Each has own model cache, own API |
| Image generation | ComfyUI, Automatic1111 | Separate downloads, separate processes |
| Audio transcription | whisper.cpp, faster-whisper | Yet another binary to manage |
| Model downloads | huggingface-cli, manual | No unified cache, no deduplication |

### The Solution

```
Before:
├── ollama          (LLMs, own model cache)
├── ComfyUI         (images, own model cache)
├── whisper.cpp     (audio, own model cache)
└── 3 different UIs, 3 different APIs, duplicate 70GB models

After:
└── ohmygpu
    ├── daemon      (one API, one model cache, one GPU scheduler)
    ├── runtimes    (candle, llamacpp, comfyui, whisper)
    └── UI          (TUI/Desktop, or Open WebUI via API)
```

### vs ollama

| Aspect | ollama | ohmygpu |
|--------|--------|---------|
| Architecture | Monolithic | Modular (core + pluggable runtimes) |
| Model formats | Own format, converts on import | HuggingFace ecosystem directly |
| Workload types | LLMs only | LLMs, images, audio, video |
| GPU scheduling | None (single model) | VRAM budget, concurrent models |

## Architecture

```
ohmygpu/
├── crates/
│   ├── core/                 # Model downloads, registry, config
│   └── runtimes/
│       ├── runtime_api/      # Runtime trait contract
│       ├── runtime_candle/   # Rust-native inference (LLMs)
│       └── runtime_diffusion/# Image generation (Flux, SD)
├── daemon/                   # HTTP server, OpenAI API
└── cli/                      # CLI binary
```

**Data flow:**
```
User → CLI → Daemon → Runtime (candle) → GPU
              ↓
           Core (model downloads, registry)
```

## Configuration

Config file: `~/.config/ohmygpu/config.toml`

```toml
[daemon]
port = 11434

[inference]
max_tokens = 2048
temperature = 0.7
use_gpu = true
```

## Supported Models

Any model from HuggingFace that candle supports:
- **LLMs:** Llama, Mistral, Phi, Qwen
- **Image:** Flux, Stable Diffusion, Z-Image

## License

Apache-2.0
