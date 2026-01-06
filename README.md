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
omg model pull microsoft/phi-2

# Start the daemon
omg serve

# Interactive chat
omg chat microsoft/phi-2

# Or chat via OpenAI-compatible API
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

### Model Management

Models are static assets. Use `omg model` to manage them.

| Command | Description |
|---------|-------------|
| `omg model list` | List installed models |
| `omg model pull <model>` | Download model from HuggingFace |
| `omg model rm <model>` | Remove an installed model |
| `omg model info <model>` | Show model details (size, path, type) |
| `omg model gc` | Garbage collect unused cache files |

### Daemon Server

| Command | Description |
|---------|-------------|
| `omg serve` | Start daemon in foreground |
| `omg serve -d` | Start daemon in background (daemon mode) |
| `omg serve status` | Check if daemon is running |
| `omg serve stop` | Stop the daemon |

### Content Generation

| Command | Description |
|---------|-------------|
| `omg gen image "<prompt>"` | Generate image from text |
| `omg gen video "<prompt>"` | Generate video (coming soon) |

### Other Commands

| Command | Description |
|---------|-------------|
| `omg chat <model>` | Interactive terminal chat |
| `omg search <query>` | Search HuggingFace models |
| `omg config [key] [value]` | View or set configuration |
| `omg mcp` | Start MCP server (Claude Desktop) |
| `omg update` | Self-update to latest version |

## API Endpoints

ohmygpu runs on port `11434` and supports **two API formats**:

### OpenAI-compatible API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/models` | GET | List installed models |
| `/health` | GET | Health check |

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "phi-2", "messages": [{"role": "user", "content": "Hello"}]}'
```

Works with Open WebUI, LangChain, and any OpenAI-compatible client.

### Ollama-compatible API (drop-in replacement)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with a model |
| `/api/generate` | POST | Generate completion |
| `/api/tags` | GET | List local models |
| `/api/show` | POST | Show model info |
| `/api/version` | GET | Version info |

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "phi-2", "messages": [{"role": "user", "content": "Hello"}]}'
```

Existing ollama scripts work without modification - just change the port if needed.

### MCP Server (Claude Desktop integration)

ohmygpu includes an MCP server for direct integration with Claude Desktop and other MCP clients.

**Setup:** Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ohmygpu": {
      "command": "ohmygpu",
      "args": ["mcp"]
    }
  }
}
```

**Available tools:**
| Tool | Description |
|------|-------------|
| `chat` | Chat with a local AI model |
| `list_models` | List installed models |
| `status` | Check daemon health |

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
├── daemon/                   # HTTP server, OpenAI/Ollama API
└── cli/                      # CLI binary (includes MCP server)
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
