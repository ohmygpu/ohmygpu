# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build all workspace crates
cargo build

# Run the CLI
cargo run --bin ohmygpu -- <command>

# Run specific commands
cargo run --bin ohmygpu -- search phi-2
cargo run --bin ohmygpu -- model pull microsoft/phi-2
cargo run --bin ohmygpu -- model list
cargo run --bin ohmygpu -- serve --port 11434
cargo run --bin ohmygpu -- chat microsoft/phi-2

# Run tests (when implemented)
cargo test
cargo test -p ohmygpu_core
```

## Architecture Overview

ohmygpu is a Rust workspace that provides unified local AI infrastructure, replacing fragmented tools (ollama, ComfyUI, whisper.cpp) with one daemon, one model repository, and one OpenAI-compatible API.

### Workspace Structure

```
crates/
├── core/                    # ohmygpu_core - Model management, HF downloads, registry
└── runtimes/
    ├── runtime_api/         # Runtime trait contract for pluggable backends
    └── runtime_candle/      # Candle-based inference (Rust-native, first runtime)
daemon/                      # HTTP server with OpenAI-compatible API (axum)
cli/                         # CLI binary with clap + commands
```

### Key Design Principles

1. **Core is GPU-agnostic**: `ohmygpu_core` handles model downloads and registry only. No libtorch, no CUDA bindings. Inference is delegated to runtime plugins.

2. **Pluggable runtimes**: Each runtime (`runtime_candle`, future `runtime_llamacpp`) implements the `Runtime` trait from `runtime_api`. Runtimes receive model paths from core and handle inference.

3. **Daemon-centric**: All UIs (CLI, future desktop, external tools like Open WebUI) connect to the daemon via HTTP. The CLI calls daemon exclusively - no direct core access.

4. **OpenAI-compatible API**: Daemon exposes `/v1/chat/completions` for ecosystem compatibility.

### Data Flow

```
User → CLI → Daemon → Runtime (candle) → GPU
              ↓
           Core (model downloads, registry)
```

### Key Types

- `ModelInfo`, `ModelSource`, `ModelType` - Model metadata (in `core/models.rs`)
- `ModelRegistry` - Local model storage and manifest (in `core/registry.rs`)
- `Runtime` trait - Lifecycle: `load()`, `unload()`, `chat()`, `chat_stream()` (in `runtime_api`)
- `RuntimeCaps`, `RuntimeStatus` - Runtime capabilities and state

### CLI Commands

Binary is `ohmygpu`, intended to be symlinked as `omg` for convenience:
- `model list/pull/rm/info/gc` - Model management (static assets)
- `serve` / `serve status` / `serve stop` - Daemon control
- `gen image/video` - Content generation
- `chat <model>` - Interactive terminal chat
- `search` / `config` / `mcp` / `update` - Utilities

### API Endpoints (Daemon)

- `GET /health` - Health check
- `GET /v1/models` - List installed models
- `POST /v1/chat/completions` - OpenAI-compatible chat

## Rules

1. **MUST NOT use `ln` during testing** - Do not create symlinks when testing. Use the full binary path instead.
