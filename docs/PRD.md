# OhMyGPU v1 - Product Requirements Document

## Overview

**Version:** 1.0
**Goal:** Ship a working local AI inference system that can replace ollama for LLM chat.

**Success criteria:** User can `omg pull`, `omg run`, and chat via OpenAI-compatible API.

**CLI naming:** Single binary `ohmygpu`, symlink `omg` (preferred, shorter, memorable).

**One-liner install:**
```bash
curl -sSL https://get.ohmygpu.com | sh
```

---

## v1 Scope

### In Scope

| Component | What ships |
|-----------|------------|
| `ohmygpu_core` | HuggingFace downloads, model repository, RunSpec |
| `runtime_api` | Runtime trait definition |
| `runtime_candle` | Rust-native LLM inference |
| `ohmygpu_daemon` | HTTP API server (OpenAI + Ollama compatible), model lifecycle |
| `ohmygpu_cli` | CLI with TUI, pull/run/stop/mcp commands |

### Out of Scope (v2+)

- Desktop app (Tauri)
- `runtime_llamacpp`, `runtime_comfyui`, `runtime_vllm`
- Training/fine-tuning
- Multi-GPU scheduling
- Remote GPU boxes / fleet management
- Whisper audio transcription (candle supports it, but not v1 priority)

---

## User Stories

### US-1: Download a model
```
As a user, I want to download a model from HuggingFace
So that I can run it locally

omg model pull microsoft/phi-2
```

**Acceptance criteria:**
- Downloads from HuggingFace with progress display
- Supports resume on interruption
- Stores in unified model repository
- Shows in `omg model list`

### US-2: Chat with a model
```
As a user, I want to chat with a local model
So that I can interact with it

omg serve              # Start daemon
omg chat microsoft/phi-2  # Interactive chat
```

**Acceptance criteria:**
- Daemon loads model on demand via candle
- Exposes OpenAI-compatible endpoint
- Interactive terminal chat with streaming
- Model is unloaded when idle

### US-3: Chat via API
```
As a developer, I want to send chat requests to a running model
So that I can integrate with my application

# OpenAI-compatible API
curl http://localhost:10692/v1/chat/completions \
  -d '{"model": "microsoft/phi-2", "messages": [{"role": "user", "content": "Hello"}]}'

# Ollama-compatible API (drop-in replacement)
curl http://localhost:10692/api/chat \
  -d '{"model": "microsoft/phi-2", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Acceptance criteria:**
- OpenAI-compatible request/response format
- Ollama-compatible API for drop-in replacement
- Streaming support (SSE)
- Works with Open WebUI, LangChain, etc.

### US-4: List and manage models
```
As a user, I want to see what models I have
So that I can manage disk space

omg model list
omg model info microsoft/phi-2
omg model rm microsoft/phi-2
omg model gc
```

**Acceptance criteria:**
- Lists installed models with size
- Shows model details (source, type, path, size)
- Can remove models to free disk space
- Garbage collect cleans orphaned cache files

### US-5: View daemon status
```
As a user, I want to check if the daemon is running
So that I can troubleshoot connection issues

omg serve status
```

**Acceptance criteria:**
- Shows daemon running state
- Shows loaded models
- Shows daemon URL

### US-6: Use with Claude Desktop (MCP)
```
As a Claude Desktop user, I want to chat with local models
So that I can use them directly from Claude

# Add to Claude Desktop config
{
  "mcpServers": {
    "ohmygpu": {
      "command": "omg",
      "args": ["mcp"]
    }
  }
}
```

**Acceptance criteria:**
- MCP server starts via `omg mcp`
- Exposes tools: `chat`, `list_models`, `status`
- Works with Claude Desktop and other MCP clients

---

## Technical Requirements

### TR-1: ohmygpu_core

| Requirement | Details |
|-------------|---------|
| HuggingFace API | Fetch model metadata, file lists, resolve revisions |
| Download manager | Concurrent downloads, resume, SHA256 verification |
| Model repository | Local storage with manifest, deduplication by hash |
| Model formats | GGUF, SafeTensors (what candle supports) |
| Event stream | Progress events for UI consumption |

**Data structures:**
```rust
ModelRef { provider, repo, kind }
Revision { branch, resolved_commit }
Artifact { filename, size, sha256, format }
InstallRecord { model_ref, revision, artifacts, local_path, installed_at }
```

### TR-2: runtime_api

| Requirement | Details |
|-------------|---------|
| Runtime trait | `load()`, `unload()`, `health()`, `chat()` |
| RuntimeCaps | Declare capabilities (chat, embeddings, etc.) |
| RuntimeStatus | `loading`, `ready`, `error`, `unloaded` |

### TR-3: runtime_candle

| Requirement | Details |
|-------------|---------|
| Model loading | Load GGUF/SafeTensors into GPU via candle |
| Inference | Text generation with sampling params |
| Streaming | Token-by-token output |
| Models supported | Llama, Mistral, Phi (candle's supported architectures) |
| Acceleration | Metal (macOS), CUDA (Linux/Windows) |

### TR-4: ohmygpu_daemon

| Requirement | Details |
|-------------|---------|
| HTTP server | axum or actix-web |
| OpenAI API | `POST /v1/chat/completions` with streaming |
| Management API | `GET /v1/models`, `POST /v1/models/pull`, `DELETE /v1/models/{id}` |
| Model lifecycle | Load on first request, unload on timeout (configurable) |
| Config | TOML file for port, model directory, defaults |
| Socket | Unix socket (macOS/Linux) + TCP fallback |

**API endpoints (v1):**

OpenAI-compatible API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/models` | GET | List installed models |
| `/v1/models/pull` | POST | Start model download |
| `/v1/models/{id}` | DELETE | Remove model |
| `/health` | GET | Daemon health check |

Ollama-compatible API (drop-in replacement):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with a model |
| `/api/generate` | POST | Generate completion |
| `/api/tags` | GET | List local models |
| `/api/show` | POST | Show model info |
| `/api/version` | GET | Version info |

### TR-5: ohmygpu_cli

| Requirement | Details |
|-------------|---------|
| Framework | clap for args, ratatui for TUI |
| Binary | `ohmygpu` (single binary), `omg` (symlink, preferred) |
| Commands | Nested subcommands: `model`, `serve`, `gen`, `chat`, `mcp` |
| Daemon connection | Connect via HTTP to localhost:10692 |
| Progress display | Download progress bars, streaming tokens |

**CLI commands (v1):**

Model management (static assets):

| Command | Description |
|---------|-------------|
| `omg model list` | List installed models |
| `omg model pull <model>` | Download model from HuggingFace |
| `omg model rm <model>` | Remove model |
| `omg model info <model>` | Show model details |
| `omg model gc` | Garbage collect unused cache files |

Daemon server:

| Command | Description |
|---------|-------------|
| `omg serve` | Start daemon in foreground |
| `omg serve -d` | Start daemon in background |
| `omg serve status` | Check daemon status |
| `omg serve stop` | Stop the daemon |

Content generation:

| Command | Description |
|---------|-------------|
| `omg gen image <prompt>` | Generate image |
| `omg gen video <prompt>` | Generate video (future) |

Other:

| Command | Description |
|---------|-------------|
| `omg chat <model>` | Interactive terminal chat |
| `omg search <query>` | Search HuggingFace models |
| `omg config` | View/set configuration |
| `omg mcp` | Start MCP server for Claude Desktop |
| `omg update` | Self-update binary and symlink |

### TR-6: Install Script

| Requirement | Details |
|-------------|---------|
| URL | `https://get.ohmygpu.com` (redirect to GitHub raw) |
| Platforms | macOS (arm64, x86_64), Linux (x86_64, arm64) |
| Actions | Detect platform, download binary, install to `/usr/local/bin`, create `omg` symlink |
| Fallback | Provide `cargo install ohmygpu` as alternative |

**Install script flow:**
1. Detect OS and architecture
2. Download correct binary from GitHub releases
3. Install to `/usr/local/bin/ohmygpu` (or `~/.local/bin` if no sudo)
4. Create symlink `omg` → `ohmygpu`
5. Verify installation with `omg --version`

---

## Non-Functional Requirements

### NFR-1: Performance
- Model loading: < 30s for 7B model on consumer GPU
- First token latency: < 500ms after model loaded
- Throughput: Competitive with llama.cpp (within 80%)

### NFR-2: Compatibility
- macOS: Apple Silicon (Metal)
- Linux: NVIDIA GPU (CUDA)
- Windows: NVIDIA GPU (CUDA) - best effort

### NFR-3: Disk usage
- Model deduplication by SHA256
- GC command to clean unused files

### NFR-4: Error handling
- Clear error messages for common issues (no GPU, OOM, model not found)
- Graceful degradation (CPU fallback if no GPU)

---

## Implementation Milestones

### M1: Core + Download
- [ ] Workspace setup (Cargo.toml)
- [ ] HuggingFace API client
- [ ] Download manager with resume
- [ ] Local model repository
- [ ] CLI: `pull`, `models`, `rm`

### M2: Runtime + Inference
- [ ] runtime_api trait definition
- [ ] runtime_candle implementation
- [ ] Load model, run inference
- [ ] CLI: `run`, `stop`

### M3: Daemon + API
- [ ] HTTP server with axum
- [ ] OpenAI-compatible `/v1/chat/completions`
- [ ] Streaming support
- [ ] Model lifecycle (load/unload)
- [ ] CLI connects to daemon

### M4: Polish + Distribution
- [ ] TUI improvements (ratatui)
- [ ] Error messages and edge cases
- [ ] Install script (`get.ohmygpu.com`)
- [ ] `omg update` self-update command
- [ ] GitHub releases CI (build for macOS/Linux)
- [ ] Documentation
- [ ] Test with Open WebUI

---

## Design Decisions

| Question | Decision |
|----------|----------|
| Default port | `10692` |
| Config directory | `~/.config/ohmygpu/` |
| Models directory | `~/.config/ohmygpu/models/` |
| PID file | `~/.config/ohmygpu/daemon.pid` |
| Auto-start daemon | Explicit `omg serve` required |
| HF token | Environment variable `HF_TOKEN` |

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `candle-core`, `candle-nn`, `candle-transformers` | Inference engine |
| `hf-hub` | HuggingFace API (or custom implementation) |
| `tokio` | Async runtime |
| `axum` | HTTP server |
| `clap` | CLI argument parsing |
| `ratatui` | TUI framework |
| `indicatif` | Progress bars |
| `serde`, `serde_json` | Serialization |
| `tracing` | Logging |
| `self_update` | Self-update from GitHub releases |
| `rmcp` | MCP server for Claude Desktop integration |

---

## References

- [Architecture doc](./architecture.md)
- [candle repo](https://github.com/huggingface/candle)
- [OpenAI API spec](https://platform.openai.com/docs/api-reference/chat)
- [ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [MCP specification](https://modelcontextprotocol.io/)
- [rmcp Rust SDK](https://github.com/modelcontextprotocol/rust-sdk)
