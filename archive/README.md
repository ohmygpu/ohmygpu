# Archived code (not built)

These crates were part of the pre-v0.1 "unified local AI" experiment and are
kept for reference only. They are **excluded from the Cargo workspace** and
target the *old* `ohmygpu_runtime_api` trait, so they do not compile against
the current tree without porting.

| Directory | What it was | Why it is archived |
|-----------|-------------|--------------------|
| `runtime_candle/` | In-process LLM inference via Candle (safetensors, llama/phi only) | v0.1 orchestrates llama.cpp instead of maintaining an inference engine — see `docs/architecture.md`, Part 2 |
| `runtime_diffusion/` | Z-Image text-to-image pipeline via Candle | image generation is outside the v0.1 product |
| `docs/z-image.md` | Z-Image usage notes | same |

Last known-good state of this code: commit `23b3356` on the `runtime` branch
(built with `cargo build --features metal`).
