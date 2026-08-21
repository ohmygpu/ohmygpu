//! ohmygpu_daemon — the OhMyGPU Runtime.
//!
//! A headless local daemon that exposes:
//!
//! * `/v1/responses` and `/v1/chat/completions` — an OpenAI-compatible
//!   *subset* for inference, both translated into one internal pipeline
//! * `/v1/models` — installed models
//! * `/ohmygpu/v1/*` — the Management API (health, status, hardware, model
//!   pull/delete/start/stop, backend install, shutdown)
//!
//! Binary: `ohmygpu-runtime` (this crate). The CLI's `omg serve` runs the same
//! [`serve`] function in-process.

pub mod api;
pub mod audio;
pub mod error;
pub mod manager;
pub mod server;
pub mod state;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use server::{serve, DaemonRecord, ServeOptions};
