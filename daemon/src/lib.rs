//! ohmygpu_daemon - HTTP server with OpenAI-compatible API
//!
//! This crate provides the daemon server that:
//! - Exposes OpenAI-compatible API endpoints
//! - Manages model lifecycle (load/unload)
//! - Handles concurrent requests

pub mod api;
pub mod server;
pub mod state;

pub use server::run_server;
