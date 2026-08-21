//! ohmygpu_runtime_common — what every subprocess-based runtime adapter needs
//! and none should reimplement:
//!
//! * [`process`] — a supervised child server: spawn with arguments, forward
//!   logs, detect exit, stop gracefully (SIGTERM → wait → SIGKILL), free ports.
//! * [`install`] — managed binary installs: locate (config / managed dir /
//!   `PATH`), install records, download + extract release archives, resolve a
//!   GitHub "latest" tag.
//!
//! Backend-specific knowledge (which asset for which machine, which arguments,
//! how to talk to the server) stays in each `runtime_*` crate.

pub mod install;
pub mod process;

pub use install::{BinarySource, InstallRecord, LocatedBinary};
pub use process::{free_port, ServerProcess};
