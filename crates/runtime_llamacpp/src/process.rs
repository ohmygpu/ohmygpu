//! `llama-server` arguments for a [`StartSpec`]; supervision itself lives in
//! `ohmygpu_runtime_common::process`.

use std::path::Path;

use ohmygpu_runtime_api::{RuntimeError, StartSpec};
pub use ohmygpu_runtime_common::process::{free_port, ServerProcess};

/// Command-line arguments for `llama-server` serving `spec` on `127.0.0.1:port`.
pub fn build_args(spec: &StartSpec, port: u16) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "--model".into(),
        spec.model_path.display().to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "--alias".into(),
        spec.model_id.clone(),
        "--jinja".into(),
        "--no-webui".into(),
    ];
    if let Some(mmproj) = &spec.mmproj_path {
        args.push("--mmproj".into());
        args.push(mmproj.display().to_string());
    }
    if let Some(ctx) = spec.context_length {
        args.push("--ctx-size".into());
        args.push(ctx.to_string());
    }
    if let Some(ngl) = spec.gpu_layers {
        args.push("--n-gpu-layers".into());
        args.push(ngl.to_string());
    }
    if let Some(t) = spec.threads {
        args.push("--threads".into());
        args.push(t.to_string());
    }
    args
}

/// Spawn a supervised `llama-server` for `spec`.
pub fn spawn(binary: &Path, spec: &StartSpec, port: u16) -> Result<ServerProcess, RuntimeError> {
    ServerProcess::spawn(
        "llama-server",
        &spec.model_id,
        binary,
        &build_args(spec, port),
        port,
        |model, line| tracing::debug!(target: "llamacpp", model = %model, "{line}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn args_include_projector_and_options() {
        let spec = StartSpec {
            model_id: "m".into(),
            kind: Default::default(),
            model_path: PathBuf::from("/m/w.gguf"),
            mmproj_path: Some(PathBuf::from("/m/proj.gguf")),
            context_length: Some(4096),
            gpu_layers: None,
            threads: Some(4),
        };
        let args = build_args(&spec, 1234);
        let joined = args.join(" ");
        assert!(joined.contains("--model /m/w.gguf"));
        assert!(joined.contains("--mmproj /m/proj.gguf"));
        assert!(joined.contains("--port 1234"));
        assert!(joined.contains("--ctx-size 4096"));
        assert!(joined.contains("--threads 4"));
        assert!(!joined.contains("--n-gpu-layers"));
    }
}
