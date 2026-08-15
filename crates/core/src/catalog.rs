//! The curated model catalog and model-reference parsing.
//!
//! v0.1 deliberately supports a *small, reliable* set of models: single-file GGUF
//! instruct models from ungated Hugging Face repositories, all verified to load
//! in llama.cpp. Power users can still pull any GGUF with an explicit reference
//! (`hf:owner/repo/file.gguf`), but that path is "advanced" and unsupported.

use serde::{Deserialize, Serialize};

use crate::registry::ModelSource;

/// One supported model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogEntry {
    /// Stable OhMyGPU model id (what clients pass as `model`).
    pub id: &'static str,
    pub display_name: &'static str,
    pub family: &'static str,
    /// Hugging Face repository holding the GGUF.
    pub repo: &'static str,
    /// GGUF file inside the repository.
    pub file: &'static str,
    pub quantization: &'static str,
    /// Approximate download size, for display before pulling.
    pub size_bytes_approx: u64,
    /// llama.cpp has a native tool-call parser for this model family.
    pub tools: bool,
}

const MB: u64 = 1_000_000;

/// Verified against the Hugging Face API on 2026-08-15 (all repos ungated,
/// single-file GGUFs).
pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "smollm2-135m-instruct",
        display_name: "SmolLM2 135M Instruct (tiny; smoke tests)",
        family: "smollm2",
        repo: "bartowski/SmolLM2-135M-Instruct-GGUF",
        file: "SmolLM2-135M-Instruct-Q8_0.gguf",
        quantization: "Q8_0",
        size_bytes_approx: 145 * MB,
        tools: false,
    },
    CatalogEntry {
        id: "qwen2.5-0.5b-instruct",
        display_name: "Qwen2.5 0.5B Instruct",
        family: "qwen2.5",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        file: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 491 * MB,
        tools: true,
    },
    CatalogEntry {
        id: "qwen2.5-1.5b-instruct",
        display_name: "Qwen2.5 1.5B Instruct",
        family: "qwen2.5",
        repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        file: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 1_120 * MB,
        tools: true,
    },
    CatalogEntry {
        id: "qwen2.5-3b-instruct",
        display_name: "Qwen2.5 3B Instruct",
        family: "qwen2.5",
        repo: "Qwen/Qwen2.5-3B-Instruct-GGUF",
        file: "qwen2.5-3b-instruct-q4_k_m.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 2_100 * MB,
        tools: true,
    },
    CatalogEntry {
        id: "qwen2.5-7b-instruct",
        display_name: "Qwen2.5 7B Instruct",
        family: "qwen2.5",
        repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
        file: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 4_680 * MB,
        tools: true,
    },
    CatalogEntry {
        id: "qwen3-4b-instruct",
        display_name: "Qwen3 4B Instruct (2507, non-thinking)",
        family: "qwen3",
        repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        file: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 2_500 * MB,
        tools: true,
    },
    CatalogEntry {
        id: "llama-3.2-1b-instruct",
        display_name: "Llama 3.2 1B Instruct",
        family: "llama3",
        repo: "bartowski/Llama-3.2-1B-Instruct-GGUF",
        file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 810 * MB,
        tools: true,
    },
    CatalogEntry {
        id: "llama-3.2-3b-instruct",
        display_name: "Llama 3.2 3B Instruct",
        family: "llama3",
        repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 2_020 * MB,
        tools: true,
    },
    CatalogEntry {
        id: "llama-3.1-8b-instruct",
        display_name: "Llama 3.1 8B Instruct",
        family: "llama3",
        repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        file: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 4_920 * MB,
        tools: true,
    },
    CatalogEntry {
        id: "phi-4-mini-instruct",
        display_name: "Phi-4 Mini Instruct",
        family: "phi4",
        repo: "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        file: "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 2_490 * MB,
        tools: false,
    },
    CatalogEntry {
        id: "gemma-3-1b-it",
        display_name: "Gemma 3 1B IT",
        family: "gemma3",
        repo: "ggml-org/gemma-3-1b-it-GGUF",
        file: "gemma-3-1b-it-Q4_K_M.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 810 * MB,
        tools: false,
    },
    CatalogEntry {
        id: "gemma-3-4b-it",
        display_name: "Gemma 3 4B IT",
        family: "gemma3",
        repo: "ggml-org/gemma-3-4b-it-GGUF",
        file: "gemma-3-4b-it-Q4_K_M.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 2_490 * MB,
        tools: false,
    },
    CatalogEntry {
        id: "gemma-3-12b-it",
        display_name: "Gemma 3 12B IT",
        family: "gemma3",
        repo: "ggml-org/gemma-3-12b-it-GGUF",
        file: "gemma-3-12b-it-Q4_K_M.gguf",
        quantization: "Q4_K_M",
        size_bytes_approx: 7_300 * MB,
        tools: false,
    },
];

pub fn find(id: &str) -> Option<&'static CatalogEntry> {
    let id = id.trim();
    CATALOG.iter().find(|e| e.id.eq_ignore_ascii_case(id))
}

/// A user-supplied model reference, resolved to something we can download.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    /// The OhMyGPU model id this will be installed under.
    pub id: String,
    /// Where the file comes from (Hugging Face repo+file, or a direct URL).
    pub source: ModelSource,
    /// The URL to download.
    pub url: String,
    /// File name to store locally.
    pub file: String,
    /// True when it came from the curated catalog.
    pub curated: bool,
    pub tools: bool,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes_approx: Option<u64>,
}

fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

impl ModelRef {
    /// Accepts:
    /// * a catalog id: `qwen2.5-1.5b-instruct`
    /// * `hf:owner/repo/path/to/file.gguf` (also `hf://…`)
    /// * `https://huggingface.co/owner/repo/resolve/<rev>/file.gguf` (or `/blob/`)
    ///
    /// `id_override` names the installed model for non-catalog refs.
    pub fn parse(reference: &str, id_override: Option<&str>) -> Result<ModelRef, String> {
        let r = reference.trim();
        if r.is_empty() {
            return Err("model reference is empty".into());
        }
        if let Some(entry) = find(r) {
            return Ok(ModelRef {
                id: entry.id.to_string(),
                source: ModelSource::HuggingFace {
                    repo: entry.repo.to_string(),
                    file: entry.file.to_string(),
                },
                url: hf_url(entry.repo, entry.file),
                file: entry.file.to_string(),
                curated: true,
                tools: entry.tools,
                display_name: entry.display_name.to_string(),
                size_bytes_approx: Some(entry.size_bytes_approx),
            });
        }

        let (source, url, file) = if let Some(rest) =
            r.strip_prefix("hf://").or_else(|| r.strip_prefix("hf:"))
        {
            let (repo, file) = split_repo_file(rest)?;
            (
                ModelSource::HuggingFace {
                    repo: repo.clone(),
                    file: file.clone(),
                },
                hf_url(&repo, &file),
                file,
            )
        } else if let Some(rest) = r
            .strip_prefix("https://huggingface.co/")
            .or_else(|| r.strip_prefix("http://huggingface.co/"))
            .or_else(|| r.strip_prefix("huggingface.co/"))
        {
            // owner/repo/(resolve|blob)/<rev>/file
            let parts: Vec<&str> = rest.split('/').collect();
            if parts.len() >= 5 && (parts[2] == "resolve" || parts[2] == "blob") {
                let repo = format!("{}/{}", parts[0], parts[1]);
                let file = parts[4..].join("/");
                (
                    ModelSource::HuggingFace {
                        repo: repo.clone(),
                        file: file.clone(),
                    },
                    hf_url(&repo, &file),
                    file,
                )
            } else {
                return Err(format!(
                    "unrecognized Hugging Face URL '{r}' (expected https://huggingface.co/owner/repo/resolve/main/file.gguf)"
                ));
            }
        } else if r.starts_with("http://") || r.starts_with("https://") {
            let path = r.split('?').next().unwrap_or(r);
            let file = path.rsplit('/').next().unwrap_or("").to_string();
            (ModelSource::Url { url: r.to_string() }, r.to_string(), file)
        } else {
            return Err(format!(
                "unknown model '{r}'. Use a catalog id (see GET /ohmygpu/v1/catalog or `omg model catalog`), \
                 an explicit GGUF reference like hf:owner/repo/file.gguf, or a direct https URL to a .gguf file"
            ));
        };

        if !file.to_ascii_lowercase().ends_with(".gguf") {
            return Err(format!(
                "'{file}' is not a .gguf file; v0.1 supports GGUF models only"
            ));
        }
        let id = match id_override.map(str::trim).filter(|s| !s.is_empty()) {
            Some(id) => validate_id(id)?,
            None => derive_id(&file),
        };
        let display_name = file
            .rsplit('/')
            .next()
            .unwrap_or(&file)
            .trim_end_matches(".gguf")
            .to_string();
        Ok(ModelRef {
            id,
            source,
            url,
            file,
            curated: false,
            tools: false,
            display_name,
            size_bytes_approx: None,
        })
    }

    /// Hugging Face repo, if this reference points at one.
    pub fn repo(&self) -> Option<&str> {
        match &self.source {
            ModelSource::HuggingFace { repo, .. } => Some(repo),
            _ => None,
        }
    }
}

fn split_repo_file(rest: &str) -> Result<(String, String), String> {
    let parts: Vec<&str> = rest.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 3 {
        return Err(format!("expected hf:owner/repo/file.gguf, got 'hf:{rest}'"));
    }
    Ok((format!("{}/{}", parts[0], parts[1]), parts[2..].join("/")))
}

/// Model ids are lowercase `[a-z0-9._-]`, must start alphanumeric.
pub fn validate_id(id: &str) -> Result<String, String> {
    let ok = !id.is_empty()
        && id
            .chars()
            .next()
            .map(|c| c.is_ascii_alphanumeric())
            .unwrap_or(false)
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
        && !id.contains("..");
    if ok {
        Ok(id.to_string())
    } else {
        Err(format!(
            "invalid model id '{id}': use lowercase letters, digits, '.', '_' or '-'"
        ))
    }
}

/// `SmolLM2-360M-Instruct-Q4_K_M.gguf` → `smollm2-360m-instruct-q4_k_m`
pub fn derive_id(file: &str) -> String {
    let stem = file
        .rsplit('/')
        .next()
        .unwrap_or(file)
        .trim_end_matches(".gguf")
        .trim_end_matches(".GGUF");
    let mut out = String::with_capacity(stem.len());
    for c in stem.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
            out.push(c);
        } else {
            out.push('-');
        }
    }
    let out = out
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_string();
    if out.is_empty() {
        "model".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_valid() {
        let mut seen = std::collections::HashSet::new();
        for e in CATALOG {
            assert!(validate_id(e.id).is_ok(), "{}", e.id);
            assert!(seen.insert(e.id), "duplicate id {}", e.id);
            assert!(e.file.ends_with(".gguf"));
            assert!(e.repo.contains('/'));
        }
    }

    #[test]
    fn parse_catalog_id_case_insensitive() {
        let r = ModelRef::parse("Qwen2.5-0.5B-Instruct", None).unwrap();
        assert_eq!(r.id, "qwen2.5-0.5b-instruct");
        assert!(r.curated);
        assert!(r.tools);
    }

    #[test]
    fn parse_hf_reference_and_url() {
        let r = ModelRef::parse(
            "hf:bartowski/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct-Q4_K_M.gguf",
            None,
        )
        .unwrap();
        assert_eq!(r.repo(), Some("bartowski/SmolLM2-360M-Instruct-GGUF"));
        assert_eq!(r.file, "SmolLM2-360M-Instruct-Q4_K_M.gguf");
        assert_eq!(r.url, "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf");
        assert_eq!(r.id, "smollm2-360m-instruct-q4_k_m");
        assert!(!r.curated);

        let r = ModelRef::parse(
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf",
            Some("qwen-small"),
        )
        .unwrap();
        assert_eq!(r.repo(), Some("Qwen/Qwen2.5-0.5B-Instruct-GGUF"));
        assert_eq!(r.file, "qwen2.5-0.5b-instruct-q8_0.gguf");
        assert_eq!(r.id, "qwen-small");

        let r = ModelRef::parse(
            "https://models.example.com/dir/My-Model.Q4.gguf?token=1",
            None,
        )
        .unwrap();
        assert_eq!(
            r.source,
            ModelSource::Url {
                url: "https://models.example.com/dir/My-Model.Q4.gguf?token=1".into()
            }
        );
        assert_eq!(r.file, "My-Model.Q4.gguf");
        assert_eq!(r.id, "my-model.q4");

        assert!(ModelRef::parse("hf:owner/repo/model.safetensors", None).is_err());
        assert!(ModelRef::parse("totally-unknown-model", None).is_err());
        assert!(ModelRef::parse("hf:owner/repo/x.gguf", Some("Bad Id!")).is_err());
    }
}
