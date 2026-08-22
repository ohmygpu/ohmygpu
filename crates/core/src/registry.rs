//! Installed-model registry: a small JSON index (`registry.json`) of the GGUF
//! files OhMyGPU manages. Lifecycle state (running/stopped/…) is *not* stored
//! here — it lives in the daemon and is rebuilt on start; the registry only
//! answers "what is installed, where, and from what source".

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use ohmygpu_inference::ModelKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSource {
    HuggingFace {
        repo: String,
        file: String,
    },
    /// Any direct http(s) URL to a GGUF file (self-hosted models).
    Url {
        url: String,
    },
    /// Imported from a local path (reserved; not exposed in v0.1).
    Local,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ModelCapabilities {
    /// Native tool calling is expected to work with this model.
    pub tools: bool,
    /// The model accepts image input (a multimodal projector is installed).
    #[serde(default)]
    pub vision: bool,
}

/// A kind of content a model can take in or give out.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
    Audio,
}

/// What goes into a model and what comes out — derived from its kind and
/// capabilities, so clients can pick a model for a task without knowing
/// OhMyGPU's model kinds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Modalities {
    pub input: Vec<Modality>,
    pub output: Vec<Modality>,
}

impl ModelCapabilities {
    /// The modalities of a `kind` model with these capabilities.
    pub fn modalities(&self, kind: ModelKind) -> Modalities {
        match kind {
            ModelKind::Llm => Modalities {
                input: if self.vision {
                    vec![Modality::Text, Modality::Image]
                } else {
                    vec![Modality::Text]
                },
                output: vec![Modality::Text],
            },
            ModelKind::Whisper => Modalities {
                input: vec![Modality::Audio],
                output: vec![Modality::Text],
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstalledModel {
    pub id: String,
    pub display_name: String,
    pub source: ModelSource,
    /// `llm` (default) or `whisper` — decides the backend and the API.
    #[serde(default)]
    pub kind: ModelKind,
    /// `gguf` for LLMs, `ggml` for whisper models.
    pub format: String,
    /// Absolute path to the model file.
    pub path: PathBuf,
    /// Multimodal projector file (vision models), stored next to `path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_path: Option<PathBuf>,
    /// Total bytes on disk (model file plus projector).
    pub size_bytes: u64,
    pub installed_at: DateTime<Utc>,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    /// Native context window (tokens) from the model file's metadata
    /// (`<arch>.context_length`); `None` when unknown (whisper models, or
    /// files that do not record it). Filled at install and backfilled on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// From the curated catalog (vs. an explicit hf: reference).
    #[serde(default)]
    pub curated: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RegistryFile {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    models: BTreeMap<String, InstalledModel>,
}

const REGISTRY_VERSION: u32 = 1;

#[derive(Debug)]
pub struct ModelRegistry {
    path: PathBuf,
    models: BTreeMap<String, InstalledModel>,
}

impl ModelRegistry {
    /// Load (or create an empty registry at) `path`.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let models = if path.exists() {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            if content.trim().is_empty() {
                BTreeMap::new()
            } else {
                let file: RegistryFile = serde_json::from_str(&content)
                    .with_context(|| format!("parsing {}", path.display()))?;
                file.models
            }
        } else {
            BTreeMap::new()
        };
        Ok(Self { path, models })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = RegistryFile {
            version: REGISTRY_VERSION,
            models: self.models.clone(),
        };
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&file)?)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    pub fn add(&mut self, model: InstalledModel) -> Result<()> {
        self.models.insert(model.id.clone(), model);
        self.save()
    }

    /// Modify one entry in place and persist. `false` when `id` is unknown.
    pub fn update(&mut self, id: &str, f: impl FnOnce(&mut InstalledModel)) -> Result<bool> {
        match self.models.get_mut(id) {
            Some(m) => {
                f(m);
                self.save()?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn remove(&mut self, id: &str) -> Result<Option<InstalledModel>> {
        let removed = self.models.remove(id);
        if removed.is_some() {
            self.save()?;
        }
        Ok(removed)
    }

    pub fn get(&self, id: &str) -> Option<&InstalledModel> {
        self.models.get(id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    /// Installed models, sorted by id.
    pub fn list(&self) -> Vec<&InstalledModel> {
        self.models.values().collect()
    }

    /// Drop entries whose files no longer exist. Returns removed ids.
    pub fn prune_missing(&mut self) -> Result<Vec<String>> {
        let missing: Vec<String> = self
            .models
            .iter()
            .filter(|(_, m)| !m.path.exists())
            .map(|(id, _)| id.clone())
            .collect();
        for id in &missing {
            self.models.remove(id);
        }
        if !missing.is_empty() {
            self.save()?;
        }
        Ok(missing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str, path: PathBuf) -> InstalledModel {
        InstalledModel {
            id: id.into(),
            display_name: id.into(),
            source: ModelSource::HuggingFace {
                repo: "o/r".into(),
                file: "f.gguf".into(),
            },
            format: "gguf".into(),
            path,
            mmproj_path: None,
            kind: Default::default(),
            size_bytes: 3,
            installed_at: Utc::now(),
            capabilities: ModelCapabilities {
                tools: true,
                vision: false,
            },
            context_length: None,
            curated: true,
        }
    }

    #[test]
    fn add_get_remove_persist() {
        let dir = tempfile::tempdir().unwrap();
        let reg_path = dir.path().join("registry.json");
        let file = dir.path().join("a.gguf");
        std::fs::write(&file, b"abc").unwrap();

        let mut reg = ModelRegistry::load(&reg_path).unwrap();
        assert!(reg.list().is_empty());
        reg.add(model("a", file.clone())).unwrap();
        assert!(reg.contains("a"));

        let reg2 = ModelRegistry::load(&reg_path).unwrap();
        assert_eq!(reg2.get("a").unwrap().path, file);
        assert!(reg2.get("a").unwrap().capabilities.tools);
        assert_eq!(reg2.get("a").unwrap().context_length, None);

        let mut reg2 = reg2;
        assert!(reg2.update("a", |m| m.context_length = Some(4096)).unwrap());
        assert!(!reg2.update("zzz", |_| {}).unwrap());
        assert_eq!(
            ModelRegistry::load(&reg_path)
                .unwrap()
                .get("a")
                .unwrap()
                .context_length,
            Some(4096)
        );

        let mut reg3 = ModelRegistry::load(&reg_path).unwrap();
        assert!(reg3.remove("a").unwrap().is_some());
        assert!(reg3.remove("a").unwrap().is_none());
        assert!(ModelRegistry::load(&reg_path).unwrap().list().is_empty());
    }

    #[test]
    fn prune_missing_drops_dangling_entries() {
        let dir = tempfile::tempdir().unwrap();
        let reg_path = dir.path().join("registry.json");
        let mut reg = ModelRegistry::load(&reg_path).unwrap();
        reg.add(model("gone", dir.path().join("nope.gguf")))
            .unwrap();
        let file = dir.path().join("here.gguf");
        std::fs::write(&file, b"x").unwrap();
        reg.add(model("here", file)).unwrap();
        let removed = reg.prune_missing().unwrap();
        assert_eq!(removed, vec!["gone".to_string()]);
        assert_eq!(reg.list().len(), 1);
    }

    #[test]
    fn tolerates_empty_or_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("registry.json");
        std::fs::write(&p, "").unwrap();
        assert!(ModelRegistry::load(&p).unwrap().list().is_empty());
        assert!(ModelRegistry::load(dir.path().join("missing.json"))
            .unwrap()
            .list()
            .is_empty());
    }
}
