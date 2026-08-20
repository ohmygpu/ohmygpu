//! Recipes — the declarative, per-model knowledge that turns "deploy any model"
//! into *data* instead of code.
//!
//! A recipe says, for one model: where the weights live, which backend runs
//! them, which **variants** exist (precision / quantization / file format), what
//! hardware each variant needs, which engine settings are required, how the
//! model talks (chat template, tool / reasoning parsers), and how to prove it
//! works (smoke tests). Most models need *no* recipe — the runtime can derive a
//! default one from Hugging Face metadata — so every field except
//! `schema_version`, `id` and *some* way to find the weights is optional.
//!
//! Format policy: recipes are **hand-authored in YAML** (comments, multi-line
//! templates, what ML contributors already use) and travel as **JSON** on the
//! wire (Management API, machine-generated skeletons). Both deserialize into
//! the same [`Recipe`]. The JSON Schema published in `schemas/recipe-v1.json`
//! is generated from these types (`cargo test -p ohmygpu_core regenerate_schema -- --ignored`).
//!
//! This module is pure data + validation: it does not download, detect
//! hardware, or start anything. The hardware-aware resolver that picks a
//! variant lives with the daemon's `ModelManager`; see `docs/recipes.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};

use crate::catalog::validate_id;

/// The recipe schema version this runtime understands.
pub const SCHEMA_VERSION: u32 = 1;

/// Canonical `$id` of the published JSON Schema.
pub const SCHEMA_ID: &str =
    "https://raw.githubusercontent.com/ohmygpu/ohmygpu/main/schemas/recipe-v1.json";

/// Errors from loading or validating a recipe.
#[derive(Debug, thiserror::Error)]
pub enum RecipeError {
    #[error("cannot read recipe {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("recipe {path}: unsupported extension (expected .yaml, .yml or .json)")]
    Extension { path: String },
    /// The file is not well-formed YAML/JSON for this schema (syntax, types, unknown keys).
    #[error("invalid recipe syntax: {0}")]
    Parse(String),
    /// Well-formed, but semantically wrong (bad id, conflicting fields, …).
    #[error("invalid recipe: {0}")]
    Invalid(String),
}

fn invalid(msg: impl Into<String>) -> RecipeError {
    RecipeError::Invalid(msg.into())
}

fn is_default<T: Default + PartialEq>(v: &T) -> bool {
    *v == T::default()
}

/// `> 0` that also rejects NaN.
fn is_positive(x: f64) -> bool {
    x > 0.0
}

// ───────────────────────────── top level ──────────────────────────────

/// One model, declaratively.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(title = "OhMyGPU model recipe (schema v1)")]
pub struct Recipe {
    /// Recipe schema version. Always `1` for this format; bumped only on
    /// breaking changes (new optional fields do not bump it).
    pub schema_version: u32,

    /// Stable OhMyGPU model id: lowercase letters, digits, `.`, `_`, `-`;
    /// starts alphanumeric. This is the `model` clients send to `/v1/*`.
    #[schemars(regex(pattern = r"^[a-z0-9][a-z0-9._-]*$"))]
    pub id: String,

    /// Human-readable name. Defaults to `id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,

    /// Model family / architecture line, e.g. `qwen3`, `llama-3.1`, `gemma-3`.
    /// Used for grouping and family-level defaults; free-form lowercase slug.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,

    /// One or two sentences for humans (`omg model show`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// License identifier (SPDX where possible, e.g. `apache-2.0`, `llama3.1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Model card or project page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,

    /// Free-form tags for discovery (`reasoning`, `tools`, `coder`, `tiny`, …).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,

    /// What kind of model this is. v1 only knows `llm`; embedding / reranker
    /// kinds will be added without a schema bump.
    #[serde(default, skip_serializing_if = "is_default")]
    pub kind: ModelKind,

    /// The model's native maximum context length (from `config.json`
    /// `max_position_embeddings`). Informational and an upper bound for
    /// `engine.context_length`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,

    /// What the model can do. Drives API validation: tool definitions are only
    /// accepted when `tools` is true, image inputs only when `vision` is true.
    /// Defaults are all `false` — never claim what was not verified.
    #[serde(default, skip_serializing_if = "is_default")]
    pub capabilities: Capabilities,

    /// Default weights location, inherited by every variant that does not set
    /// its own `source`. A recipe without `variants` needs this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,

    /// Default backend, inherited by variants that do not set `backend`.
    /// If neither is set it is inferred: GGUF sources → `llamacpp`, otherwise `vllm`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<Backend>,

    /// How the model talks: chat template override, tool-call / reasoning parsers.
    #[serde(default, skip_serializing_if = "is_default")]
    pub chat: Chat,

    /// Sampling defaults recommended by the model authors, applied when a
    /// request omits the parameter.
    #[serde(default, skip_serializing_if = "is_default")]
    pub generation_defaults: GenerationDefaults,

    /// Engine settings shared by all variants; each variant's `engine` is merged
    /// on top (scalars override, `extra_args` append, `env` overlays).
    #[serde(default, skip_serializing_if = "is_default")]
    pub engine: Engine,

    /// Concrete ways to run this model, in order of preference. The resolver
    /// keeps the variants the current hardware can run and picks the first.
    /// Empty = one implicit variant named `default` built from `source`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<Variant>,

    /// Smoke tests run after start (and by the nightly SLA matrix).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tests: Vec<SmokeTest>,

    /// Provenance: real runs that proved a variant works on given hardware.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub verified: Vec<Verification>,

    /// Free-form extension data for tooling. The only place unknown keys are
    /// allowed; the runtime ignores it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Kind of model. v1: `llm` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    #[default]
    Llm,
}

/// Inference backend that runs a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// vLLM (`vllm serve`): safetensors, CUDA; day-0 coverage through its
    /// Transformers modeling backend.
    Vllm,
    /// llama.cpp (`llama-server`): GGUF; CUDA, Metal, CPU.
    Llamacpp,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Vllm => "vllm",
            Backend::Llamacpp => "llamacpp",
        }
    }

    /// Weight format a backend uses unless the source says otherwise.
    pub fn default_format(&self) -> Format {
        match self {
            Backend::Vllm => Format::Safetensors,
            Backend::Llamacpp => Format::Gguf,
        }
    }
}

/// Weight file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Safetensors,
    Gguf,
}

/// GPU / compute API a variant can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Accelerator {
    Cuda,
    Metal,
    Rocm,
    Cpu,
}

/// Operating system a variant can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Platform {
    Linux,
    Macos,
    Windows,
}

/// What the model can do. All default to `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    /// Native tool calling works (verified with a smoke test).
    #[serde(default)]
    pub tools: bool,
    /// Accepts image inputs.
    #[serde(default)]
    pub vision: bool,
    /// Emits separable reasoning / thinking content.
    #[serde(default)]
    pub reasoning: bool,
}

// ───────────────────────────── source ──────────────────────────────

/// Where weights come from. Exactly one of `hf` or `url`.
///
/// Merge rule (recipe `source` → variant `source`): a variant that names a new
/// *location* (`hf` or `url`) replaces the recipe source entirely; a variant
/// that only sets `file` / `include` / `exclude` / `revision` refines it and
/// inherits the rest.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// Hugging Face repo id, `owner/name`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf: Option<String>,
    /// Git revision on the Hub (branch, tag or commit). Default: the repo's default branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// One file inside the repo — the GGUF to load. For sharded GGUF name the
    /// first shard (`…-00001-of-00005.gguf`); llama.cpp finds the rest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Glob patterns of repo files to download. Empty = the downloader's
    /// default (every weight, tokenizer and config file; no `original/`, no
    /// legacy `.bin`/`.pth` when safetensors exist).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<String>,
    /// Glob patterns to skip, applied after `include`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
    /// Direct http(s) URL to a single file (self-hosted GGUF). Cannot be
    /// combined with `hf`, `file`, `include`, `exclude` or `revision`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

impl Source {
    /// The Hugging Face repo id, if this is a Hub source.
    pub fn repo(&self) -> Option<&str> {
        self.hf.as_deref()
    }

    /// Does this source point at GGUF weights? (`file`/`url` end in `.gguf`,
    /// or an `include` pattern mentions `.gguf`.)
    pub fn is_gguf(&self) -> bool {
        let ends_gguf = |s: &str| s.to_ascii_lowercase().ends_with(".gguf");
        self.file.as_deref().map(ends_gguf).unwrap_or(false)
            || self
                .url
                .as_deref()
                .map(|u| ends_gguf(u.split('?').next().unwrap_or(u)))
                .unwrap_or(false)
            || self
                .include
                .iter()
                .any(|p| p.to_ascii_lowercase().contains(".gguf"))
    }

    /// Apply the merge rule documented on [`Source`].
    pub fn merged(base: Option<&Source>, overlay: Option<&Source>) -> Option<Source> {
        match (base, overlay) {
            (None, None) => None,
            (Some(b), None) => Some(b.clone()),
            (None, Some(o)) => Some(o.clone()),
            (Some(b), Some(o)) => {
                if o.hf.is_some() || o.url.is_some() {
                    return Some(o.clone());
                }
                let mut m = b.clone();
                if o.revision.is_some() {
                    m.revision = o.revision.clone();
                }
                if o.file.is_some() {
                    m.file = o.file.clone();
                }
                if !o.include.is_empty() {
                    m.include = o.include.clone();
                }
                if !o.exclude.is_empty() {
                    m.exclude = o.exclude.clone();
                }
                Some(m)
            }
        }
    }

    fn validate(&self, ctx: &str) -> Result<(), RecipeError> {
        match (&self.hf, &self.url) {
            (None, None) => return Err(invalid(format!("{ctx}: source needs `hf` or `url`"))),
            (Some(_), Some(_)) => {
                return Err(invalid(format!(
                    "{ctx}: source has both `hf` and `url`; use one"
                )))
            }
            (Some(repo), None) => {
                if !is_valid_repo_id(repo) {
                    return Err(invalid(format!(
                        "{ctx}: `hf` must be a Hugging Face repo id like `owner/name`, got '{repo}'"
                    )));
                }
            }
            (None, Some(url)) => {
                if !(url.starts_with("https://") || url.starts_with("http://")) {
                    return Err(invalid(format!(
                        "{ctx}: `url` must start with http:// or https://"
                    )));
                }
                if self.revision.is_some()
                    || self.file.is_some()
                    || !self.include.is_empty()
                    || !self.exclude.is_empty()
                {
                    return Err(invalid(format!(
                        "{ctx}: `url` sources are a single file; `revision`, `file`, `include` and `exclude` only apply to `hf`"
                    )));
                }
            }
        }
        if let Some(file) = &self.file {
            if file.is_empty() || file.starts_with('/') || file.split('/').any(|p| p == "..") {
                return Err(invalid(format!(
                    "{ctx}: `file` must be a relative path inside the repo, got '{file}'"
                )));
            }
        }
        if let Some(rev) = &self.revision {
            if rev.trim().is_empty() {
                return Err(invalid(format!("{ctx}: `revision` must not be empty")));
            }
        }
        for (name, pats) in [("include", &self.include), ("exclude", &self.exclude)] {
            if pats.iter().any(|p| p.trim().is_empty()) {
                return Err(invalid(format!(
                    "{ctx}: `{name}` patterns must not be empty"
                )));
            }
        }
        Ok(())
    }
}

fn is_valid_repo_id(repo: &str) -> bool {
    let mut parts = repo.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    let ok_char = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-');
    !owner.is_empty()
        && !name.is_empty()
        && owner
            .chars()
            .next()
            .map(|c| c.is_ascii_alphanumeric())
            .unwrap_or(false)
        && owner.chars().all(ok_char)
        && name.chars().all(ok_char)
        && name != "."
        && name != ".."
}

// ───────────────────────────── chat / generation ──────────────────────────────

/// How the model talks.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Chat {
    /// Override the chat template shipped in `tokenizer_config.json` / the GGUF.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<ChatTemplate>,
    /// How tool calls appear in the model's output. Names follow vLLM's
    /// `--tool-call-parser` vocabulary (`hermes`, `llama3_json`, `mistral`,
    /// `qwen3_coder`, `deepseek_v3`, `glm45`, `kimi_k2`, …). llama.cpp derives
    /// this from the Jinja template and ignores the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_parser: Option<String>,
    /// How reasoning content is delimited. Names follow vLLM's
    /// `--reasoning-parser` vocabulary (`deepseek_r1`, `qwen3`, `glm45`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,
    /// Extra stop strings applied to every request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

impl Chat {
    fn validate(&self) -> Result<(), RecipeError> {
        if let Some(t) = &self.template {
            match (&t.file, &t.inline) {
                (Some(_), Some(_)) => {
                    return Err(invalid("chat.template: use `file` or `inline`, not both"))
                }
                (None, None) => return Err(invalid("chat.template: needs `file` or `inline`")),
                (Some(f), None) => {
                    if f.trim().is_empty() || f.starts_with('/') || f.split('/').any(|p| p == "..")
                    {
                        return Err(invalid(
                            "chat.template.file must be a relative path next to the recipe",
                        ));
                    }
                }
                (None, Some(s)) => {
                    if s.trim().is_empty() {
                        return Err(invalid("chat.template.inline must not be empty"));
                    }
                }
            }
        }
        for (name, v) in [
            ("tool_call_parser", &self.tool_call_parser),
            ("reasoning_parser", &self.reasoning_parser),
        ] {
            if let Some(v) = v {
                if v.trim().is_empty() {
                    return Err(invalid(format!("chat.{name} must not be empty")));
                }
            }
        }
        if self.stop.iter().any(|s| s.is_empty()) {
            return Err(invalid("chat.stop strings must not be empty"));
        }
        Ok(())
    }
}

/// A Jinja chat template, either a file next to the recipe or inline.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatTemplate {
    /// Path relative to the recipe file, e.g. `templates/qwen3.jinja`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// The template text itself (use a YAML `|` block).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline: Option<String>,
}

/// Sampling defaults recommended by the model authors.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GenerationDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub repetition_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Default cap on generated tokens when a request does not set one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_output_tokens: Option<u32>,
}

impl GenerationDefaults {
    fn validate(&self) -> Result<(), RecipeError> {
        let unit = |name: &str, v: Option<f32>| -> Result<(), RecipeError> {
            match v {
                Some(x) if !(0.0..=1.0).contains(&x) => Err(invalid(format!(
                    "generation_defaults.{name} must be within 0..=1, got {x}"
                ))),
                _ => Ok(()),
            }
        };
        if let Some(t) = self.temperature {
            if t < 0.0 {
                return Err(invalid("generation_defaults.temperature must be >= 0"));
            }
        }
        unit("top_p", self.top_p)?;
        unit("min_p", self.min_p)?;
        if let Some(r) = self.repetition_penalty {
            if r < 0.0 {
                return Err(invalid(
                    "generation_defaults.repetition_penalty must be >= 0",
                ));
            }
        }
        if self.max_output_tokens == Some(0) {
            return Err(invalid(
                "generation_defaults.max_output_tokens must be >= 1",
            ));
        }
        Ok(())
    }
}

// ───────────────────────────── engine ──────────────────────────────

/// Engine settings. The backend-agnostic scalars are mapped by every backend;
/// `vllm` / `llamacpp` hold the stable, high-value flags of each engine plus an
/// `extra_args` / `env` escape hatch for everything else. New exotic flags go
/// into `extra_args` first and are promoted to typed fields once common.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Engine {
    /// Context window to allocate. Must not exceed the recipe's `context_length`.
    /// Default: the backend's default, capped by the model's native length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub context_length: Option<u32>,
    /// Concurrent sequences (vLLM `--max-num-seqs`, llama.cpp `--parallel`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_concurrency: Option<u32>,
    /// GPUs to shard across (vLLM `--tensor-parallel-size`). llama.cpp splits
    /// layers across visible GPUs on its own and ignores this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub tensor_parallel: Option<u32>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub vllm: VllmSettings,
    #[serde(default, skip_serializing_if = "is_default")]
    pub llamacpp: LlamaCppSettings,
}

fn or<T: Clone>(overlay: &Option<T>, base: &Option<T>) -> Option<T> {
    overlay.clone().or_else(|| base.clone())
}

impl Engine {
    /// Variant settings on top of recipe settings: scalars override,
    /// `extra_args` append, `env` overlays (variant wins).
    pub fn merged(base: &Engine, overlay: &Engine) -> Engine {
        Engine {
            context_length: or(&overlay.context_length, &base.context_length),
            max_concurrency: or(&overlay.max_concurrency, &base.max_concurrency),
            tensor_parallel: or(&overlay.tensor_parallel, &base.tensor_parallel),
            vllm: VllmSettings::merged(&base.vllm, &overlay.vllm),
            llamacpp: LlamaCppSettings::merged(&base.llamacpp, &overlay.llamacpp),
        }
    }

    fn validate(&self, ctx: &str) -> Result<(), RecipeError> {
        for (name, v) in [
            ("context_length", self.context_length),
            ("max_concurrency", self.max_concurrency),
            ("tensor_parallel", self.tensor_parallel),
        ] {
            if v == Some(0) {
                return Err(invalid(format!("{ctx}.{name} must be >= 1")));
            }
        }
        self.vllm.validate(ctx)?;
        self.llamacpp.validate(ctx)
    }
}

/// vLLM-specific settings (`vllm serve` flags).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VllmSettings {
    /// Oldest vLLM release known to run this model, e.g. `0.10.1` (semver).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_version: Option<String>,
    /// `--dtype`: `auto`, `bfloat16`, `float16`, `float32`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    /// `--quantization`: force a method (`awq`, `gptq`, `fp8`, `compressed-tensors`,
    /// `bitsandbytes`, `gguf`). Rarely needed — vLLM reads `quantization_config`
    /// from `config.json`. Note `variants[].quantization` is the human label;
    /// this is the engine override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// `--model-impl`: `auto` (default), `vllm`, or `transformers` to force the
    /// Transformers modeling backend (day-0 fallback for new architectures).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_impl: Option<String>,
    /// `--trust-remote-code`. Executes Python from the model repo — set only
    /// when the architecture needs it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_remote_code: Option<bool>,
    /// `--gpu-memory-utilization`, `(0, 1]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub gpu_memory_utilization: Option<f32>,
    /// `--max-num-batched-tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_num_batched_tokens: Option<u32>,
    /// `--enable-prefix-caching` / `--no-enable-prefix-caching`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_prefix_caching: Option<bool>,
    /// `--kv-cache-dtype`: `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_dtype: Option<String>,
    /// `--enforce-eager` (skip CUDA graphs; faster start, slower decode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enforce_eager: Option<bool>,
    /// `--tokenizer`: use another repo's tokenizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    /// Raw extra CLI arguments, appended last. Variant values append to recipe values.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
    /// Environment variables for the engine process (e.g. `VLLM_USE_V1: "1"`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
}

impl VllmSettings {
    fn merged(base: &Self, overlay: &Self) -> Self {
        let mut env = base.env.clone();
        env.extend(overlay.env.iter().map(|(k, v)| (k.clone(), v.clone())));
        let mut extra_args = base.extra_args.clone();
        extra_args.extend(overlay.extra_args.iter().cloned());
        VllmSettings {
            min_version: or(&overlay.min_version, &base.min_version),
            dtype: or(&overlay.dtype, &base.dtype),
            quantization: or(&overlay.quantization, &base.quantization),
            model_impl: or(&overlay.model_impl, &base.model_impl),
            trust_remote_code: or(&overlay.trust_remote_code, &base.trust_remote_code),
            gpu_memory_utilization: or(
                &overlay.gpu_memory_utilization,
                &base.gpu_memory_utilization,
            ),
            max_num_batched_tokens: or(
                &overlay.max_num_batched_tokens,
                &base.max_num_batched_tokens,
            ),
            enable_prefix_caching: or(&overlay.enable_prefix_caching, &base.enable_prefix_caching),
            kv_cache_dtype: or(&overlay.kv_cache_dtype, &base.kv_cache_dtype),
            enforce_eager: or(&overlay.enforce_eager, &base.enforce_eager),
            tokenizer: or(&overlay.tokenizer, &base.tokenizer),
            extra_args,
            env,
        }
    }

    fn validate(&self, ctx: &str) -> Result<(), RecipeError> {
        if let Some(v) = &self.min_version {
            semver::Version::parse(v).map_err(|e| {
                invalid(format!(
                    "{ctx}.vllm.min_version must be a semver version like 0.10.1, got '{v}' ({e})"
                ))
            })?;
        }
        if let Some(u) = self.gpu_memory_utilization {
            if !(u > 0.0 && u <= 1.0) {
                return Err(invalid(format!(
                    "{ctx}.vllm.gpu_memory_utilization must be within (0, 1], got {u}"
                )));
            }
        }
        if let Some(m) = &self.model_impl {
            if !matches!(m.as_str(), "auto" | "vllm" | "transformers") {
                return Err(invalid(format!(
                    "{ctx}.vllm.model_impl must be auto, vllm or transformers, got '{m}'"
                )));
            }
        }
        if self.max_num_batched_tokens == Some(0) {
            return Err(invalid(format!(
                "{ctx}.vllm.max_num_batched_tokens must be >= 1"
            )));
        }
        if self.extra_args.iter().any(|a| a.trim().is_empty()) {
            return Err(invalid(format!(
                "{ctx}.vllm.extra_args must not contain empty strings"
            )));
        }
        Ok(())
    }
}

/// llama.cpp-specific settings (`llama-server` flags).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LlamaCppSettings {
    /// Oldest llama.cpp release known to run this model, e.g. `b6000`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = r"^b[0-9]+$"))]
    pub min_release: Option<String>,
    /// `--n-gpu-layers`. Default: all that fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<i32>,
    /// `--threads`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub threads: Option<u32>,
    /// `--batch-size`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub batch_size: Option<u32>,
    /// `--ubatch-size`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub ubatch_size: Option<u32>,
    /// `--flash-attn`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flash_attention: Option<bool>,
    /// `--cache-type-k` (`f16`, `q8_0`, `q4_0`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_type_k: Option<String>,
    /// `--cache-type-v`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_type_v: Option<String>,
    /// `--mmap` / `--no-mmap`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmap: Option<bool>,
    /// `--mlock`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlock: Option<bool>,
    /// Raw extra CLI arguments, appended last. Variant values append to recipe values.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
    /// Environment variables for the engine process.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
}

impl LlamaCppSettings {
    fn merged(base: &Self, overlay: &Self) -> Self {
        let mut env = base.env.clone();
        env.extend(overlay.env.iter().map(|(k, v)| (k.clone(), v.clone())));
        let mut extra_args = base.extra_args.clone();
        extra_args.extend(overlay.extra_args.iter().cloned());
        LlamaCppSettings {
            min_release: or(&overlay.min_release, &base.min_release),
            gpu_layers: or(&overlay.gpu_layers, &base.gpu_layers),
            threads: or(&overlay.threads, &base.threads),
            batch_size: or(&overlay.batch_size, &base.batch_size),
            ubatch_size: or(&overlay.ubatch_size, &base.ubatch_size),
            flash_attention: or(&overlay.flash_attention, &base.flash_attention),
            cache_type_k: or(&overlay.cache_type_k, &base.cache_type_k),
            cache_type_v: or(&overlay.cache_type_v, &base.cache_type_v),
            mmap: or(&overlay.mmap, &base.mmap),
            mlock: or(&overlay.mlock, &base.mlock),
            extra_args,
            env,
        }
    }

    fn validate(&self, ctx: &str) -> Result<(), RecipeError> {
        if let Some(r) = &self.min_release {
            let ok =
                r.len() > 1 && r.starts_with('b') && r[1..].chars().all(|c| c.is_ascii_digit());
            if !ok {
                return Err(invalid(format!(
                    "{ctx}.llamacpp.min_release must look like b6000, got '{r}'"
                )));
            }
        }
        for (name, v) in [
            ("threads", self.threads),
            ("batch_size", self.batch_size),
            ("ubatch_size", self.ubatch_size),
        ] {
            if v == Some(0) {
                return Err(invalid(format!("{ctx}.llamacpp.{name} must be >= 1")));
            }
        }
        if self.extra_args.iter().any(|a| a.trim().is_empty()) {
            return Err(invalid(format!(
                "{ctx}.llamacpp.extra_args must not contain empty strings"
            )));
        }
        Ok(())
    }
}

// ───────────────────────────── variants ──────────────────────────────

/// One concrete way to run the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    /// Unique within the recipe; same character rules as model ids
    /// (`bf16`, `fp8`, `awq`, `q4_k_m`, `bf16-tp2`, …). Users can force one
    /// with `omg run <id>@<variant>` / `"variant"` in the start request.
    #[schemars(regex(pattern = r"^[a-z0-9][a-z0-9._-]*$"))]
    pub name: String,
    /// Why this variant exists / when to pick it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Backend for this variant. Default: the recipe's `backend`, else inferred
    /// from the source (GGUF → `llamacpp`, otherwise `vllm`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<Backend>,
    /// Weight format. Default: inferred from the source / backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Format>,
    /// Weights for this variant; merged over the recipe `source` (see [`Source`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    /// Human label of the precision / quantization: `bf16`, `fp16`, `fp8`,
    /// `awq`, `gptq`, `int4`, `q4_k_m`, `q8_0`, … Used for display and for
    /// `--quant` filters; engines read the real quantization from the weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Download size in GB (10^9 bytes) — for ETAs and disk checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub size_gb: Option<f64>,
    /// Hardware this variant needs. Missing numbers mean "unknown" — the
    /// resolver does not filter on them (and warns).
    #[serde(default, skip_serializing_if = "is_default")]
    pub requires: Requirements,
    /// Engine settings merged over the recipe's `engine`.
    #[serde(default, skip_serializing_if = "is_default")]
    pub engine: Engine,
}

/// Hardware requirements of a variant, as measured for its `engine.context_length`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Requirements {
    /// Total GPU memory needed in GB across all GPUs used (weights + KV cache
    /// at the variant's context length). On Apple Silicon this is unified memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub vram_gb: Option<f64>,
    /// System RAM needed in GB (CPU inference, or mmap'd GGUF).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub ram_gb: Option<f64>,
    /// Number of GPUs used. Default 1. Should match `engine.tensor_parallel` for vLLM.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub gpus: Option<u32>,
    /// Accelerators this variant runs on. Empty = whatever the backend supports
    /// (vLLM: cuda; llama.cpp: cuda, metal, cpu).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accelerators: Vec<Accelerator>,
    /// Operating systems this variant runs on. Empty = any the backend supports.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub platforms: Vec<Platform>,
    /// Minimum NVIDIA compute capability as `major.minor` (`"7.5"` Turing,
    /// `"8.0"` Ampere, `"8.9"` Ada, `"9.0"` Hopper). bf16 needs 8.0, fp8 8.9.
    /// Quote it in YAML — `8.0` unquoted is the number 8.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "de_compute_capability"
    )]
    #[schemars(with = "Option<String>", regex(pattern = r"^[0-9]+\.[0-9]$"))]
    pub min_compute_capability: Option<String>,
}

/// Accept `"8.0"`, `8.0` or `8` and normalise to `"8.0"`.
fn de_compute_capability<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Str(String),
        Num(f64),
    }
    Ok(match Option::<Raw>::deserialize(d)? {
        None => None,
        Some(Raw::Str(s)) => Some(s),
        Some(Raw::Num(n)) => Some(format!("{n:.1}")),
    })
}

impl Requirements {
    fn validate(&self, ctx: &str) -> Result<(), RecipeError> {
        for (name, v) in [("vram_gb", self.vram_gb), ("ram_gb", self.ram_gb)] {
            if let Some(x) = v {
                if !is_positive(x) {
                    return Err(invalid(format!(
                        "{ctx}.requires.{name} must be > 0, got {x}"
                    )));
                }
            }
        }
        if self.gpus == Some(0) {
            return Err(invalid(format!("{ctx}.requires.gpus must be >= 1")));
        }
        if let Some(cc) = &self.min_compute_capability {
            let ok = match cc.split_once('.') {
                Some((major, minor)) => {
                    !major.is_empty()
                        && major.chars().all(|c| c.is_ascii_digit())
                        && minor.len() == 1
                        && minor.chars().all(|c| c.is_ascii_digit())
                }
                None => false,
            };
            if !ok {
                return Err(invalid(format!(
                    "{ctx}.requires.min_compute_capability must be `major.minor` like \"8.0\", got '{cc}'"
                )));
            }
        }
        Ok(())
    }
}

/// A variant with every inherited field filled in — what the resolver and the
/// backends consume.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedVariant {
    pub name: String,
    pub description: Option<String>,
    pub backend: Backend,
    pub format: Format,
    pub source: Source,
    pub quantization: Option<String>,
    pub size_gb: Option<f64>,
    pub requires: Requirements,
    pub engine: Engine,
}

// ───────────────────────────── tests / provenance ──────────────────────────────

/// A smoke test: one prompt, one expectation. With no `expect_*` the test
/// only requires a successful, non-empty response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SmokeTest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// User message to send.
    pub prompt: String,
    /// Optional system message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Generation cap for this test. Default 32.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_tokens: Option<u32>,
    /// The response text must contain this substring (case-insensitive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expect_contains: Option<String>,
    /// The response text must match this regular expression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expect_regex: Option<String>,
    /// The runtime sends a fixed `get_weather(city)` tool definition with the
    /// prompt and requires the model to answer with a tool call. Use on
    /// recipes that claim `capabilities.tools`.
    #[serde(default, skip_serializing_if = "is_default")]
    pub expect_tool_call: bool,
}

impl SmokeTest {
    fn validate(&self, idx: usize) -> Result<(), RecipeError> {
        if self.prompt.trim().is_empty() {
            return Err(invalid(format!("tests[{idx}].prompt must not be empty")));
        }
        if self.max_tokens == Some(0) {
            return Err(invalid(format!("tests[{idx}].max_tokens must be >= 1")));
        }
        if let Some(s) = &self.expect_contains {
            if s.is_empty() {
                return Err(invalid(format!(
                    "tests[{idx}].expect_contains must not be empty"
                )));
            }
        }
        if let Some(s) = &self.expect_regex {
            if s.is_empty() {
                return Err(invalid(format!(
                    "tests[{idx}].expect_regex must not be empty"
                )));
            }
        }
        Ok(())
    }
}

/// A real run that proved a variant works.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Verification {
    /// `YYYY-MM-DD`.
    #[schemars(regex(pattern = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"))]
    pub date: String,
    /// Variant name that was run.
    pub variant: String,
    /// Free text, e.g. `1x RTX 4090 24GB` or `Apple M3 Max 64GB`.
    pub hardware: String,
    /// Engine and version, e.g. `vllm 0.10.1` or `llama.cpp b6234`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_version: Option<String>,
    /// Who ran it (handle or name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl Verification {
    fn validate(&self, idx: usize, variants: &BTreeSet<&str>) -> Result<(), RecipeError> {
        let date_ok = self.date.len() == 10
            && self.date.as_bytes()[4] == b'-'
            && self.date.as_bytes()[7] == b'-'
            && self.date.chars().enumerate().all(|(i, c)| {
                if i == 4 || i == 7 {
                    c == '-'
                } else {
                    c.is_ascii_digit()
                }
            });
        if !date_ok {
            return Err(invalid(format!(
                "verified[{idx}].date must be YYYY-MM-DD, got '{}'",
                self.date
            )));
        }
        if !variants.contains(self.variant.as_str()) {
            return Err(invalid(format!(
                "verified[{idx}].variant '{}' is not a variant of this recipe",
                self.variant
            )));
        }
        if self.hardware.trim().is_empty() {
            return Err(invalid(format!(
                "verified[{idx}].hardware must not be empty"
            )));
        }
        Ok(())
    }
}

// ───────────────────────────── loading / validation ──────────────────────────────

impl Recipe {
    /// Parse and validate a YAML recipe.
    pub fn from_yaml(text: &str) -> Result<Recipe, RecipeError> {
        let recipe: Recipe =
            serde_saphyr::from_str(text).map_err(|e| RecipeError::Parse(e.to_string()))?;
        recipe.validate()?;
        Ok(recipe)
    }

    /// Parse and validate a JSON recipe (the wire form).
    pub fn from_json(text: &str) -> Result<Recipe, RecipeError> {
        let recipe: Recipe =
            serde_json::from_str(text).map_err(|e| RecipeError::Parse(e.to_string()))?;
        recipe.validate()?;
        Ok(recipe)
    }

    /// Load by extension: `.yaml` / `.yml` / `.json`.
    pub fn from_path(path: &Path) -> Result<Recipe, RecipeError> {
        let text = std::fs::read_to_string(path).map_err(|source| RecipeError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        let parsed = match ext.as_str() {
            "yaml" | "yml" => Self::from_yaml(&text),
            "json" => Self::from_json(&text),
            _ => {
                return Err(RecipeError::Extension {
                    path: path.display().to_string(),
                })
            }
        };
        parsed.map_err(|e| match e {
            RecipeError::Parse(m) => RecipeError::Parse(format!("{}: {m}", path.display())),
            RecipeError::Invalid(m) => RecipeError::Invalid(format!("{}: {m}", path.display())),
            other => other,
        })
    }

    /// Wire form.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("recipe serializes")
    }

    /// Authoring form.
    pub fn to_yaml(&self) -> Result<String, RecipeError> {
        serde_saphyr::to_string(self).map_err(|e| RecipeError::Parse(e.to_string()))
    }

    /// The JSON Schema for this format (draft 2020-12), with `$id` set.
    pub fn json_schema() -> serde_json::Value {
        let schema = schemars::schema_for!(Recipe);
        let mut value = serde_json::to_value(schema).expect("schema serializes");
        if let serde_json::Value::Object(map) = &mut value {
            map.insert("$id".into(), serde_json::Value::String(SCHEMA_ID.into()));
        }
        value
    }

    /// `display_name`, falling back to `id`.
    pub fn display_name(&self) -> &str {
        self.display_name.as_deref().unwrap_or(&self.id)
    }

    /// Every variant with inherited fields filled in, in recipe order.
    /// A recipe without `variants` yields one variant named `default`.
    pub fn resolved_variants(&self) -> Result<Vec<ResolvedVariant>, RecipeError> {
        if self.variants.is_empty() {
            let source = self.source.clone().ok_or_else(|| {
                invalid("recipe has no `variants` and no `source`; add one of them")
            })?;
            source.validate("source")?;
            let backend = self.backend.unwrap_or(infer_backend(&source));
            let format = infer_format(backend, &source);
            return Ok(vec![ResolvedVariant {
                name: "default".to_string(),
                description: None,
                backend,
                format,
                source,
                quantization: None,
                size_gb: None,
                requires: Requirements::default(),
                engine: self.engine.clone(),
            }]);
        }
        let mut out = Vec::with_capacity(self.variants.len());
        for v in &self.variants {
            let ctx = format!("variants[{}]", v.name);
            let source =
                Source::merged(self.source.as_ref(), v.source.as_ref()).ok_or_else(|| {
                    invalid(format!(
                        "{ctx}: no source (set `source` on the recipe or the variant)"
                    ))
                })?;
            source.validate(&ctx)?;
            let backend = v.backend.or(self.backend).unwrap_or(infer_backend(&source));
            let format = v.format.unwrap_or(infer_format(backend, &source));
            out.push(ResolvedVariant {
                name: v.name.clone(),
                description: v.description.clone(),
                backend,
                format,
                source,
                quantization: v.quantization.clone(),
                size_gb: v.size_gb,
                requires: v.requires.clone(),
                engine: Engine::merged(&self.engine, &v.engine),
            });
        }
        Ok(out)
    }

    /// Semantic validation. Called by every `from_*`; cheap to call again.
    pub fn validate(&self) -> Result<(), RecipeError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(invalid(format!(
                "schema_version {} is not supported by this runtime (supports {SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        validate_id(&self.id).map_err(invalid)?;
        for (name, v) in [
            ("display_name", &self.display_name),
            ("family", &self.family),
            ("license", &self.license),
            ("homepage", &self.homepage),
        ] {
            if let Some(s) = v {
                if s.trim().is_empty() {
                    return Err(invalid(format!("{name} must not be empty when set")));
                }
            }
        }
        if let Some(h) = &self.homepage {
            if !(h.starts_with("https://") || h.starts_with("http://")) {
                return Err(invalid("homepage must be an http(s) URL"));
            }
        }
        if self.tags.iter().any(|t| t.trim().is_empty()) {
            return Err(invalid("tags must not contain empty strings"));
        }
        if self.context_length == Some(0) {
            return Err(invalid("context_length must be >= 1"));
        }
        self.chat.validate()?;
        self.generation_defaults.validate()?;
        self.engine.validate("engine")?;

        let mut names = BTreeSet::new();
        for v in &self.variants {
            validate_id(&v.name).map_err(|e| invalid(format!("variant name: {e}")))?;
            if !names.insert(v.name.as_str()) {
                return Err(invalid(format!("duplicate variant name '{}'", v.name)));
            }
            let ctx = format!("variants[{}]", v.name);
            if let Some(s) = v.size_gb {
                if !is_positive(s) {
                    return Err(invalid(format!("{ctx}.size_gb must be > 0, got {s}")));
                }
            }
            if let Some(q) = &v.quantization {
                if q.trim().is_empty() {
                    return Err(invalid(format!(
                        "{ctx}.quantization must not be empty when set"
                    )));
                }
            }
            v.requires.validate(&ctx)?;
            v.engine.validate(&ctx.clone().add(".engine"))?;
        }

        let resolved = self.resolved_variants()?;
        for r in &resolved {
            let ctx = format!("variants[{}]", r.name);
            if r.backend == Backend::Llamacpp && r.format != Format::Gguf {
                return Err(invalid(format!(
                    "{ctx}: backend llamacpp needs gguf weights, got {:?}",
                    r.format
                )));
            }
            if r.format == Format::Gguf {
                let names_gguf = r.source.file.is_some()
                    || r.source.url.is_some()
                    || r.source
                        .include
                        .iter()
                        .any(|p| p.to_ascii_lowercase().contains(".gguf"));
                if !names_gguf {
                    return Err(invalid(format!(
                        "{ctx}: backend {} needs gguf weights: set source.file to the .gguf to load (or an include pattern)",
                        r.backend.as_str()
                    )));
                }
                if let Some(url) = &r.source.url {
                    if !url
                        .split('?')
                        .next()
                        .unwrap_or(url)
                        .to_ascii_lowercase()
                        .ends_with(".gguf")
                    {
                        return Err(invalid(format!(
                            "{ctx}: format is gguf but source.url does not point at a .gguf file"
                        )));
                    }
                }
            }
            if let Some(file) = &r.source.file {
                let is_gguf_file = file.to_ascii_lowercase().ends_with(".gguf");
                if r.format == Format::Gguf && !is_gguf_file {
                    return Err(invalid(format!(
                        "{ctx}: format is gguf but source.file '{file}' is not a .gguf"
                    )));
                }
                if r.format == Format::Safetensors && is_gguf_file {
                    return Err(invalid(format!(
                        "{ctx}: source.file is a .gguf but format is safetensors (set backend: llamacpp or format: gguf)"
                    )));
                }
            }
            if let (Some(native), Some(alloc)) = (self.context_length, r.engine.context_length) {
                if alloc > native {
                    return Err(invalid(format!(
                        "{ctx}: engine.context_length {alloc} exceeds the model's context_length {native}"
                    )));
                }
            }
            if let (Some(tp), Some(gpus)) = (r.engine.tensor_parallel, r.requires.gpus) {
                if r.backend == Backend::Vllm && tp != gpus {
                    return Err(invalid(format!(
                        "{ctx}: engine.tensor_parallel {tp} does not match requires.gpus {gpus}"
                    )));
                }
            }
        }

        for (i, t) in self.tests.iter().enumerate() {
            t.validate(i)?;
        }
        let variant_names: BTreeSet<&str> = resolved.iter().map(|r| r.name.as_str()).collect();
        for (i, v) in self.verified.iter().enumerate() {
            v.validate(i, &variant_names)?;
        }
        Ok(())
    }
}

fn infer_backend(source: &Source) -> Backend {
    if source.is_gguf() {
        Backend::Llamacpp
    } else {
        Backend::Vllm
    }
}

fn infer_format(backend: Backend, source: &Source) -> Format {
    if source.is_gguf() {
        Format::Gguf
    } else {
        backend.default_format()
    }
}

trait StrAdd {
    fn add(self, s: &str) -> String;
}
impl StrAdd for String {
    fn add(mut self, s: &str) -> String {
        self.push_str(s);
        self
    }
}

// ───────────────────────────── tests ──────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn schema_path() -> PathBuf {
        repo_root().join("schemas/recipe-v1.json")
    }

    fn example_recipe_paths() -> Vec<PathBuf> {
        let dir = repo_root().join("recipes");
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("recipes dir {}: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("yaml") | Some("yml") | Some("json")
                )
            })
            .collect();
        paths.sort();
        paths
    }

    fn minimal(extra: &str) -> String {
        format!("schema_version: 1\nid: test-model\n{extra}")
    }

    #[test]
    fn example_recipes_load_resolve_and_round_trip() {
        let paths = example_recipe_paths();
        assert!(paths.len() >= 3, "expected example recipes under recipes/");
        for path in paths {
            let recipe =
                Recipe::from_path(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let variants = recipe.resolved_variants().unwrap();
            assert!(!variants.is_empty(), "{}: no variants", path.display());

            // JSON is the wire form: must round-trip exactly.
            let json = recipe.to_json_pretty();
            let back = Recipe::from_json(&json).unwrap();
            assert_eq!(back, recipe, "{}: JSON round trip", path.display());

            // YAML is the authoring form: our own output must load back.
            let yaml = recipe.to_yaml().unwrap();
            let back = Recipe::from_yaml(&yaml)
                .unwrap_or_else(|e| panic!("{}: YAML round trip: {e}\n{yaml}", path.display()));
            assert_eq!(back, recipe, "{}: YAML round trip", path.display());
        }
    }

    #[test]
    fn minimal_safetensors_recipe_becomes_one_vllm_variant() {
        let r = Recipe::from_yaml(&minimal("source: { hf: Qwen/Qwen3-8B }\n")).unwrap();
        let v = r.resolved_variants().unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].name, "default");
        assert_eq!(v[0].backend, Backend::Vllm);
        assert_eq!(v[0].format, Format::Safetensors);
        assert_eq!(v[0].source.repo(), Some("Qwen/Qwen3-8B"));
    }

    #[test]
    fn minimal_gguf_recipe_becomes_one_llamacpp_variant() {
        let r = Recipe::from_yaml(&minimal(
            "source:\n  hf: bartowski/SmolLM2-135M-Instruct-GGUF\n  file: SmolLM2-135M-Instruct-Q8_0.gguf\n",
        ))
        .unwrap();
        let v = r.resolved_variants().unwrap();
        assert_eq!(v[0].backend, Backend::Llamacpp);
        assert_eq!(v[0].format, Format::Gguf);
    }

    #[test]
    fn url_gguf_source_is_llamacpp() {
        let r = Recipe::from_yaml(&minimal(
            "source: { url: https://example.com/m.gguf?x=1 }\n",
        ))
        .unwrap();
        assert_eq!(r.resolved_variants().unwrap()[0].backend, Backend::Llamacpp);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let err = Recipe::from_yaml(&minimal("source: { hf: a/b }\ncolour: red\n")).unwrap_err();
        assert!(matches!(err, RecipeError::Parse(_)), "{err}");
        assert!(err.to_string().contains("colour"), "{err}");
    }

    #[test]
    fn schema_version_must_match() {
        let err = Recipe::from_yaml("schema_version: 2\nid: x\nsource: { hf: a/b }\n").unwrap_err();
        assert!(err.to_string().contains("schema_version 2"), "{err}");
    }

    #[test]
    fn bad_ids_are_rejected() {
        for bad in ["Qwen3", "-x", "a b", "a/b", "a..b", ""] {
            let err = Recipe::from_yaml(&format!(
                "schema_version: 1\nid: \"{bad}\"\nsource: {{ hf: a/b }}\n"
            ))
            .unwrap_err();
            assert!(matches!(err, RecipeError::Invalid(_)), "{bad}: {err}");
        }
    }

    #[test]
    fn recipe_without_source_or_variants_is_rejected() {
        let err = Recipe::from_yaml(&minimal("")).unwrap_err();
        assert!(
            err.to_string().contains("no `variants` and no `source`"),
            "{err}"
        );
    }

    #[test]
    fn variant_without_any_source_is_rejected() {
        let err = Recipe::from_yaml(&minimal("variants:\n  - name: a\n")).unwrap_err();
        assert!(err.to_string().contains("variants[a]: no source"), "{err}");
    }

    #[test]
    fn duplicate_variant_names_are_rejected() {
        let err = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\nvariants:\n  - name: a\n  - name: a\n",
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("duplicate variant name 'a'"),
            "{err}"
        );
    }

    #[test]
    fn llamacpp_requires_gguf() {
        let err = Recipe::from_yaml(&minimal(
            "source: { hf: Qwen/Qwen3-8B }\nbackend: llamacpp\n",
        ))
        .unwrap_err();
        assert!(err.to_string().contains("needs gguf weights"), "{err}");

        let err = Recipe::from_yaml(&minimal(
            "source: { hf: a/b, file: model.gguf }\nbackend: vllm\nvariants:\n  - name: x\n    format: safetensors\n",
        ))
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("is a .gguf but format is safetensors"),
            "{err}"
        );
    }

    #[test]
    fn source_rules() {
        for (yaml, needle) in [
            (
                "source: { hf: a/b, url: https://x/y.gguf }\n",
                "both `hf` and `url`",
            ),
            ("source: { hf: justname }\n", "owner/name"),
            ("source: { url: ftp://x/y.gguf }\n", "http:// or https://"),
            (
                "source: { url: https://x/y.gguf, file: z.gguf }\n",
                "single file",
            ),
            ("source: { hf: a/b, file: /abs.gguf }\n", "relative path"),
            ("source: { hf: a/b, file: ../x.gguf }\n", "relative path"),
        ] {
            let err = Recipe::from_yaml(&minimal(yaml)).unwrap_err();
            assert!(err.to_string().contains(needle), "{yaml}: {err}");
        }
    }

    #[test]
    fn source_merge_refines_or_replaces() {
        let base = Source {
            hf: Some("org/base-GGUF".into()),
            revision: Some("v1".into()),
            include: vec!["*.json".into()],
            ..Default::default()
        };
        // refine: only `file` → inherits hf, revision, include
        let refined = Source::merged(
            Some(&base),
            Some(&Source {
                file: Some("m-Q4_K_M.gguf".into()),
                ..Default::default()
            }),
        )
        .unwrap();
        assert_eq!(refined.hf.as_deref(), Some("org/base-GGUF"));
        assert_eq!(refined.revision.as_deref(), Some("v1"));
        assert_eq!(refined.file.as_deref(), Some("m-Q4_K_M.gguf"));
        assert_eq!(refined.include, vec!["*.json".to_string()]);
        // replace: a new `hf` drops everything from the base
        let replaced = Source::merged(
            Some(&base),
            Some(&Source {
                hf: Some("org/other".into()),
                ..Default::default()
            }),
        )
        .unwrap();
        assert_eq!(replaced.hf.as_deref(), Some("org/other"));
        assert_eq!(replaced.revision, None);
        assert!(replaced.include.is_empty());
    }

    #[test]
    fn engine_merge_overrides_appends_and_overlays() {
        let r = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\n\
             engine:\n  context_length: 8192\n  max_concurrency: 8\n  vllm:\n    extra_args: [\"--a\"]\n    env: { X: \"1\", Y: \"1\" }\n\
             variants:\n  - name: v\n    engine:\n      context_length: 4096\n      vllm:\n        extra_args: [\"--b\"]\n        env: { Y: \"2\" }\n",
        ))
        .unwrap();
        let e = &r.resolved_variants().unwrap()[0].engine;
        assert_eq!(e.context_length, Some(4096));
        assert_eq!(e.max_concurrency, Some(8));
        assert_eq!(
            e.vllm.extra_args,
            vec!["--a".to_string(), "--b".to_string()]
        );
        assert_eq!(e.vllm.env.get("X").map(String::as_str), Some("1"));
        assert_eq!(e.vllm.env.get("Y").map(String::as_str), Some("2"));
    }

    #[test]
    fn context_length_cap_and_tp_consistency() {
        let err = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\ncontext_length: 4096\nengine: { context_length: 8192 }\n",
        ))
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("exceeds the model's context_length"),
            "{err}"
        );

        let err = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\nvariants:\n  - name: v\n    requires: { gpus: 1 }\n    engine: { tensor_parallel: 2 }\n",
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("does not match requires.gpus"),
            "{err}"
        );
    }

    #[test]
    fn compute_capability_accepts_string_or_number() {
        for (lit, want) in [
            ("\"8.0\"", "8.0"),
            ("8.0", "8.0"),
            ("8", "8.0"),
            ("8.9", "8.9"),
        ] {
            let r = Recipe::from_yaml(&minimal(&format!(
                "source: {{ hf: a/b }}\nvariants:\n  - name: v\n    requires: {{ min_compute_capability: {lit} }}\n"
            )))
            .unwrap_or_else(|e| panic!("{lit}: {e}"));
            assert_eq!(
                r.variants[0].requires.min_compute_capability.as_deref(),
                Some(want),
                "{lit}"
            );
        }
        let err = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\nvariants:\n  - name: v\n    requires: { min_compute_capability: \"sm80\" }\n",
        ))
        .unwrap_err();
        assert!(err.to_string().contains("major.minor"), "{err}");
    }

    #[test]
    fn engine_setting_validation() {
        for (yaml, needle) in [
            ("engine: { vllm: { min_version: \"v0.10\" } }\n", "semver"),
            (
                "engine: { vllm: { gpu_memory_utilization: 1.5 } }\n",
                "(0, 1]",
            ),
            (
                "engine: { vllm: { model_impl: fast } }\n",
                "auto, vllm or transformers",
            ),
            ("engine: { llamacpp: { min_release: \"6000\" } }\n", "b6000"),
            ("engine: { tensor_parallel: 0 }\n", ">= 1"),
        ] {
            let err =
                Recipe::from_yaml(&minimal(&format!("source: {{ hf: a/b }}\n{yaml}"))).unwrap_err();
            assert!(err.to_string().contains(needle), "{yaml}: {err}");
        }
    }

    #[test]
    fn tests_and_verified_are_validated() {
        let ok = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\n\
             tests:\n  - prompt: \"Reply with exactly: pong\"\n    expect_contains: pong\n  - prompt: weather?\n    expect_tool_call: true\n\
             verified:\n  - date: 2026-08-20\n    variant: default\n    hardware: 1x RTX 4090 24GB\n    backend_version: vllm 0.10.1\n    by: someone\n",
        ))
        .unwrap();
        assert_eq!(ok.tests.len(), 2);
        assert!(ok.tests[1].expect_tool_call);

        let err = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\nverified:\n  - date: 2026-08-20\n    variant: nope\n    hardware: x\n",
        ))
        .unwrap_err();
        assert!(err.to_string().contains("not a variant"), "{err}");

        let err = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\nverified:\n  - date: 20/08/2026\n    variant: default\n    hardware: x\n",
        ))
        .unwrap_err();
        assert!(err.to_string().contains("YYYY-MM-DD"), "{err}");

        let err = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\ntests:\n  - prompt: \"  \"\n",
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("prompt must not be empty"),
            "{err}"
        );
    }

    #[test]
    fn chat_template_needs_exactly_one_form() {
        let err = Recipe::from_yaml(&minimal("source: { hf: a/b }\nchat: { template: {} }\n"))
            .unwrap_err();
        assert!(
            err.to_string().contains("needs `file` or `inline`"),
            "{err}"
        );
        let ok = Recipe::from_yaml(&minimal(
            "source: { hf: a/b }\nchat:\n  template:\n    inline: |\n      {%- for m in messages %}{{ m.content }}{%- endfor %}\n",
        ))
        .unwrap();
        assert!(ok
            .chat
            .template
            .unwrap()
            .inline
            .unwrap()
            .contains("messages"));
    }

    #[test]
    fn json_input_is_accepted_and_identical() {
        let yaml = Recipe::from_path(&repo_root().join("recipes/qwen3-32b.yaml")).unwrap();
        let json = Recipe::from_json(&yaml.to_json_pretty()).unwrap();
        assert_eq!(yaml, json);
        let tmp = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        std::fs::write(tmp.path(), yaml.to_json_pretty()).unwrap();
        assert_eq!(Recipe::from_path(tmp.path()).unwrap(), yaml);
        let bad = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
        assert!(matches!(
            Recipe::from_path(bad.path()).unwrap_err(),
            RecipeError::Extension { .. }
        ));
    }

    #[test]
    fn json_schema_has_id_and_rejects_extra_properties() {
        let schema = Recipe::json_schema();
        assert_eq!(schema["$id"], SCHEMA_ID);
        assert_eq!(schema["additionalProperties"], false);
        assert!(schema["properties"]["variants"].is_object());
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "schema_version"));
    }

    /// `schemas/recipe-v1.json` is generated from these types; keep it current:
    /// `cargo test -p ohmygpu_core regenerate_schema -- --ignored`
    #[test]
    fn schema_file_is_current() {
        let want = serde_json::to_string_pretty(&Recipe::json_schema()).unwrap() + "\n";
        let have = std::fs::read_to_string(schema_path()).unwrap_or_default();
        assert!(
            have == want,
            "schemas/recipe-v1.json is stale; run: cargo test -p ohmygpu_core regenerate_schema -- --ignored"
        );
    }

    #[test]
    #[ignore]
    fn regenerate_schema() {
        let text = serde_json::to_string_pretty(&Recipe::json_schema()).unwrap() + "\n";
        std::fs::write(schema_path(), text).unwrap();
    }
}
