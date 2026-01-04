pub mod registry;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub source: ModelSource,
    pub model_type: ModelType,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub files: Vec<String>,
    pub downloaded_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSource {
    HuggingFace { repo_id: String },
    GitHub { repo: String, release: Option<String> },
    Local,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    LLM,
    Embedding,
    ImageGeneration,
    ImageClassification,
    AudioTranscription,
    AudioGeneration,
    Unknown,
}

impl ModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelType::LLM => "LLM",
            ModelType::Embedding => "Embedding",
            ModelType::ImageGeneration => "Image Gen",
            ModelType::ImageClassification => "Image Class",
            ModelType::AudioTranscription => "Audio STT",
            ModelType::AudioGeneration => "Audio Gen",
            ModelType::Unknown => "Unknown",
        }
    }

    pub fn from_pipeline_tag(tag: &str) -> Self {
        match tag {
            "text-generation" | "text2text-generation" => ModelType::LLM,
            "feature-extraction" | "sentence-similarity" => ModelType::Embedding,
            "text-to-image" | "image-to-image" => ModelType::ImageGeneration,
            "image-classification" | "object-detection" => ModelType::ImageClassification,
            "automatic-speech-recognition" => ModelType::AudioTranscription,
            "text-to-audio" | "text-to-speech" => ModelType::AudioGeneration,
            _ => ModelType::Unknown,
        }
    }
}
