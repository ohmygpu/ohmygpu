//! `POST /v1/audio/transcriptions` — OpenAI-compatible speech to text.
//!
//! Multipart form: `file` (required), `model` (required), `language`, `prompt`,
//! `response_format` (`json` default | `text` | `verbose_json` | `srt` | `vtt`),
//! `temperature`, `timestamp_granularities[]` (`segment` only). The upload is
//! decoded and resampled here (see `crate::audio`); backends get 16 kHz PCM.

use axum::extract::multipart::{Multipart, MultipartRejection};
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use axum::Json;
use ohmygpu_inference::{TranscriptionRequest, TranscriptionResponse};
use serde_json::json;

use crate::audio;
use crate::error::ApiError;
use crate::state::SharedState;

/// Largest upload we accept (decoded later; the body limit on the route is
/// slightly larger to leave room for the multipart framing).
pub const MAX_UPLOAD_BYTES: usize = 50 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

impl ResponseFormat {
    fn parse(s: &str) -> Result<Self, ApiError> {
        Ok(match s {
            "json" => Self::Json,
            "text" => Self::Text,
            "verbose_json" => Self::VerboseJson,
            "srt" => Self::Srt,
            "vtt" => Self::Vtt,
            other => {
                return Err(ApiError::invalid(format!(
                    "response_format '{other}' (json, text, verbose_json, srt, vtt)"
                ))
                .with_param("response_format"))
            }
        })
    }
}

#[derive(Default)]
struct Form {
    file: Option<(Vec<u8>, Option<String>, Option<String>)>,
    model: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
    temperature: Option<f32>,
    granularities: Vec<String>,
    stream: Option<bool>,
}

async fn read_form(mut mp: Multipart) -> Result<Form, ApiError> {
    let mut form = Form::default();
    while let Some(field) = mp
        .next_field()
        .await
        .map_err(|e| ApiError::invalid(format!("malformed multipart body: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let file_name = field.file_name().map(str::to_string);
                let content_type = field.content_type().map(str::to_string);
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::invalid(format!("could not read `file`: {e}")))?;
                if bytes.len() > MAX_UPLOAD_BYTES {
                    return Err(ApiError::invalid(format!(
                        "`file` is too large (limit {} MB)",
                        MAX_UPLOAD_BYTES / (1024 * 1024)
                    ))
                    .with_param("file"));
                }
                form.file = Some((bytes.to_vec(), file_name, content_type));
            }
            other => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| ApiError::invalid(format!("could not read `{other}`: {e}")))?;
                match other {
                    "model" => form.model = Some(text),
                    "language" => form.language = Some(text),
                    "prompt" => form.prompt = Some(text),
                    "response_format" => form.response_format = Some(text),
                    "temperature" => {
                        form.temperature = Some(text.trim().parse::<f32>().map_err(|_| {
                            ApiError::invalid("temperature must be a number")
                                .with_param("temperature")
                        })?)
                    }
                    "timestamp_granularities[]" | "timestamp_granularities" => {
                        form.granularities.push(text)
                    }
                    "stream" => form.stream = Some(matches!(text.trim(), "true" | "1")),
                    // Ignored, like OpenAI clients expect of optional knobs we do not have.
                    "include[]" | "include" | "chunking_strategy" => {}
                    unknown => {
                        return Err(ApiError::unsupported(format!("form field '{unknown}'"))
                            .with_param(unknown.to_string()))
                    }
                }
            }
        }
    }
    Ok(form)
}

pub async fn transcriptions(
    State(state): State<SharedState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    let multipart = multipart.map_err(|e| {
        ApiError::invalid(format!(
            "expected multipart/form-data with a `file` and a `model` field: {e}"
        ))
    })?;
    let form = read_form(multipart).await?;
    let model = form
        .model
        .clone()
        .filter(|m| !m.trim().is_empty())
        .ok_or_else(|| ApiError::invalid("`model` is required").with_param("model"))?;
    let (bytes, file_name, content_type) = form
        .file
        .ok_or_else(|| ApiError::invalid("`file` is required").with_param("file"))?;
    if bytes.is_empty() {
        return Err(ApiError::invalid("`file` is empty").with_param("file"));
    }
    let format = ResponseFormat::parse(form.response_format.as_deref().unwrap_or("json"))?;
    if form.stream == Some(true) {
        return Err(ApiError::unsupported("streaming transcriptions").with_param("stream"));
    }
    for g in &form.granularities {
        if g != "segment" {
            return Err(ApiError::unsupported(format!(
                "timestamp_granularities '{g}' (only segment)"
            ))
            .with_param("timestamp_granularities"));
        }
    }
    let language = form
        .language
        .map(|l| l.trim().to_ascii_lowercase())
        .filter(|l| !l.is_empty() && l != "auto");

    // Find the model first so "not found / not running" beat decode errors.
    let instance = state.manager.instance_for(&model).await?;

    // Decode on a blocking thread (CPU-bound, can be tens of MB).
    let audio = tokio::task::spawn_blocking(move || {
        audio::decode(bytes, file_name.as_deref(), content_type.as_deref())
            .map(|d| audio::to_audio_input(&d))
    })
    .await
    .map_err(|e| ApiError::invalid(format!("audio decoding failed: {e}")))?
    .map_err(|e| ApiError::invalid(e.to_string()).with_param("file"))?;

    let req = TranscriptionRequest {
        model: model.clone(),
        audio,
        language,
        prompt: form.prompt.filter(|p| !p.trim().is_empty()),
        temperature: form.temperature,
        translate: false,
    };
    let resp = instance.transcribe(req).await?;
    Ok(render(format, &resp))
}

fn render(format: ResponseFormat, r: &TranscriptionResponse) -> Response {
    match format {
        ResponseFormat::Json => Json(json!({ "text": r.text })).into_response(),
        ResponseFormat::VerboseJson => Json(json!({
            "task": "transcribe",
            "language": r.language.clone().unwrap_or_else(|| "unknown".into()),
            "duration": r.duration_secs,
            "text": r.text,
            "segments": r.segments.iter().map(|s| json!({
                "id": s.id,
                "start": s.start_secs,
                "end": s.end_secs,
                "text": s.text,
            })).collect::<Vec<_>>(),
        }))
        .into_response(),
        ResponseFormat::Text => (
            [(CONTENT_TYPE, "text/plain; charset=utf-8")],
            format!("{}\n", r.text),
        )
            .into_response(),
        ResponseFormat::Srt => {
            ([(CONTENT_TYPE, "text/plain; charset=utf-8")], srt(r)).into_response()
        }
        ResponseFormat::Vtt => {
            ([(CONTENT_TYPE, "text/vtt; charset=utf-8")], vtt(r)).into_response()
        }
    }
}

fn stamp(secs: f32, sep: char) -> String {
    let ms = (secs.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{sep}{:03}",
        ms / 3_600_000,
        (ms / 60_000) % 60,
        (ms / 1000) % 60,
        ms % 1000
    )
}

fn segments_or_whole(r: &TranscriptionResponse) -> Vec<(f32, f32, &str)> {
    if r.segments.is_empty() {
        vec![(0.0, r.duration_secs, r.text.as_str())]
    } else {
        r.segments
            .iter()
            .map(|s| (s.start_secs, s.end_secs, s.text.as_str()))
            .collect()
    }
}

fn srt(r: &TranscriptionResponse) -> String {
    let mut out = String::new();
    for (i, (start, end, text)) in segments_or_whole(r).into_iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            stamp(start, ','),
            stamp(end, ','),
            text.trim()
        ));
    }
    out
}

fn vtt(r: &TranscriptionResponse) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for (start, end, text) in segments_or_whole(r) {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            stamp(start, '.'),
            stamp(end, '.'),
            text.trim()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ohmygpu_inference::TranscriptionSegment;

    fn resp() -> TranscriptionResponse {
        TranscriptionResponse {
            model: "w".into(),
            text: "hello world".into(),
            language: Some("english".into()),
            duration_secs: 3.5,
            segments: vec![
                TranscriptionSegment {
                    id: 0,
                    start_secs: 0.0,
                    end_secs: 1.25,
                    text: "hello".into(),
                },
                TranscriptionSegment {
                    id: 1,
                    start_secs: 1.25,
                    end_secs: 3.5,
                    text: "world".into(),
                },
            ],
        }
    }

    #[test]
    fn srt_and_vtt_timestamps() {
        assert_eq!(stamp(3661.5, ','), "01:01:01,500");
        let s = srt(&resp());
        assert!(
            s.starts_with("1\n00:00:00,000 --> 00:00:01,250\nhello\n\n2\n"),
            "{s}"
        );
        let v = vtt(&resp());
        assert!(
            v.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nhello\n\n"),
            "{v}"
        );
    }

    #[test]
    fn response_format_names() {
        assert_eq!(
            ResponseFormat::parse("verbose_json").unwrap(),
            ResponseFormat::VerboseJson
        );
        assert!(ResponseFormat::parse("xml").is_err());
    }
}
