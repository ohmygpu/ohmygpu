//! Translation between OhMyGPU's transcription types and whisper-server's
//! `/inference` endpoint (multipart in, `verbose_json` out). This is the only
//! place that knows whisper-server's wire format.

use ohmygpu_inference::{
    AudioInput, InferenceError, TranscriptionRequest, TranscriptionResponse, TranscriptionSegment,
};
use reqwest::multipart::{Form, Part};
use serde_json::Value;

/// 16-bit PCM mono WAV for the given audio (whisper-server reads WAV only).
pub fn wav_bytes(audio: &AudioInput) -> Vec<u8> {
    let channels: u16 = 1;
    let bits: u16 = 16;
    let byte_rate = audio.sample_rate * u32::from(channels) * u32::from(bits / 8);
    let block_align = channels * (bits / 8);
    let data_len = (audio.samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&audio.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in &audio.samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// The multipart form for `POST /inference`.
pub fn build_form(req: &TranscriptionRequest) -> Form {
    let wav = wav_bytes(&req.audio);
    let mut form = Form::new()
        .part(
            "file",
            Part::bytes(wav)
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .expect("static mime"),
        )
        .text("response_format", "verbose_json")
        .text("no_language_probabilities", "true")
        .text(
            "language",
            req.language.clone().unwrap_or_else(|| "auto".to_string()),
        )
        .text("translate", if req.translate { "true" } else { "false" });
    if let Some(t) = req.temperature {
        form = form.text("temperature", t.to_string());
    }
    if let Some(p) = &req.prompt {
        if !p.is_empty() {
            form = form.text("prompt", p.clone());
        }
    }
    form
}

/// whisper-server's `verbose_json` → internal response.
pub fn parse_response(
    model: &str,
    audio: &AudioInput,
    body: &Value,
) -> Result<TranscriptionResponse, InferenceError> {
    let text = body
        .get("text")
        .and_then(|t| t.as_str())
        .ok_or_else(|| InferenceError::Backend("whisper-server response has no `text`".into()))?
        .trim()
        .to_string();
    let language = body
        .get("language")
        .and_then(|l| l.as_str())
        .filter(|l| !l.is_empty() && *l != "auto")
        .map(str::to_string);
    let duration_secs = body
        .get("duration")
        .and_then(|d| d.as_f64())
        .map(|d| d as f32)
        .unwrap_or_else(|| audio.duration_secs());
    let mut segments = Vec::new();
    if let Some(arr) = body.get("segments").and_then(|s| s.as_array()) {
        for (i, seg) in arr.iter().enumerate() {
            let seg_text = seg
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            segments.push(TranscriptionSegment {
                id: seg.get("id").and_then(|v| v.as_u64()).unwrap_or(i as u64) as u32,
                start_secs: seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                end_secs: seg.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                text: seg_text,
            });
        }
    }
    Ok(TranscriptionResponse {
        model: model.to_string(),
        text,
        language,
        duration_secs,
        segments,
    })
}

/// Error text from a whisper-server error body (`{"error": "..."}` or plain text).
pub fn error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
        .unwrap_or_else(|| body.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wav_header_is_right() {
        let audio = AudioInput {
            sample_rate: 16000,
            samples: vec![0.0, 0.5, -0.5, 1.0],
        };
        let wav = wav_bytes(&audio);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16000);
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 8);
        assert_eq!(wav.len(), 44 + 8);
        assert_eq!(i16::from_le_bytes(wav[46..48].try_into().unwrap()), 16384);
        assert_eq!(i16::from_le_bytes(wav[50..52].try_into().unwrap()), 32767);
    }

    #[test]
    fn parses_verbose_json() {
        let audio = AudioInput {
            sample_rate: 16000,
            samples: vec![0.0; 16000],
        };
        let body = json!({
            "task": "transcribe", "language": "english", "duration": 11.0,
            "text": " And so my fellow Americans, ask not what your country can do for you.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 8.0, "text": " And so my fellow Americans,", "tokens": [1,2]},
                {"id": 1, "start": 8.0, "end": 11.0, "text": " ask not what your country can do for you."}
            ]
        });
        let r = parse_response("whisper-tiny", &audio, &body).unwrap();
        assert!(r.text.starts_with("And so my fellow Americans"));
        assert_eq!(r.language.as_deref(), Some("english"));
        assert_eq!(r.duration_secs, 11.0);
        assert_eq!(r.segments.len(), 2);
        assert_eq!(r.segments[1].start_secs, 8.0);
        assert_eq!(
            r.segments[1].text,
            "ask not what your country can do for you."
        );
        // duration falls back to the audio length
        let r = parse_response("m", &audio, &json!({"text": "hi"})).unwrap();
        assert_eq!(r.duration_secs, 1.0);
        assert!(parse_response("m", &audio, &json!({"nope": 1})).is_err());
    }

    #[test]
    fn error_bodies() {
        assert_eq!(
            error_message(r#"{"error":"no 'file' field in the request"}"#),
            "no 'file' field in the request"
        );
        assert_eq!(error_message("boom"), "boom");
    }
}
