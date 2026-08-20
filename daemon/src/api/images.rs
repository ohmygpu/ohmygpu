//! Image inputs for vision models.
//!
//! Both protocol adapters produce `ContentPart::Image { url }` with either a
//! `data:` URL or an http(s) URL. Before a request is dispatched we (1) check
//! that the model can see at all, (2) validate inline images, and (3) fetch
//! remote ones into `data:` URLs — so runtime adapters only ever see inline
//! images and llama-server never fetches anything itself.

use base64::Engine;
use ohmygpu_core::registry::ModelCapabilities;
use ohmygpu_inference::{ContentPart, InferenceRequest, InputItem};
use reqwest::header::CONTENT_TYPE;

use crate::error::ApiError;

/// Largest image we accept, in decoded bytes.
pub const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const SUPPORTED: &[&str] = &[
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
];

fn bad(msg: impl Into<String>, param: &'static str) -> ApiError {
    ApiError::invalid(msg).with_param(param)
}

fn normalize_mime(m: &str) -> String {
    let m = m
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if m == "image/jpg" {
        "image/jpeg".to_string()
    } else {
        m
    }
}

/// The model must have a vision projector installed to accept images.
pub fn require_vision(
    caps: Option<ModelCapabilities>,
    model: &str,
    param: &'static str,
) -> Result<(), ApiError> {
    if caps.map(|c| c.vision).unwrap_or(false) {
        return Ok(());
    }
    Err(ApiError::unsupported(format!(
        "model '{model}' does not accept images (no vision projector installed); \
         pull a vision model such as qwen2.5-vl-3b-instruct, or add a projector with \
         `omg model pull <ref> --mmproj <file>`"
    ))
    .with_param(param))
}

/// Validate an inline `data:<image mime>;base64,<payload>` URL.
pub fn validate_data_url(url: &str, param: &'static str) -> Result<(), ApiError> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| bad("image url must be a data: URL or an http(s) URL", param))?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| bad("malformed data: URL (no ',' separator)", param))?;
    let Some(mime) = meta.strip_suffix(";base64") else {
        return Err(bad("data: URL images must be base64-encoded", param));
    };
    let mime = normalize_mime(mime);
    if !SUPPORTED.contains(&mime.as_str()) {
        return Err(ApiError::unsupported(format!(
            "image type '{mime}' (supported: png, jpeg, gif, webp, bmp)"
        ))
        .with_param(param));
    }
    if payload.is_empty() {
        return Err(bad("empty image", param));
    }
    if payload.len() / 4 * 3 > MAX_IMAGE_BYTES {
        return Err(bad(
            format!(
                "image is too large (limit {} MB)",
                MAX_IMAGE_BYTES / (1024 * 1024)
            ),
            param,
        ));
    }
    Ok(())
}

/// Recognise common image formats from their first bytes.
pub fn sniff_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

async fn fetch_to_data_url(
    client: &reqwest::Client,
    url: &str,
    param: &'static str,
) -> Result<String, ApiError> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| bad(format!("could not fetch image {url}: {e}"), param))?;
    if !resp.status().is_success() {
        return Err(bad(
            format!("could not fetch image {url}: HTTP {}", resp.status()),
            param,
        ));
    }
    if let Some(len) = resp.content_length() {
        if len as usize > MAX_IMAGE_BYTES {
            return Err(bad(
                format!(
                    "image is too large (limit {} MB)",
                    MAX_IMAGE_BYTES / (1024 * 1024)
                ),
                param,
            ));
        }
    }
    let declared = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(normalize_mime);
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| bad(format!("could not read image {url}: {e}"), param))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(bad(
            format!(
                "image is too large (limit {} MB)",
                MAX_IMAGE_BYTES / (1024 * 1024)
            ),
            param,
        ));
    }
    let mime = match sniff_mime(&bytes) {
        Some(m) => m.to_string(),
        None => declared
            .filter(|d| SUPPORTED.contains(&d.as_str()))
            .ok_or_else(|| bad(format!("{url} is not an image we recognise"), param))?,
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{mime};base64,{b64}"))
}

/// Validate inline images and inline remote ones, in place.
pub async fn resolve_images(
    req: &mut InferenceRequest,
    client: &reqwest::Client,
    param: &'static str,
) -> Result<(), ApiError> {
    for item in &mut req.input {
        let InputItem::Message { content, .. } = item else {
            continue;
        };
        for part in content.iter_mut() {
            let ContentPart::Image { url } = part else {
                continue;
            };
            if url.starts_with("data:") {
                validate_data_url(url, param)?;
            } else if url.starts_with("http://") || url.starts_with("https://") {
                let inlined = fetch_to_data_url(client, url, param).await?;
                *url = inlined;
            } else {
                return Err(bad(
                    "image url must be a data: URL or an http(s) URL",
                    param,
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_url_rules() {
        validate_data_url("data:image/png;base64,iVBORw0KGgo=", "input").unwrap();
        validate_data_url("data:image/jpg;base64,/9j/4AAQ", "input").unwrap();
        for (bad_url, needle) in [
            ("data:text/plain;base64,aGk=", "image type"),
            ("data:image/png,abc", "base64"),
            ("data:image/png;base64,", "empty"),
            ("ftp://x/y.png", "data: URL or an http"),
            ("data:image/pngabc", "malformed"),
        ] {
            let e = validate_data_url(bad_url, "input").unwrap_err();
            assert!(
                e.body.message.contains(needle),
                "{bad_url}: {}",
                e.body.message
            );
        }
        let huge = format!(
            "data:image/png;base64,{}",
            "A".repeat(MAX_IMAGE_BYTES / 3 * 4 + 8)
        );
        assert!(validate_data_url(&huge, "input")
            .unwrap_err()
            .body
            .message
            .contains("too large"));
    }

    #[test]
    fn sniffing() {
        assert_eq!(sniff_mime(b"\x89PNG\r\n\x1a\nxxxx"), Some("image/png"));
        assert_eq!(sniff_mime(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("image/jpeg"));
        assert_eq!(sniff_mime(b"GIF89a..."), Some("image/gif"));
        assert_eq!(
            sniff_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 "),
            Some("image/webp")
        );
        assert_eq!(sniff_mime(b"hello"), None);
    }

    #[test]
    fn vision_gate() {
        assert!(require_vision(None, "m", "input").is_err());
        assert!(require_vision(
            Some(ModelCapabilities {
                tools: true,
                vision: false
            }),
            "m",
            "input"
        )
        .is_err());
        require_vision(
            Some(ModelCapabilities {
                tools: false,
                vision: true,
            }),
            "m",
            "input",
        )
        .unwrap();
    }
}
