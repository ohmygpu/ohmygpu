//! Decode uploaded audio into what speech backends want: mono f32 PCM at
//! 16 kHz. Containers/codecs come from symphonia (wav, mp3, m4a/aac, flac,
//! ogg-vorbis, mkv/webm containers); resampling is a windowed-sinc low-pass
//! plus linear interpolation — plenty for speech recognition, no ffmpeg.

use std::io::Cursor;

use ohmygpu_inference::AudioInput;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// What whisper wants.
pub const TARGET_RATE: u32 = 16_000;

/// Formats we decode, for error messages.
pub const SUPPORTED: &str = "wav, mp3, m4a/aac, flac, ogg (vorbis)";

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("unrecognised audio format{0}; supported: {SUPPORTED}")]
    Unrecognised(String),
    #[error("unsupported audio codec ({0}); supported: {SUPPORTED} — opus (webm/ogg-opus) is not supported yet")]
    UnsupportedCodec(String),
    #[error("could not decode audio: {0}")]
    Decode(String),
    #[error("audio contains no samples")]
    Empty,
}

/// Interleaved PCM as decoded.
#[derive(Debug, Clone)]
pub struct Decoded {
    pub sample_rate: u32,
    pub channels: usize,
    pub samples: Vec<f32>,
}

impl Decoded {
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 || self.channels == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.channels as f32 / self.sample_rate as f32
        }
    }
}

/// Decode any supported container/codec to interleaved f32 PCM. `file_name`
/// and `content_type` are only hints for the probe.
pub fn decode(
    bytes: Vec<u8>,
    file_name: Option<&str>,
    content_type: Option<&str>,
) -> Result<Decoded, AudioError> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = file_name
        .and_then(|n| n.rsplit('.').next())
        .filter(|e| !e.is_empty() && e.len() <= 5)
    {
        hint.with_extension(ext);
    }
    if let Some(ct) = content_type {
        let ct = ct.split(';').next().unwrap_or("").trim();
        if !ct.is_empty() && ct != "application/octet-stream" {
            hint.mime_type(ct);
        }
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| match e {
            SymError::Unsupported(_) => AudioError::Unrecognised(String::new()),
            SymError::IoError(_) => AudioError::Unrecognised(" (truncated or empty file)".into()),
            other => AudioError::Unrecognised(format!(" ({other})")),
        })?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::Unrecognised(" (no audio track)".into()))?;
    let track_id = track.id;
    let params = track.codec_params.clone();
    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|e| match e {
            SymError::Unsupported(what) => AudioError::UnsupportedCodec(what.to_string()),
            other => AudioError::Decode(other.to_string()),
        })?;

    let mut sample_rate = params.sample_rate.unwrap_or(0);
    let mut channels = params.channels.map(|c| c.count()).unwrap_or(0);
    let mut samples: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => {
                if samples.is_empty() {
                    return Err(AudioError::Decode(e.to_string()));
                }
                break;
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = *buf.spec();
                sample_rate = spec.rate;
                channels = spec.channels.count();
                let mut sbuf = SampleBuffer::<f32>::new(buf.capacity() as u64, spec);
                sbuf.copy_interleaved_ref(buf);
                samples.extend_from_slice(sbuf.samples());
            }
            // A damaged frame: skip it, like every player does.
            Err(SymError::DecodeError(e)) => tracing::debug!("skipping undecodable packet: {e}"),
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(AudioError::Decode(e.to_string())),
        }
    }
    if samples.is_empty() || sample_rate == 0 || channels == 0 {
        return Err(AudioError::Empty);
    }
    Ok(Decoded {
        sample_rate,
        channels,
        samples,
    })
}

/// Interleaved → mono by averaging channels.
pub fn to_mono(d: &Decoded) -> Vec<f32> {
    if d.channels <= 1 {
        return d.samples.clone();
    }
    let n = d.channels;
    d.samples
        .chunks(n)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// Resample mono PCM from `from` Hz to `to` Hz. Windowed-sinc (Hann) low-pass
/// at the lower Nyquist, evaluated at each output position. Identity when the
/// rates match.
pub fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() || from == 0 || to == 0 {
        return samples.to_vec();
    }
    let ratio = from as f64 / to as f64; // input samples per output sample
                                         // Cut-off (cycles per input sample) a little under the lower Nyquist.
    let fc = (0.5f64).min(0.5 / ratio) * 0.92;
    let half = if ratio > 1.0 {
        (12.0 * ratio).ceil()
    } else {
        12.0
    };
    let out_len = (samples.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    let n = samples.len() as isize;
    for i in 0..out_len {
        let center = i as f64 * ratio;
        let lo = ((center - half).ceil() as isize).max(0);
        let hi = ((center + half).floor() as isize).min(n - 1);
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for j in lo..=hi {
            let d = j as f64 - center;
            let x = std::f64::consts::PI * d;
            let sinc = if d.abs() < 1e-9 {
                1.0
            } else {
                (2.0 * fc * x).sin() / x
            };
            let window = 0.5 * (1.0 + (std::f64::consts::PI * d / half).cos());
            let w = sinc * window;
            acc += samples[j as usize] as f64 * w;
            wsum += w;
        }
        out.push(if wsum.abs() > 1e-12 {
            (acc / wsum) as f32
        } else {
            0.0
        });
    }
    out
}

/// Decoded audio → what the backends take (mono, 16 kHz).
pub fn to_audio_input(d: &Decoded) -> AudioInput {
    let mono = to_mono(d);
    let samples = resample(&mono, d.sample_rate, TARGET_RATE);
    AudioInput {
        sample_rate: TARGET_RATE,
        samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16-bit PCM WAV, `secs` of a `freq` Hz sine per channel.
    fn wav(rate: u32, channels: u16, secs: f32, freq: f32) -> Vec<u8> {
        let frames = (rate as f32 * secs) as usize;
        let data_len = (frames * channels as usize * 2) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        out.extend_from_slice(&(channels * 2).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for i in 0..frames {
            let v = (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin();
            let s = (v * 20000.0) as i16;
            for _ in 0..channels {
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
        out
    }

    fn fixture(name: &str) -> Vec<u8> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/audio")
            .join(name);
        std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    }

    #[test]
    fn wav_stereo_44k_becomes_mono_16k() {
        let d = decode(wav(44_100, 2, 0.5, 440.0), Some("a.wav"), None).unwrap();
        assert_eq!(d.sample_rate, 44_100);
        assert_eq!(d.channels, 2);
        assert!((d.duration_secs() - 0.5).abs() < 0.01);
        let a = to_audio_input(&d);
        assert_eq!(a.sample_rate, 16_000);
        assert!(
            (a.duration_secs() - 0.5).abs() < 0.01,
            "{}",
            a.duration_secs()
        );
        // the tone survives resampling: peak amplitude stays near the original
        let peak = a.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.5 && peak <= 1.0, "peak {peak}");
    }

    #[test]
    fn resample_identity_and_upsample() {
        let x: Vec<f32> = (0..160).map(|i| (i as f32 / 10.0).sin()).collect();
        assert_eq!(resample(&x, 16_000, 16_000), x);
        let up = resample(&x, 8_000, 16_000);
        assert_eq!(up.len(), 320);
        assert!((up[100] - x[50]).abs() < 0.05);
    }

    #[test]
    fn compressed_fixtures_decode_to_about_one_second() {
        for (name, ct) in [
            ("tone.mp3", "audio/mpeg"),
            ("tone.m4a", "audio/mp4"),
            ("tone.flac", "audio/flac"),
            ("tone.ogg", "audio/ogg"),
            ("tone-16k.wav", "audio/wav"),
        ] {
            let d = decode(fixture(name), Some(name), Some(ct))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let a = to_audio_input(&d);
            let expect = if name == "tone-16k.wav" { 0.5 } else { 1.0 };
            assert!(
                (a.duration_secs() - expect).abs() < 0.15,
                "{name}: {}s (decoded {} Hz × {} ch)",
                a.duration_secs(),
                d.sample_rate,
                d.channels
            );
            let peak = a.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            assert!(peak > 0.3, "{name}: peak {peak}");
        }
    }

    #[test]
    fn opus_and_garbage_are_clear_errors() {
        let e = decode(fixture("tone.webm"), Some("tone.webm"), Some("audio/webm")).unwrap_err();
        assert!(matches!(e, AudioError::UnsupportedCodec(_)), "{e}");
        assert!(e.to_string().contains("opus"), "{e}");
        let e = decode(b"definitely not audio".to_vec(), Some("x.txt"), None).unwrap_err();
        assert!(matches!(e, AudioError::Unrecognised(_)), "{e}");
    }
}
