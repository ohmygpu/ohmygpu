//! Minimal GGUF header reader — just enough metadata to *describe* a model
//! file without loading it: the architecture and the native context window.
//!
//! The reader walks the key/value header sequentially and stops as soon as it
//! has what it needs, which in practice is the first few KB of the file (the
//! large tokenizer arrays come later). It never panics on bad input: anything
//! malformed is an `io::Error` with `InvalidData` / `UnexpectedEof`.
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

const MAGIC: &[u8; 4] = b"GGUF";
/// Longer strings are treated as corruption (real ones top out at a few KB —
/// chat templates).
const MAX_STRING: u64 = 64 * 1024 * 1024;
/// More key/value pairs than this is not a model file.
const MAX_KV: u64 = 1 << 20;

// GGUF metadata value types.
const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

/// What we read from a GGUF header.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GgufMetadata {
    /// GGUF format version (2 or 3).
    pub version: u32,
    /// `general.architecture`, e.g. `llama`, `qwen2`, `gemma3`.
    pub architecture: Option<String>,
    /// `<architecture>.context_length` — the context window the model was
    /// trained with (tokens). `None` when the file does not record it.
    pub context_length: Option<u32>,
}

/// Read the metadata of the GGUF file at `path`.
pub fn read_metadata(path: impl AsRef<Path>) -> io::Result<GgufMetadata> {
    let file = File::open(path)?;
    read_metadata_from(&mut BufReader::new(file))
}

/// Native context window of the GGUF at `path`, or `None` when the file
/// cannot be read, is not a GGUF, or does not record one. Never fails: this
/// is decoration for an installed model, not a precondition.
pub fn context_length(path: &Path) -> Option<u32> {
    match read_metadata(path) {
        Ok(meta) => meta.context_length,
        Err(e) => {
            tracing::debug!("no GGUF metadata for {}: {e}", path.display());
            None
        }
    }
}

/// Read the metadata from the start of a GGUF byte stream.
pub fn read_metadata_from<R: Read>(r: &mut R) -> io::Result<GgufMetadata> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("not a GGUF file (bad magic)"));
    }
    let version = read_u32(r)?;
    if !(2..=3).contains(&version) {
        return Err(invalid(format!("unsupported GGUF version {version}")));
    }
    let _n_tensors = read_u64(r)?;
    let n_kv = read_u64(r)?;
    if n_kv > MAX_KV {
        return Err(invalid(format!("implausible metadata count {n_kv}")));
    }

    let mut meta = GgufMetadata {
        version,
        ..Default::default()
    };
    // `<arch>.context_length` normally follows `general.architecture`, but the
    // format does not require any order: remember every candidate and pick the
    // one that matches the architecture once both are known.
    let mut ctx_candidates: Vec<(String, u64)> = Vec::new();
    for _ in 0..n_kv {
        let key = read_string(r)?;
        let vtype = read_u32(r)?;
        if key == "general.architecture" {
            if vtype == T_STRING {
                meta.architecture = Some(read_string(r)?);
            } else {
                skip_value(r, vtype)?;
            }
        } else if key.ends_with(".context_length") {
            if let Some(v) = read_integer(r, vtype)? {
                ctx_candidates.push((key, v));
            }
        } else {
            skip_value(r, vtype)?;
        }
        if let Some(arch) = &meta.architecture {
            let want = format!("{arch}.context_length");
            if let Some((_, v)) = ctx_candidates.iter().find(|(k, _)| *k == want) {
                meta.context_length = u32::try_from(*v).ok();
                break;
            }
        }
    }
    Ok(meta)
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)?;
    if len > MAX_STRING {
        return Err(invalid(format!("implausible string length {len}")));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Read a scalar integer value of type `vtype` as `u64`; `None` (value
/// skipped) when the type is not an integer or the value is negative.
fn read_integer<R: Read>(r: &mut R, vtype: u32) -> io::Result<Option<u64>> {
    let size = match scalar_size(vtype) {
        Some(s) => s,
        None => {
            skip_value(r, vtype)?;
            return Ok(None);
        }
    };
    let mut b = [0u8; 8];
    r.read_exact(&mut b[..size])?;
    let v = match vtype {
        T_UINT8 => Some(b[0] as u64),
        T_INT8 => u64::try_from(b[0] as i8).ok(),
        T_UINT16 => Some(u16::from_le_bytes([b[0], b[1]]) as u64),
        T_INT16 => u64::try_from(i16::from_le_bytes([b[0], b[1]])).ok(),
        T_UINT32 => Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64),
        T_INT32 => u64::try_from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok(),
        T_UINT64 => Some(u64::from_le_bytes(b)),
        T_INT64 => u64::try_from(i64::from_le_bytes(b)).ok(),
        _ => None, // floats / bool: not a context length
    };
    Ok(v)
}

/// Byte size of a fixed-size scalar type; `None` for strings and arrays.
fn scalar_size(vtype: u32) -> Option<usize> {
    match vtype {
        T_UINT8 | T_INT8 | T_BOOL => Some(1),
        T_UINT16 | T_INT16 => Some(2),
        T_UINT32 | T_INT32 | T_FLOAT32 => Some(4),
        T_UINT64 | T_INT64 | T_FLOAT64 => Some(8),
        _ => None,
    }
}

fn skip_bytes<R: Read>(r: &mut R, n: u64) -> io::Result<()> {
    let copied = io::copy(&mut r.by_ref().take(n), &mut io::sink())?;
    if copied != n {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "truncated GGUF header",
        ));
    }
    Ok(())
}

fn skip_value<R: Read>(r: &mut R, vtype: u32) -> io::Result<()> {
    if let Some(size) = scalar_size(vtype) {
        return skip_bytes(r, size as u64);
    }
    match vtype {
        T_STRING => {
            let len = read_u64(r)?;
            if len > MAX_STRING {
                return Err(invalid(format!("implausible string length {len}")));
            }
            skip_bytes(r, len)
        }
        T_ARRAY => {
            let elem_type = read_u32(r)?;
            let n = read_u64(r)?;
            if let Some(size) = scalar_size(elem_type) {
                let bytes = n
                    .checked_mul(size as u64)
                    .ok_or_else(|| invalid("implausible array length"))?;
                return skip_bytes(r, bytes);
            }
            match elem_type {
                // Strings (tokenizer vocab) and nested arrays: element by element.
                T_STRING | T_ARRAY => {
                    for _ in 0..n {
                        skip_value(r, elem_type)?;
                    }
                    Ok(())
                }
                other => Err(invalid(format!("unknown GGUF array element type {other}"))),
            }
        }
        other => Err(invalid(format!("unknown GGUF value type {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Value kinds the test builder can emit.
    enum Val<'a> {
        U32(u32),
        U64(u64),
        I32(i32),
        F32(f32),
        Bool(bool),
        Str(&'a str),
        StrArr(&'a [&'a str]),
        U32Arr(&'a [u32]),
        /// Array of string arrays (nested).
        NestedStrArr(&'a [&'a [&'a str]]),
    }

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn put_val(out: &mut Vec<u8>, v: &Val) {
        match v {
            Val::U32(x) => {
                out.extend_from_slice(&T_UINT32.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            Val::U64(x) => {
                out.extend_from_slice(&T_UINT64.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            Val::I32(x) => {
                out.extend_from_slice(&T_INT32.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            Val::F32(x) => {
                out.extend_from_slice(&T_FLOAT32.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            Val::Bool(x) => {
                out.extend_from_slice(&T_BOOL.to_le_bytes());
                out.push(*x as u8);
            }
            Val::Str(s) => {
                out.extend_from_slice(&T_STRING.to_le_bytes());
                put_str(out, s);
            }
            Val::StrArr(items) => {
                out.extend_from_slice(&T_ARRAY.to_le_bytes());
                out.extend_from_slice(&T_STRING.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for s in *items {
                    put_str(out, s);
                }
            }
            Val::U32Arr(items) => {
                out.extend_from_slice(&T_ARRAY.to_le_bytes());
                out.extend_from_slice(&T_UINT32.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for x in *items {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            Val::NestedStrArr(rows) => {
                out.extend_from_slice(&T_ARRAY.to_le_bytes());
                out.extend_from_slice(&T_ARRAY.to_le_bytes());
                out.extend_from_slice(&(rows.len() as u64).to_le_bytes());
                for row in *rows {
                    // each element: its own (type, len, items)
                    out.extend_from_slice(&T_STRING.to_le_bytes());
                    out.extend_from_slice(&(row.len() as u64).to_le_bytes());
                    for s in *row {
                        put_str(out, s);
                    }
                }
            }
        }
    }

    /// A GGUF v3 header with the given key/values and no tensors.
    fn gguf(kvs: &[(&str, Val)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // tensors
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            put_str(&mut out, k);
            put_val(&mut out, v);
        }
        out
    }

    fn parse(bytes: &[u8]) -> io::Result<GgufMetadata> {
        read_metadata_from(&mut io::Cursor::new(bytes))
    }

    #[test]
    fn reads_architecture_and_context_length() {
        let bytes = gguf(&[
            ("general.architecture", Val::Str("qwen2")),
            ("general.name", Val::Str("Qwen2.5 0.5B Instruct")),
            ("qwen2.block_count", Val::U32(24)),
            ("qwen2.context_length", Val::U32(32768)),
            ("qwen2.embedding_length", Val::U32(896)),
            (
                "tokenizer.ggml.tokens",
                Val::StrArr(&["<s>", "hello", "world"]),
            ),
        ]);
        let meta = parse(&bytes).unwrap();
        assert_eq!(meta.version, 3);
        assert_eq!(meta.architecture.as_deref(), Some("qwen2"));
        assert_eq!(meta.context_length, Some(32768));
    }

    #[test]
    fn stops_early_and_tolerates_garbage_after_what_it_needs() {
        let mut bytes = gguf(&[
            ("general.architecture", Val::Str("llama")),
            ("llama.context_length", Val::U32(4096)),
        ]);
        // Claim more kv pairs than are present: a strict reader would hit EOF.
        let count_at = 4 + 4 + 8;
        bytes[count_at..count_at + 8].copy_from_slice(&99u64.to_le_bytes());
        bytes.extend_from_slice(b"\xff\xff\xff");
        assert_eq!(parse(&bytes).unwrap().context_length, Some(4096));
    }

    #[test]
    fn skips_every_value_type_before_the_keys_it_wants() {
        let bytes = gguf(&[
            ("a.u32", Val::U32(1)),
            ("a.u64", Val::U64(2)),
            ("a.i32", Val::I32(-3)),
            ("a.f32", Val::F32(1.5)),
            ("a.bool", Val::Bool(true)),
            ("a.str", Val::Str("chat template {{ messages }}")),
            ("a.u32s", Val::U32Arr(&[1, 2, 3, 4])),
            ("a.strs", Val::StrArr(&["x", "yy", "zzz"])),
            ("a.nested", Val::NestedStrArr(&[&["a", "b"], &[], &["c"]])),
            ("general.architecture", Val::Str("gemma3")),
            ("gemma3.context_length", Val::U32(131072)),
        ]);
        let meta = parse(&bytes).unwrap();
        assert_eq!(meta.architecture.as_deref(), Some("gemma3"));
        assert_eq!(meta.context_length, Some(131072));
    }

    #[test]
    fn context_length_may_precede_architecture_and_other_archs_are_ignored() {
        let bytes = gguf(&[
            ("clip.context_length", Val::U32(77)),
            ("llama.context_length", Val::U64(8192)),
            ("general.architecture", Val::Str("llama")),
        ]);
        assert_eq!(parse(&bytes).unwrap().context_length, Some(8192));

        let bytes = gguf(&[
            ("general.architecture", Val::Str("llama")),
            ("clip.context_length", Val::U32(77)),
        ]);
        let meta = parse(&bytes).unwrap();
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.context_length, None);
    }

    #[test]
    fn negative_or_non_integer_context_length_is_ignored() {
        let bytes = gguf(&[
            ("general.architecture", Val::Str("llama")),
            ("llama.context_length", Val::I32(-1)),
        ]);
        assert_eq!(parse(&bytes).unwrap().context_length, None);
        let bytes = gguf(&[
            ("general.architecture", Val::Str("llama")),
            ("llama.context_length", Val::Str("lots")),
        ]);
        assert_eq!(parse(&bytes).unwrap().context_length, None);
    }

    #[test]
    fn rejects_non_gguf_and_truncated_input_without_panicking() {
        assert_eq!(
            parse(b"GGUF-not-really").unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            parse(b"ggml\x01\x00\x00\x00").unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            parse(b"GG").unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        // version 1 is not supported
        let mut v1 = gguf(&[("general.architecture", Val::Str("llama"))]);
        v1[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(parse(&v1).unwrap_err().kind(), io::ErrorKind::InvalidData);
        // truncated in the middle of a value
        let full = gguf(&[
            ("general.architecture", Val::Str("llama")),
            ("llama.context_length", Val::U32(4096)),
        ]);
        for cut in [20, 30, full.len() - 2] {
            assert!(parse(&full[..cut]).is_err(), "cut at {cut}");
        }
        // implausible string length
        let mut huge = gguf(&[("general.architecture", Val::Str("llama"))]);
        let key_len_at = 4 + 4 + 8 + 8;
        huge[key_len_at..key_len_at + 8].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
        assert_eq!(parse(&huge).unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn context_length_helper_never_fails() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(context_length(&dir.path().join("missing.gguf")), None);
        let fake = dir.path().join("fake.gguf");
        std::fs::write(&fake, b"GGUF-not-really").unwrap();
        assert_eq!(context_length(&fake), None);
        let real = dir.path().join("real.gguf");
        std::fs::write(
            &real,
            gguf(&[
                ("general.architecture", Val::Str("llama")),
                ("llama.context_length", Val::U32(2048)),
            ]),
        )
        .unwrap();
        assert_eq!(context_length(&real), Some(2048));
    }

    /// Manual check against a real file: `OHMYGPU_GGUF_FILE=/path/to/x.gguf
    /// cargo test -p ohmygpu_core gguf::tests::real_file -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_file() {
        let path = std::env::var("OHMYGPU_GGUF_FILE").expect("OHMYGPU_GGUF_FILE");
        let meta = read_metadata(&path).unwrap();
        println!("{path}: {meta:?}");
        assert!(meta.context_length.is_some());
    }
}
