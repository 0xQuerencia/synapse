//! Block-format compression helpers used by the tensor wire layer.
//!
//! Both lz4 and zstd operate on the raw element byte buffer; the
//! dtype/shape metadata lives in the `TensorPayload` header and is
//! compressed separately or sent uncompressed depending on the
//! framing choice.
//!
//! These functions are deliberately tiny — the actual algorithm is
//! provided by `lz4_flex` and `zstd`. We just dispatch on the
//! `CompressionCodec` enum and validate inputs.

use anyhow::{Context, Result, bail};

use crate::protocol::CompressionCodec;

/// Compress a buffer with the given codec.
///
/// The input is always the raw element byte buffer. The caller decides
/// whether to ship the compressed output or keep the raw buffer based
/// on size (see [`crate::protocol::TensorPayload::new_compressed`]).
pub fn compress(input: &[u8], codec: CompressionCodec) -> Result<Vec<u8>> {
    validate_codec(codec)?;
    match codec {
        CompressionCodec::None => Ok(input.to_vec()),
        CompressionCodec::Lz4 => {
            // lz4_flex prepends a 4-byte LE size header, which makes
            // it self-describing on the wire — the receiver does not
            // need a separate `original_size` field for lz4.
            Ok(lz4_flex::block::compress_prepend_size(input))
        }
        CompressionCodec::Zstd { level } => {
            let level = level.clamp(1, 22);
            zstd::stream::encode_all(input, level).context("zstd encode failed")
        }
    }
}

/// Inverse of [`compress`].
pub fn decompress(input: &[u8], codec: CompressionCodec) -> Result<Vec<u8>> {
    match codec {
        CompressionCodec::None => Ok(input.to_vec()),
        CompressionCodec::Lz4 => {
            lz4_flex::block::decompress_size_prepended(input).context("lz4 decode failed")
        }
        CompressionCodec::Zstd { .. } => {
            zstd::stream::decode_all(input).context("zstd decode failed")
        }
    }
}

/// Reject obviously-bogus zstd levels before we hand them to the
/// compressor (which would silently clamp but log nothing).
pub fn validate_codec(codec: CompressionCodec) -> Result<()> {
    if let CompressionCodec::Zstd { level } = codec
        && !(1..=22).contains(&level)
    {
        bail!("zstd level out of range: {level} (must be 1..=22)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
