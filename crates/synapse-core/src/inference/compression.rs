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
    fn lz4_roundtrip_preserves_bytes() {
        // Realistic-ish activation bytes: floats clustered around 0.
        // lz4 achieves 2-3x on this; random bytes would not compress
        // at all, which is the right outcome (we test that elsewhere).
        let mut floats: Vec<f32> = Vec::with_capacity(262_144);
        let mut x: f32 = 0.0;
        for _ in 0..262_144 {
            x = (x * 1.13 + 0.07).fract() * 2.0 - 1.0;
            floats.push(x);
        }
        let big: Vec<u8> = bytemuck::cast_slice(&floats).to_vec();
        let compressed = compress(&big, CompressionCodec::Lz4).unwrap();
        assert!(
            compressed.len() < big.len() / 2,
            "lz4 should shrink activation-like bytes at least 2x; got {} -> {}",
            big.len(),
            compressed.len()
        );
        let back = decompress(&compressed, CompressionCodec::Lz4).unwrap();
        assert_eq!(back, big);
    }

    #[test]
    fn zstd_roundtrip_preserves_bytes() {
        let big: Vec<u8> = (0..4096_u32)
            .flat_map(|i| (i ^ 0x55).to_le_bytes())
            .collect();
        let compressed = compress(&big, CompressionCodec::Zstd { level: 3 }).unwrap();
        let back = decompress(&compressed, CompressionCodec::Zstd { level: 3 }).unwrap();
        assert_eq!(back, big);
    }

    #[test]
    fn none_codec_is_passthrough() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(compress(&data, CompressionCodec::None).unwrap(), data);
        assert_eq!(decompress(&data, CompressionCodec::None).unwrap(), data);
    }

    #[test]
    fn zstd_level_validation() {
        assert!(validate_codec(CompressionCodec::Zstd { level: 0 }).is_err());
        assert!(validate_codec(CompressionCodec::Zstd { level: 23 }).is_err());
        assert!(validate_codec(CompressionCodec::Zstd { level: 1 }).is_ok());
        assert!(validate_codec(CompressionCodec::Zstd { level: 22 }).is_ok());
    }

    #[test]
    fn activations_compress_well() {
        // Realistic-ish Llama hidden state: floats clustered around 0
        // with a normal distribution. lz4 typically achieves 2-3x.
        let mut state: Vec<f32> = Vec::with_capacity(4096);
        let mut x: f32 = 0.0;
        for _ in 0..4096 {
            // Cheap LCG-ish PRNG to avoid pulling in `rand` here.
            x = (x * 1.13 + 0.07).fract() * 2.0 - 1.0;
            state.push(x);
        }
        let raw = bytemuck::cast_slice::<f32, u8>(&state).to_vec();
        let compressed = compress(&raw, CompressionCodec::Lz4).unwrap();
        let ratio = compressed.len() as f32 / raw.len() as f32;
        assert!(
            ratio < 0.9,
            "lz4 should compress activations < 10%; got {ratio:.2}"
        );
    }
}
