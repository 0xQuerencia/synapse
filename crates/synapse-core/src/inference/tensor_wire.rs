//! Zero-copy tensor serialization between candle `Tensor` and wire `TensorPayload`.
//!
//! Uses `bytemuck::cast_slice` for fast f32↔u8 conversion. Both sides are assumed
//! little-endian (ARM64 + x86_64), which covers all practical deployment targets.

use anyhow::{Context, Result, bail, ensure};
use candle_core::{Device, Tensor};

use crate::protocol::{DType, TensorPayload};

/// Convert a protocol `DType` to a candle `DType`.
pub fn to_candle_dtype(dtype: DType) -> candle_core::DType {
    match dtype {
        DType::F32 => candle_core::DType::F32,
        DType::F16 => candle_core::DType::F16,
        DType::BF16 => candle_core::DType::BF16,
        DType::I64 => candle_core::DType::I64,
    }
}

/// Convert a candle `DType` to a protocol `DType`.
pub fn from_candle_dtype(dtype: candle_core::DType) -> Result<DType> {
    match dtype {
        candle_core::DType::F32 => Ok(DType::F32),
        candle_core::DType::F16 => Ok(DType::F16),
        candle_core::DType::BF16 => Ok(DType::BF16),
        candle_core::DType::I64 => Ok(DType::I64),
        other => bail!("unsupported candle dtype for wire transfer: {:?}", other),
    }
}

/// Serialize a candle `Tensor` into a wire-safe `TensorPayload`.
///
/// Tensors on accelerator devices are copied to `Device::Cpu` before serialization.
/// CPU tensors are serialized directly via `bytemuck`.
pub fn tensor_to_payload(tensor: &Tensor) -> Result<TensorPayload> {
    let tensor = if tensor.device().is_cpu() {
        tensor.clone()
    } else {
        tensor
            .to_device(&Device::Cpu)
            .context("failed to copy tensor to CPU for wire serialization")?
    };

    let dtype = from_candle_dtype(tensor.dtype())?;
    let shape = tensor.dims().to_vec();

    let bytes = match dtype {
        DType::F32 => {
            let data = tensor.flatten_all()?.to_vec1::<f32>()?;
            bytemuck::cast_slice::<f32, u8>(&data).to_vec()
        }
        DType::I64 => {
            let data = tensor.flatten_all()?.to_vec1::<i64>()?;
            bytemuck::cast_slice::<i64, u8>(&data).to_vec()
        }
        DType::F16 => {
            // half::f16 is repr(transparent) for u16, so 2 bytes per element
            let raw = tensor.flatten_all()?.to_vec1::<half::f16>()?;
            bytemuck::cast_slice::<half::f16, u8>(&raw).to_vec()
        }
        DType::BF16 => {
            let raw = tensor.flatten_all()?.to_vec1::<half::bf16>()?;
            bytemuck::cast_slice::<half::bf16, u8>(&raw).to_vec()
        }
    };

    Ok(TensorPayload::new(dtype, shape, bytes))
}

/// Deserialize a wire `TensorPayload` back into a candle `Tensor` on CPU.
pub fn payload_to_tensor(payload: &TensorPayload) -> Result<Tensor> {
    bytes_to_tensor(&payload.shape, payload.dtype, &payload.bytes)
}

/// Reconstruct a candle `Tensor` from raw shape, dtype, and bytes.
pub fn bytes_to_tensor(shape: &[usize], dtype: DType, bytes: &[u8]) -> Result<Tensor> {
    let element_count = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| anyhow::anyhow!("tensor shape overflows element count"))
    })?;
    let expected_len = element_count
        .checked_mul(dtype.bytes_per_element())
        .ok_or_else(|| anyhow::anyhow!("tensor byte length overflow"))?;
    ensure!(
        bytes.len() == expected_len,
        "tensor byte length mismatch: expected {}, got {}",
        expected_len,
        bytes.len()
    );

    let tensor = match dtype {
        DType::F32 => {
            let data = decode_f32(bytes)?;
            Tensor::from_slice(data.as_slice(), shape, &Device::Cpu)?
        }
        DType::I64 => {
            let data = decode_i64(bytes)?;
            Tensor::from_slice(data.as_slice(), shape, &Device::Cpu)?
        }
        DType::F16 => {
            let data = decode_f16(bytes)?;
            Tensor::from_slice(data.as_slice(), shape, &Device::Cpu)?
        }
        DType::BF16 => {
            let data = decode_bf16(bytes)?;
            Tensor::from_slice(data.as_slice(), shape, &Device::Cpu)?
        }
    };

    Ok(tensor)
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if let Ok(aligned) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        return Ok(aligned.to_vec());
    }

    let mut values = Vec::with_capacity(bytes.len() / std::mem::size_of::<f32>());
    for chunk in bytes.chunks_exact(4) {
        let mut raw = [0_u8; 4];
        raw.copy_from_slice(chunk);
        values.push(f32::from_le_bytes(raw));
    }
    Ok(values)
}

fn decode_i64(bytes: &[u8]) -> Result<Vec<i64>> {
    if let Ok(aligned) = bytemuck::try_cast_slice::<u8, i64>(bytes) {
        return Ok(aligned.to_vec());
    }

    let mut values = Vec::with_capacity(bytes.len() / std::mem::size_of::<i64>());
    for chunk in bytes.chunks_exact(8) {
        let mut raw = [0_u8; 8];
        raw.copy_from_slice(chunk);
        values.push(i64::from_le_bytes(raw));
    }
    Ok(values)
}

fn decode_f16(bytes: &[u8]) -> Result<Vec<half::f16>> {
    if let Ok(aligned) = bytemuck::try_cast_slice::<u8, half::f16>(bytes) {
        return Ok(aligned.to_vec());
    }

    let mut values = Vec::with_capacity(bytes.len() / std::mem::size_of::<half::f16>());
    for chunk in bytes.chunks_exact(2) {
        let mut raw = [0_u8; 2];
        raw.copy_from_slice(chunk);
        values.push(half::f16::from_bits(u16::from_le_bytes(raw)));
    }
    Ok(values)
}

fn decode_bf16(bytes: &[u8]) -> Result<Vec<half::bf16>> {
    if let Ok(aligned) = bytemuck::try_cast_slice::<u8, half::bf16>(bytes) {
        return Ok(aligned.to_vec());
    }

    let mut values = Vec::with_capacity(bytes.len() / std::mem::size_of::<half::bf16>());
    for chunk in bytes.chunks_exact(2) {
        let mut raw = [0_u8; 2];
        raw.copy_from_slice(chunk);
        values.push(half::bf16::from_bits(u16::from_le_bytes(raw)));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maybe_test_accelerator_device() -> Option<Device> {
        if std::env::var_os("SYNAPSE_TEST_ACCEL").is_none() {
            return None;
        }
        if !cfg!(any(feature = "cuda", feature = "metal")) {
            return None;
        }

        if candle_core::utils::cuda_is_available() {
            if let Ok(device) = Device::new_cuda(0) {
                return Some(device);
            }
        }
        if candle_core::utils::metal_is_available() {
            if let Ok(device) = Device::new_metal(0) {
                return Some(device);
            }
        }
        None
    }

    #[test]
    fn f32_roundtrip() {
        let original = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &Device::Cpu)
            .unwrap()
            .reshape((2, 2))
            .unwrap();
        assert!(original.device().is_cpu());

        let payload = tensor_to_payload(&original).unwrap();
        assert_eq!(payload.dtype, DType::F32);
        assert_eq!(payload.shape, vec![2, 2]);
        assert_eq!(payload.bytes.len(), 4 * 4); // 4 floats × 4 bytes

        let restored = payload_to_tensor(&payload).unwrap();
        assert!(restored.device().is_cpu());
        let orig_values = original.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rest_values = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(orig_values, rest_values);
    }

    #[test]
    fn accelerator_tensor_roundtrip_is_opt_in_and_non_flaky() {
        let Some(device) = maybe_test_accelerator_device() else {
            return;
        };

        let original = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &device)
            .unwrap()
            .reshape((2, 2))
            .unwrap();
        assert!(!original.device().is_cpu());

        let payload = tensor_to_payload(&original).expect("accelerator tensor should serialize");
        let restored = payload_to_tensor(&payload).expect("payload should deserialize on CPU");
        assert!(restored.device().is_cpu());

        let orig_values = original.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rest_values = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(orig_values, rest_values);
    }

    #[test]
    fn payload_validates() {
        let tensor = Tensor::new(&[1.0_f32, 2.0, 3.0], &Device::Cpu).unwrap();
        let payload = tensor_to_payload(&tensor).unwrap();
        payload
            .validate()
            .expect("roundtripped payload should validate");
    }

    #[test]
    fn dtype_roundtrip() {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I64] {
            let candle_dt = to_candle_dtype(dtype);
            let back = from_candle_dtype(candle_dt).unwrap();
            assert_eq!(dtype, back);
        }
    }

    #[test]
    fn bytes_to_tensor_handles_unaligned_f32_bytes() {
        let mut raw = vec![0_u8];
        raw.extend_from_slice(bytemuck::cast_slice(&[1.0_f32]));

        let tensor = bytes_to_tensor(&[1], DType::F32, &raw[1..]).expect("unaligned bytes decode");
        let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values, vec![1.0_f32]);
    }

    #[test]
    fn bytes_to_tensor_rejects_invalid_length() {
        let err = bytes_to_tensor(&[2], DType::F32, &[0_u8; 4]).unwrap_err();
        assert!(err.to_string().contains("tensor byte length mismatch"));
    }
}
