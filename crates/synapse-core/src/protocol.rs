use crate::{ModelIdentity, hardware::ComputeBackend, model::validate_printable};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const PROTOCOL_VERSION: &str = "synapse/0.3.0";

/// Maximum nonce length (bytes, hex-encoded).
pub const MAX_NONCE_LEN: usize = 128;
/// Maximum SHA-256 hex digest length (64 hex chars).
pub const MAX_HASH_LEN: usize = 128;
/// Maximum signature length in bytes.
pub const MAX_SIGNATURE_LEN: usize = 256;
/// Maximum public key length in bytes.
pub const MAX_PUBKEY_LEN: usize = 256;
/// Maximum model weight chunk size (4 MiB).
pub const MAX_WEIGHT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
/// Maximum model ID length.
pub const MAX_MODEL_ID_LEN: usize = 512;
pub const MAX_TENSOR_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_TENSOR_DIMS: usize = 8;
pub const MAX_SEQUENCE_LEN: usize = 4096;
pub const MAX_TRACE_ID_LEN: usize = 64;
pub const MAX_PLATFORM_LEN: usize = 128;
pub const MAX_CPU_THREADS: u16 = 256;
pub const MAX_COMPUTE_SCORE: u64 = 10_000;
pub const MAX_AVAILABLE_RAM_MB: u64 = 1_048_576; // 1 TB
pub const MAX_WIRE_MESSAGE_BYTES: usize = MAX_TENSOR_BYTES + 4096;
pub const MAX_LAYER_RANGE_END: usize = MAX_SEQUENCE_LEN;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ProtocolError {
    #[error("protocol version mismatch: expected {expected}, got {actual}")]
    ProtocolVersion {
        expected: &'static str,
        actual: String,
    },
    #[error("invalid {field}: {reason}")]
    InvalidField {
        field: &'static str,
        reason: &'static str,
    },
    #[error("invalid shard plan: {reason}")]
    InvalidShardPlan { reason: &'static str },
    #[error("tensor payload too large: {actual} bytes exceeds {max} bytes")]
    TensorTooLarge { actual: usize, max: usize },
    #[error("tensor byte length mismatch: expected {expected}, got {actual}")]
    TensorByteLength { expected: usize, actual: usize },
    #[error("numeric overflow while validating {field}")]
    NumericOverflow { field: &'static str },
    #[error("authentication failed")]
    AuthenticationFailed,
    #[error("invalid wire message: {reason}")]
    InvalidWire { reason: &'static str },
    #[error("wire message too large: {actual} bytes exceeds {max} bytes")]
    WireMessageTooLarge { actual: usize, max: usize },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NodeRole {
    Master,
    Worker,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodeHello {
    pub protocol_version: String,
    pub role: NodeRole,
    pub model: ModelIdentity,
    pub platform: String,
    pub cpu_threads: u16,
    pub memory_bytes: Option<u64>,
    pub compute_score: Option<u64>,
    pub available_ram_mb: Option<u64>,
}

impl NodeHello {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        require_protocol_version(&self.protocol_version)?;
        self.model.validate()?;
        validate_printable("platform", &self.platform, 1, MAX_PLATFORM_LEN)?;

        if self.cpu_threads == 0 {
            return Err(ProtocolError::InvalidField {
                field: "cpu_threads",
                reason: "must be greater than zero",
            });
        }

        if self.cpu_threads > MAX_CPU_THREADS {
            return Err(ProtocolError::InvalidField {
                field: "cpu_threads",
                reason: "exceeds maximum allowed value",
            });
        }

        if let Some(score) = self.compute_score {
            if score == 0 {
                return Err(ProtocolError::InvalidField {
                    field: "compute_score",
                    reason: "must be greater than zero when provided",
                });
            }
            if score > MAX_COMPUTE_SCORE {
                return Err(ProtocolError::InvalidField {
                    field: "compute_score",
                    reason: "exceeds maximum allowed value",
                });
            }
        }

        if let Some(ram) = self.available_ram_mb {
            if ram == 0 {
                return Err(ProtocolError::InvalidField {
                    field: "available_ram_mb",
                    reason: "must be greater than zero when provided",
                });
            }
            if ram > MAX_AVAILABLE_RAM_MB {
                return Err(ProtocolError::InvalidField {
                    field: "available_ram_mb",
                    reason: "exceeds maximum allowed value",
                });
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I64,
}

impl DType {
    pub fn bytes_per_element(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I64 => 8,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorPayload {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl TensorPayload {
    pub fn new(dtype: DType, shape: Vec<usize>, bytes: Vec<u8>) -> Self {
        Self {
            dtype,
            shape,
            bytes,
        }
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.bytes.len() > MAX_TENSOR_BYTES {
            return Err(ProtocolError::TensorTooLarge {
                actual: self.bytes.len(),
                max: MAX_TENSOR_BYTES,
            });
        }

        if self.shape.is_empty() || self.shape.len() > MAX_TENSOR_DIMS {
            return Err(ProtocolError::InvalidField {
                field: "tensor shape",
                reason: "dimension count outside accepted range",
            });
        }

        let mut elements = 1_usize;
        for dim in &self.shape {
            if *dim == 0 {
                return Err(ProtocolError::InvalidField {
                    field: "tensor shape",
                    reason: "dimensions must be greater than zero",
                });
            }
            elements = elements
                .checked_mul(*dim)
                .ok_or(ProtocolError::NumericOverflow {
                    field: "tensor element count",
                })?;
        }

        let expected = elements.checked_mul(self.dtype.bytes_per_element()).ok_or(
            ProtocolError::NumericOverflow {
                field: "tensor byte count",
            },
        )?;

        if expected != self.bytes.len() {
            return Err(ProtocolError::TensorByteLength {
                expected,
                actual: self.bytes.len(),
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShardPlan {
    pub total_layers: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub dtype: DType,
    pub max_sequence_len: usize,
}

impl ShardPlan {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.total_layers == 0 {
            return Err(ProtocolError::InvalidShardPlan {
                reason: "total layer count must be greater than zero",
            });
        }

        if self.layer_start >= self.layer_end {
            return Err(ProtocolError::InvalidShardPlan {
                reason: "layer range must be non-empty",
            });
        }

        if self.layer_end > self.total_layers {
            return Err(ProtocolError::InvalidShardPlan {
                reason: "layer range exceeds total layer count",
            });
        }

        if self.max_sequence_len == 0 || self.max_sequence_len > MAX_SEQUENCE_LEN {
            return Err(ProtocolError::InvalidField {
                field: "max_sequence_len",
                reason: "outside accepted range",
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ActivationRequest {
    pub trace_id: String,
    pub sequence_position: usize,
    pub layer_range: (usize, usize),
    pub hidden_state: TensorPayload,
}

impl ActivationRequest {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        validate_trace_id(&self.trace_id)?;

        if self.sequence_position >= MAX_SEQUENCE_LEN {
            return Err(ProtocolError::InvalidField {
                field: "sequence_position",
                reason: "outside accepted range",
            });
        }

        let (start, end) = self.layer_range;
        if start >= end {
            return Err(ProtocolError::InvalidField {
                field: "layer_range",
                reason: "must be non-empty and ordered",
            });
        }

        if end > MAX_LAYER_RANGE_END {
            return Err(ProtocolError::InvalidField {
                field: "layer_range",
                reason: "outside accepted range",
            });
        }

        self.hidden_state.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ActivationResponse {
    pub trace_id: String,
    pub logits: TensorPayload,
}

impl ActivationResponse {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        validate_trace_id(&self.trace_id)?;
        self.logits.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ClearCache {
    pub trace_id: String,
}

impl ClearCache {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        validate_trace_id(&self.trace_id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Health {
    pub loaded_shard: Option<(u32, u32)>,
    pub latency_ms: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub last_error: Option<String>,
}

impl Health {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if let Some((start, end)) = self.loaded_shard {
            if start >= end {
                return Err(ProtocolError::InvalidShardPlan {
                    reason: "health shard range must be non-empty",
                });
            }
        }

        if let Some(last_error) = &self.last_error {
            validate_printable("last_error", last_error, 1, 512)?;
        }

        Ok(())
    }
}

/// Master → Worker: authentication challenge.
///
/// Sent after `NodeHello` to begin the cryptographic handshake.
/// `nonce` is a random hex string; `timestamp` is seconds since epoch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Challenge {
    pub nonce: String,
    pub timestamp: u64,
}

impl Challenge {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.nonce.is_empty() || self.nonce.len() > MAX_NONCE_LEN {
            return Err(ProtocolError::InvalidField {
                field: "nonce",
                reason: "length outside accepted range",
            });
        }
        if !self.nonce.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(ProtocolError::InvalidField {
                field: "nonce",
                reason: "must be a hex string",
            });
        }
        Ok(())
    }

    /// Generate a fresh challenge using OS randomness.
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        let nonce = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self { nonce, timestamp }
    }

    /// Return the canonical byte string that should be signed.
    pub fn signing_payload(&self) -> Vec<u8> {
        format!("synapse-challenge:{}:{}", self.nonce, self.timestamp).into_bytes()
    }
}

/// Worker → Master: response to authentication challenge.
///
/// - `pairing_code_hash`: SHA-256 hex digest of the normalised pairing code.
/// - `nonce`: echo of the challenge nonce (replay protection).
/// - `signature`: Ed25519 signature over `Challenge::signing_payload()`.
/// - `public_key`: protobuf-encoded public key (matches claimed `PeerId`).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChallengeResponse {
    pub pairing_code_hash: String,
    pub nonce: String,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
}

impl ChallengeResponse {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.pairing_code_hash.len() != 64 {
            return Err(ProtocolError::InvalidField {
                field: "pairing_code_hash",
                reason: "must be a 64-character SHA-256 hex digest",
            });
        }
        if !self
            .pairing_code_hash
            .bytes()
            .all(|b| b.is_ascii_hexdigit())
        {
            return Err(ProtocolError::InvalidField {
                field: "pairing_code_hash",
                reason: "must be a hex string",
            });
        }
        if self.nonce.is_empty() || self.nonce.len() > MAX_NONCE_LEN {
            return Err(ProtocolError::InvalidField {
                field: "nonce",
                reason: "length outside accepted range",
            });
        }
        if self.signature.is_empty() || self.signature.len() > MAX_SIGNATURE_LEN {
            return Err(ProtocolError::InvalidField {
                field: "signature",
                reason: "length outside accepted range",
            });
        }
        if self.public_key.is_empty() || self.public_key.len() > MAX_PUBKEY_LEN {
            return Err(ProtocolError::InvalidField {
                field: "public_key",
                reason: "length outside accepted range",
            });
        }
        Ok(())
    }
}

/// Worker → Master (or Master → Worker): request model weight chunk.
///
/// Used in the peer weight-transfer fallback when HF Hub is unreachable.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WeightRequest {
    /// The model repo ID (e.g. `hf-internal-testing/tiny-random-LlamaForCausalLM`).
    pub model_id: String,
    /// File to request (`config.json`, `model.safetensors`, or `tokenizer.json`).
    pub filename: String,
    /// Byte offset for chunked transfers.
    pub offset: u64,
}

impl WeightRequest {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.model_id.is_empty() || self.model_id.len() > MAX_MODEL_ID_LEN {
            return Err(ProtocolError::InvalidField {
                field: "model_id",
                reason: "length outside accepted range",
            });
        }
        // Only allow specific safe filenames
        match self.filename.as_str() {
            "config.json" | "model.safetensors" | "tokenizer.json" => {}
            _ => {
                return Err(ProtocolError::InvalidField {
                    field: "filename",
                    reason: "only config.json, model.safetensors, and tokenizer.json are allowed",
                });
            }
        }
        Ok(())
    }
}

/// Master → Worker: a chunk of model weight data.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WeightTransfer {
    pub model_id: String,
    pub filename: String,
    /// Byte offset of this chunk in the file.
    pub offset: u64,
    /// Total file size (for progress tracking and final validation).
    pub total_bytes: u64,
    /// The chunk payload (raw bytes).
    pub chunk: Vec<u8>,
    /// SHA-256 hex digest of the *entire* file (present only on the final chunk).
    pub sha256: Option<String>,
}

impl WeightTransfer {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.model_id.is_empty() || self.model_id.len() > MAX_MODEL_ID_LEN {
            return Err(ProtocolError::InvalidField {
                field: "model_id",
                reason: "length outside accepted range",
            });
        }
        match self.filename.as_str() {
            "config.json" | "model.safetensors" | "tokenizer.json" => {}
            _ => {
                return Err(ProtocolError::InvalidField {
                    field: "filename",
                    reason: "only config.json, model.safetensors, and tokenizer.json are allowed",
                });
            }
        }
        if self.chunk.len() > MAX_WEIGHT_CHUNK_BYTES {
            return Err(ProtocolError::TensorTooLarge {
                actual: self.chunk.len(),
                max: MAX_WEIGHT_CHUNK_BYTES,
            });
        }
        if let Some(sha256) = &self.sha256 {
            if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(ProtocolError::InvalidField {
                    field: "sha256",
                    reason: "must be a 64-character SHA-256 hex digest",
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum WireMessage {
    NodeHello(NodeHello),
    ShardPlan(ShardPlan),
    ActivationRequest(ActivationRequest),
    ActivationResponse(ActivationResponse),
    ClearCache(ClearCache),
    Health(Health),
    Challenge(Challenge),
    ChallengeResponse(ChallengeResponse),
    WeightRequest(WeightRequest),
    WeightTransfer(WeightTransfer),
}

impl WireMessage {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        match self {
            WireMessage::NodeHello(message) => message.validate(),
            WireMessage::ShardPlan(message) => message.validate(),
            WireMessage::ActivationRequest(message) => message.validate(),
            WireMessage::ActivationResponse(message) => message.validate(),
            WireMessage::ClearCache(message) => message.validate(),
            WireMessage::Health(message) => message.validate(),
            WireMessage::Challenge(message) => message.validate(),
            WireMessage::ChallengeResponse(message) => message.validate(),
            WireMessage::WeightRequest(message) => message.validate(),
            WireMessage::WeightTransfer(message) => message.validate(),
        }
    }
}

pub fn encode_wire_message(message: &WireMessage) -> Result<Vec<u8>, ProtocolError> {
    message.validate()?;
    serde_json::to_vec(message).map_err(|_| ProtocolError::InvalidWire {
        reason: "message could not be encoded as JSON",
    })
}

pub fn decode_wire_message(bytes: &[u8]) -> Result<WireMessage, ProtocolError> {
    if bytes.len() > MAX_WIRE_MESSAGE_BYTES {
        return Err(ProtocolError::WireMessageTooLarge {
            actual: bytes.len(),
            max: MAX_WIRE_MESSAGE_BYTES,
        });
    }

    let message: WireMessage =
        serde_json::from_slice(bytes).map_err(|_| ProtocolError::InvalidWire {
            reason: "message JSON could not be decoded",
        })?;
    message.validate()?;
    Ok(message)
}

fn require_protocol_version(actual: &str) -> Result<(), ProtocolError> {
    if actual != PROTOCOL_VERSION {
        return Err(ProtocolError::ProtocolVersion {
            expected: PROTOCOL_VERSION,
            actual: actual.to_owned(),
        });
    }

    Ok(())
}

fn validate_trace_id(trace_id: &str) -> Result<(), ProtocolError> {
    validate_printable("trace_id", trace_id, 1, MAX_TRACE_ID_LEN)?;

    if !trace_id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    {
        return Err(ProtocolError::InvalidField {
            field: "trace_id",
            reason: "only ASCII letters, digits, dash, and underscore are allowed",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_protocol_version_mismatch() {
        let hello = NodeHello {
            protocol_version: "old".to_owned(),
            role: NodeRole::Master,
            model: ModelIdentity::new("hf-internal-testing/tiny-random-LlamaForCausalLM"),
            platform: "linux".to_owned(),
            cpu_threads: 8,
            memory_bytes: None,
            compute_score: None,
            available_ram_mb: None,
        };

        assert!(matches!(
            hello.validate(),
            Err(ProtocolError::ProtocolVersion { .. })
        ));
    }

    #[test]
    fn validates_tensor_byte_length() {
        let tensor = TensorPayload::new(DType::F32, vec![1, 4], vec![0; 16]);
        tensor.validate().expect("shape and bytes match");
    }

    #[test]
    fn rejects_tensor_byte_length_mismatch() {
        let tensor = TensorPayload::new(DType::F32, vec![1, 4], vec![0; 12]);
        assert!(matches!(
            tensor.validate(),
            Err(ProtocolError::TensorByteLength { .. })
        ));
    }

    #[test]
    fn rejects_invalid_trace_id() {
        let message = ActivationRequest {
            trace_id: "../bad".to_owned(),
            sequence_position: 0,
            layer_range: (0, 1),
            hidden_state: TensorPayload::new(DType::F32, vec![1, 1], vec![0; 4]),
        };

        assert!(matches!(
            message.validate(),
            Err(ProtocolError::InvalidField {
                field: "trace_id",
                ..
            })
        ));
    }

    #[test]
    fn wire_message_round_trips_after_validation() {
        let message = WireMessage::Health(Health {
            loaded_shard: Some((2, 4)),
            latency_ms: Some(7),
            memory_bytes: Some(1024),
            last_error: None,
        });

        let encoded = encode_wire_message(&message).expect("message should encode");
        let decoded = decode_wire_message(&encoded).expect("message should decode");

        assert_eq!(decoded, message);
    }

    #[test]
    fn decode_rejects_invalid_wire_data() {
        let err = decode_wire_message(b"not-json").unwrap_err();
        assert!(matches!(err, ProtocolError::InvalidWire { .. }));
    }

    #[test]
    fn rejects_invalid_layer_range() {
        let message = ActivationRequest {
            trace_id: "trace_ok".to_owned(),
            sequence_position: 0,
            layer_range: (3, 3),
            hidden_state: TensorPayload::new(DType::F32, vec![1, 1], vec![0; 4]),
        };

        assert!(matches!(
            message.validate(),
            Err(ProtocolError::InvalidField {
                field: "layer_range",
                ..
            })
        ));
    }

    #[test]
    fn rejects_layer_range_above_protocol_bounds() {
        let message = ActivationRequest {
            trace_id: "trace_ok".to_owned(),
            sequence_position: 0,
            layer_range: (0, MAX_LAYER_RANGE_END + 1),
            hidden_state: TensorPayload::new(DType::F32, vec![1, 1], vec![0; 4]),
        };

        assert!(matches!(
            message.validate(),
            Err(ProtocolError::InvalidField {
                field: "layer_range",
                ..
            })
        ));
    }

    #[test]
    fn rejects_reversed_layer_range() {
        let message = ActivationRequest {
            trace_id: "trace_ok".to_owned(),
            sequence_position: 0,
            layer_range: (4, 1),
            hidden_state: TensorPayload::new(DType::F32, vec![1, 1], vec![0; 4]),
        };

        assert!(matches!(
            message.validate(),
            Err(ProtocolError::InvalidField {
                field: "layer_range",
                ..
            })
        ));
    }

    #[test]
    fn accepts_layer_range_at_protocol_upper_bound() {
        let message = ActivationRequest {
            trace_id: "trace_ok".to_owned(),
            sequence_position: 0,
            layer_range: (MAX_LAYER_RANGE_END - 1, MAX_LAYER_RANGE_END),
            hidden_state: TensorPayload::new(DType::F32, vec![1, 1], vec![0; 4]),
        };

        message
            .validate()
            .expect("upper-bound layer range should be accepted");
    }

    #[test]
    fn clear_cache_wire_message_round_trips() {
        let message = WireMessage::ClearCache(ClearCache {
            trace_id: "clear_1".to_owned(),
        });

        let encoded = encode_wire_message(&message).expect("message should encode");
        let decoded = decode_wire_message(&encoded).expect("message should decode");

        assert_eq!(decoded, message);
    }

    #[test]
    fn clear_cache_validation_rejects_invalid_trace_id() {
        let message = ClearCache {
            trace_id: "../bad".to_owned(),
        };

        assert!(matches!(
            message.validate(),
            Err(ProtocolError::InvalidField {
                field: "trace_id",
                ..
            })
        ));
    }

    #[test]
    fn clear_cache_wire_encode_rejects_invalid_trace_id() {
        let message = WireMessage::ClearCache(ClearCache {
            trace_id: "../bad".to_owned(),
        });

        assert!(matches!(
            encode_wire_message(&message),
            Err(ProtocolError::InvalidField {
                field: "trace_id",
                ..
            })
        ));
    }

    #[test]
    fn clear_cache_wire_decode_rejects_invalid_trace_id() {
        let err = decode_wire_message(br#"{"ClearCache":{"trace_id":"../bad"}}"#).unwrap_err();
        assert!(matches!(
            err,
            ProtocolError::InvalidField {
                field: "trace_id",
                ..
            }
        ));
    }

    #[test]
    fn rejects_cpu_threads_above_maximum() {
        let hello = NodeHello {
            protocol_version: PROTOCOL_VERSION.to_owned(),
            role: NodeRole::Worker,
            model: ModelIdentity::new("hf-internal-testing/tiny-random-LlamaForCausalLM"),
            platform: "linux".to_owned(),
            cpu_threads: MAX_CPU_THREADS + 1,
            memory_bytes: None,
            compute_score: None,
            available_ram_mb: None,
        };
        assert!(matches!(
            hello.validate(),
            Err(ProtocolError::InvalidField {
                field: "cpu_threads",
                ..
            })
        ));
    }

    #[test]
    fn rejects_compute_score_above_maximum() {
        let hello = NodeHello {
            protocol_version: PROTOCOL_VERSION.to_owned(),
            role: NodeRole::Worker,
            model: ModelIdentity::new("hf-internal-testing/tiny-random-LlamaForCausalLM"),
            platform: "linux".to_owned(),
            cpu_threads: 8,
            memory_bytes: None,
            compute_score: Some(MAX_COMPUTE_SCORE + 1),
            available_ram_mb: None,
        };
        assert!(matches!(
            hello.validate(),
            Err(ProtocolError::InvalidField {
                field: "compute_score",
                ..
            })
        ));
    }

    #[test]
    fn rejects_available_ram_above_maximum() {
        let hello = NodeHello {
            protocol_version: PROTOCOL_VERSION.to_owned(),
            role: NodeRole::Worker,
            model: ModelIdentity::new("hf-internal-testing/tiny-random-LlamaForCausalLM"),
            platform: "linux".to_owned(),
            cpu_threads: 8,
            memory_bytes: None,
            compute_score: None,
            available_ram_mb: Some(MAX_AVAILABLE_RAM_MB + 1),
        };
        assert!(matches!(
            hello.validate(),
            Err(ProtocolError::InvalidField {
                field: "available_ram_mb",
                ..
            })
        ));
    }

    #[test]
    fn decode_rejects_wire_message_too_large() {
        let huge = vec![b'x'; MAX_WIRE_MESSAGE_BYTES + 1];
        let err = decode_wire_message(&huge).unwrap_err();
        assert!(matches!(err, ProtocolError::WireMessageTooLarge { .. }));
    }
}
