use serde::{Deserialize, Serialize};

use crate::{
    ActivationRequest, ActivationResponse, Challenge, ChallengeResponse, ClearCache, Health,
    KvTruncate, NodeHello, WeightRequest, WeightTransfer, WorkerCapability,
};

/// Maximum length for the `Rejected.reason` string to prevent memory abuse.
const MAX_REJECTED_REASON_LEN: usize = 512;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RequestMessage {
    NodeHello {
        hello: NodeHello,
        /// Reserved field, empty in protocol 0.4. Shared secrets and static
        /// hashes must never be sent to discovery candidates.
        pairing_code: String,
    },
    ActivationRequest(ActivationRequest),
    ClearCache(ClearCache),
    /// Master → Worker: begin challenge-response handshake.
    Challenge(Challenge),
    /// Worker → Master: respond to a challenge with proof of identity.
    ChallengeResponse(ChallengeResponse),
    /// Request a chunk of model weights from a peer.
    WeightRequest(WeightRequest),
    /// Worker → Master: "my capabilities are now X" (heartbeat).
    /// Master updates the worker registry and may re-plan if VRAM
    /// is tight.
    CapabilityPing(WorkerCapability),
    /// Master → Worker: pre-warm connection with health check.
    HealthPing,
    /// Master → Worker: roll the KV cache back after rejected speculative
    /// draft tokens.
    KvTruncate(KvTruncate),
    // Append-only: codec framing relies on existing data-message ordinals.
    PrepareShard(crate::inference::assignment::ShardAssignment),
    ReleaseShard(crate::inference::assignment::LeaseKey),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ResponseMessage {
    Authenticated,
    ActivationResponse(ActivationResponse),
    ClearCacheAck(ClearCache),
    Rejected {
        reason: String,
    },
    /// Sent by master in response to NodeHello — worker must reply with ChallengeResponse.
    ChallengeIssued(Challenge),
    /// Deliver a chunk of model weight data.
    WeightTransfer(WeightTransfer),
    /// Master → Worker: capability ping accepted.
    CapabilityAck {
        accepted: bool,
    },
    /// Worker → Master: health check response.
    HealthPong(Health),
    /// Worker → Master: KV cache truncated to the requested length.
    KvTruncateAck(KvTruncate),
    ShardReady(crate::inference::assignment::ShardReady),
    ShardReleased(crate::inference::assignment::LeaseKey),
    PairingAccepted(crate::pairing::CoordinatorProof),
}

impl ResponseMessage {
    /// Create a `Rejected` response, truncating the reason to `MAX_REJECTED_REASON_LEN`.
    pub fn rejected(reason: &str) -> Self {
        let mut end = reason.len().min(MAX_REJECTED_REASON_LEN);
        while !reason.is_char_boundary(end) {
            end -= 1;
        }
        let truncated = &reason[..end];
        ResponseMessage::Rejected {
            reason: truncated.to_owned(),
        }
    }
}

impl RequestMessage {
    pub fn validate_wire(&self) -> Result<(), crate::ProtocolError> {
        match self {
            Self::NodeHello {
                hello,
                pairing_code,
            } => {
                if !pairing_code.is_empty() {
                    return Err(crate::ProtocolError::AuthenticationFailed);
                }
                hello.validate()
            }
            Self::ActivationRequest(value) => value.validate(),
            Self::ClearCache(value) => value.validate(),
            Self::Challenge(value) => value.validate(),
            Self::ChallengeResponse(value) => value.validate(),
            Self::WeightRequest(value) => value.validate(),
            Self::CapabilityPing(value) => value.validate(),
            Self::HealthPing => Ok(()),
            Self::KvTruncate(value) => value.validate(),
            Self::PrepareShard(value) => {
                value
                    .validate()
                    .map_err(|_| crate::ProtocolError::InvalidField {
                        field: "assignment",
                        reason: "invalid shard assignment",
                    })
            }
            Self::ReleaseShard(value) => {
                value
                    .validate()
                    .map_err(|_| crate::ProtocolError::InvalidField {
                        field: "lease",
                        reason: "invalid lease key",
                    })
            }
        }
    }
}

impl ResponseMessage {
    pub fn validate_wire(&self) -> Result<(), crate::ProtocolError> {
        match self {
            // Protocol 0.4 requires a coordinator proof, never a bare ACK.
            Self::Authenticated => Err(crate::ProtocolError::AuthenticationFailed),
            Self::ActivationResponse(value) => value.validate(),
            Self::ClearCacheAck(value) => value.validate(),
            Self::Rejected { reason } if reason.len() <= MAX_REJECTED_REASON_LEN => Ok(()),
            Self::Rejected { .. } => Err(crate::ProtocolError::InvalidField {
                field: "reason",
                reason: "rejection exceeds length limit",
            }),
            Self::ChallengeIssued(value) => value.validate(),
            Self::WeightTransfer(value) => value.validate(),
            Self::CapabilityAck { .. } => Ok(()),
            Self::HealthPong(value) => value.validate(),
            Self::KvTruncateAck(value) => value.validate(),
            Self::ShardReady(value) => {
                value
                    .validate()
                    .map_err(|_| crate::ProtocolError::InvalidField {
                        field: "ready",
                        reason: "invalid ready shard",
                    })
            }
            Self::ShardReleased(value) => {
                value
                    .validate()
                    .map_err(|_| crate::ProtocolError::InvalidField {
                        field: "lease",
                        reason: "invalid lease key",
                    })
            }
            Self::PairingAccepted(value) => {
                if value.nonce.len() != 32
                    || value.proof.len() != 64
                    || !value.nonce.bytes().all(|b| b.is_ascii_hexdigit())
                    || !value.proof.bytes().all(|b| b.is_ascii_hexdigit())
                {
                    return Err(crate::ProtocolError::AuthenticationFailed);
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RequestMessage, ResponseMessage};

    #[test]
    fn clear_cache_request_round_trips_and_validates() {
        let message = RequestMessage::ClearCache(crate::ClearCache {
            trace_id: "cache_req_1".to_owned(),
        });

        let encoded = serde_json::to_vec(&message).expect("message should serialize");
        let decoded: RequestMessage =
            serde_json::from_slice(&encoded).expect("message should deserialize");

        match decoded {
            RequestMessage::ClearCache(clear) => {
                clear
                    .validate()
                    .expect("round-tripped request should remain valid");
                assert_eq!(clear.trace_id, "cache_req_1");
            }
            _ => panic!("decoded wrong request variant"),
        }
    }

    #[test]
    fn clear_cache_ack_round_trips_and_validates() {
        let message = ResponseMessage::ClearCacheAck(crate::ClearCache {
            trace_id: "cache_ack_1".to_owned(),
        });

        let encoded = serde_json::to_vec(&message).expect("message should serialize");
        let decoded: ResponseMessage =
            serde_json::from_slice(&encoded).expect("message should deserialize");

        match decoded {
            ResponseMessage::ClearCacheAck(ack) => {
                ack.validate()
                    .expect("round-tripped response should remain valid");
                assert_eq!(ack.trace_id, "cache_ack_1");
            }
            _ => panic!("decoded wrong response variant"),
        }
    }

    #[test]
    fn clear_cache_ack_invalid_trace_fails_validation() {
        let decoded: ResponseMessage =
            serde_json::from_str(r#"{"ClearCacheAck":{"trace_id":"../bad"}}"#)
                .expect("json should deserialize");

        match decoded {
            ResponseMessage::ClearCacheAck(ack) => {
                assert!(
                    ack.validate().is_err(),
                    "invalid trace_id should be rejected"
                );
            }
            _ => panic!("decoded wrong response variant"),
        }
    }

    #[test]
    fn challenge_round_trips() {
        let challenge = crate::Challenge::generate();
        let message = ResponseMessage::ChallengeIssued(challenge.clone());
        let encoded = serde_json::to_vec(&message).expect("should serialize");
        let decoded: ResponseMessage =
            serde_json::from_slice(&encoded).expect("should deserialize");
        match decoded {
            ResponseMessage::ChallengeIssued(c) => {
                assert_eq!(c.nonce, challenge.nonce);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn weight_request_rejects_bad_filename() {
        let req = crate::WeightRequest {
            model_id: "some/model".to_owned(),
            filename: "../../etc/passwd".to_owned(),
            offset: 0,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn rejection_truncates_on_a_utf8_boundary() {
        let reason = "é".repeat(255) + "界";
        let ResponseMessage::Rejected { reason } = ResponseMessage::rejected(&reason) else {
            panic!("expected rejection");
        };
        assert_eq!(reason.len(), 510);
    }
}
