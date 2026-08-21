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
        /// Pairing code is now sent as a SHA-256 hash (never plaintext after Phase 4).
        /// For backward compat, this field is kept but the handler validates the hash.
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
}

impl ResponseMessage {
    /// Create a `Rejected` response, truncating the reason to `MAX_REJECTED_REASON_LEN`.
    pub fn rejected(reason: &str) -> Self {
        let truncated = if reason.len() > MAX_REJECTED_REASON_LEN {
            &reason[..MAX_REJECTED_REASON_LEN]
        } else {
            reason
        };
        ResponseMessage::Rejected {
            reason: truncated.to_owned(),
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
}
