pub mod backend;
pub mod compute_score;
pub mod hardware;
pub mod hash;
pub mod identity;
pub mod inference;
pub mod model;
pub mod netconfig;
pub mod network;
pub mod pairing;
pub mod protocol;
pub mod shard;

pub use compute_score::{MODEL_LAYER_FOOTPRINT_MB, blend_compute_score};
pub use hardware::{ComputeBackend, HardwareProbe};
pub use identity::{NodeIdentity, set_identity_path_override};
pub use inference::{
    EncodedTensor, SimpleTokenizer, ToyLlama, ToyLlamaConfig, download_model, encode_tensor,
    encode_with_compression, format_instruct_prompt, get_best_device, list_cached_hf_models,
    payload_to_tensor, run_micro_benchmark, tensor_to_payload, unescape_prompt,
};
pub use model::ModelIdentity;
pub use netconfig::{NetConfig, RelayInfo};
pub use pairing::{
    PairingRateLimiter, generate_pairing_code, hash_pairing_code, normalize_pairing_code,
    validate_pairing_code, verify_pairing_code, verify_pairing_code_hash,
};
pub use protocol::{
    ActivationRequest, ActivationResponse, Challenge, ChallengeResponse, ClearCache,
    CompressionCodec, DType, HardwareSummary, Health, KvTruncate, NodeHello, NodeRole,
    PROTOCOL_VERSION, ProtocolError, ShardPlan, TensorPayload, WeightRequest, WeightTransfer,
    WireMessage, WorkerCapability, decode_wire_message, encode_wire_message,
};
pub use shard::{
    CapabilityShardParticipant, LayerRange, NodeCapability, RAM_PER_LAYER_HEURISTIC_MB,
    StaticShardAssignment, WeightedShardParticipant, per_layer_throughput, plan_static_shards,
    plan_weighted_shards, plan_weighted_shards_capable, split_layers, worker_shard_plan,
};
