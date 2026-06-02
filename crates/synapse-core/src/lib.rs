pub mod hash;
pub mod identity;
pub mod inference;
pub mod model;
pub mod netconfig;
pub mod network;
pub mod pairing;
pub mod protocol;
pub mod shard;

pub use identity::{NodeIdentity, set_identity_path_override};
pub use inference::{
    SimpleTokenizer, ToyLlama, ToyLlamaConfig, download_model, format_instruct_prompt,
    get_best_device, list_cached_hf_models, payload_to_tensor, tensor_to_payload, unescape_prompt,
};
pub use model::ModelIdentity;
pub use netconfig::NetConfig;
pub use pairing::{
    PairingRateLimiter, generate_pairing_code, hash_pairing_code, normalize_pairing_code,
    validate_pairing_code, verify_pairing_code, verify_pairing_code_hash,
};
pub use protocol::{
    ActivationRequest, ActivationResponse, Challenge, ChallengeResponse, ClearCache, DType, Health,
    NodeHello, NodeRole, PROTOCOL_VERSION, ProtocolError, ShardPlan, TensorPayload, WeightRequest,
    WeightTransfer, WireMessage, decode_wire_message, encode_wire_message,
};
pub use shard::{
    LayerRange, NodeCapability, RAM_PER_LAYER_HEURISTIC_MB, StaticShardAssignment,
    WeightedShardParticipant, plan_static_shards, plan_weighted_shards, split_layers,
    worker_shard_plan,
};
