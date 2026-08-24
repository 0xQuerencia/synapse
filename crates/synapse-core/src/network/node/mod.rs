use std::{
    collections::{HashMap, HashSet, VecDeque},
    io,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use libp2p::{
    Multiaddr, PeerId, Swarm, autonat, dcutr, identify, kad, mdns, ping, relay, request_response,
    swarm::NetworkBehaviour, swarm::behaviour::toggle::Toggle,
};
use thiserror::Error;
use tokio::sync::mpsc;

mod adaptive_timeout;
mod batch_tuning;
mod speculation;

use crate::{
    ModelIdentity, NodeCapability, PairingRateLimiter,
    inference::{ShardedModel, SimpleTokenizer},
};

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const PROMPT_CHANNEL_CAPACITY: usize = 32;
const EVENT_CHANNEL_CAPACITY: usize = 1024;

/// Maximum number of prompts batched into a single forward pass.
pub(super) const MAX_BATCH_SIZE: usize = 4;
/// Timeout for individual worker hop responses (self-healing trigger).
const WORKER_HOP_TIMEOUT: Duration = Duration::from_secs(10);
/// How often the run loop checks for worker timeouts.
const TIMEOUT_CHECK_INTERVAL: Duration = Duration::from_millis(250);

mod generation;
mod handlers;
mod helpers;
mod lifecycle;
pub(super) use speculation::NgramDraftCache;

#[derive(Debug, Error)]
pub enum NetworkError {
    #[error("libp2p transport build failed: {0}")]
    Transport(String),
    #[error("mDNS setup failed: {0}")]
    Mdns(#[from] io::Error),
    #[error("listen address failed: {0}")]
    Listen(String),
    #[error("dial failed: {0}")]
    Dial(String),
    #[error("inference failed: {0}")]
    Inference(String),
}

/// Events sent from the node back to the CLI.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Worker has authenticated and is ready for generation.
    WorkerConnected,
    /// A single decoded token was generated.
    Token(String),
    /// One generation round is complete.
    GenerationDone {
        tokens_generated: usize,
        reason: StopReason,
    },
}

#[derive(Debug, Clone)]
pub enum StopReason {
    Eos,
    MaxTokens,
    Error,
}

#[derive(NetworkBehaviour)]
#[behaviour(prelude = "libp2p::swarm::derive_prelude")]
pub struct SynapseBehaviour {
    pub identify: identify::Behaviour,
    pub relay: relay::client::Behaviour,
    /// Relay server — only enabled when running as a public relay node.
    pub relay_server: Toggle<relay::Behaviour>,
    pub dcutr: dcutr::Behaviour,
    pub autonat: autonat::Behaviour,
    pub kad: kad::Behaviour<kad::store::MemoryStore>,
    pub ping: ping::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
    pub request_response: request_response::Behaviour<super::ProtocolCodec>,
}

#[derive(Clone, Debug)]
pub enum SynapseRole {
    Master {
        expected_pairing_code: String,
        max_tokens: usize,
    },
    Worker {
        pairing_code: String,
        manual_master: Option<Multiaddr>,
    },
    /// Public relay + bootstrap node: routes circuits and seeds the DHT.
    RelayServer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerConnectionState {
    Connected,
    Authenticated,
    Disconnected,
    Expired,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WorkerCapabilitySnapshot {
    compute_score: Option<u64>,
    available_ram_mb: Option<u64>,
}

impl WorkerCapabilitySnapshot {
    fn from_hello(hello: &crate::NodeHello) -> Option<Self> {
        let available_ram_mb = hello.available_ram_mb.or_else(|| {
            hello.memory_bytes.and_then(|bytes| {
                let mib = bytes / (1024 * 1024);
                (mib > 0).then_some(mib)
            })
        });
        let snapshot = Self {
            compute_score: hello.compute_score,
            available_ram_mb,
        };
        if snapshot.compute_score.is_none() && snapshot.available_ram_mb.is_none() {
            return None;
        }
        Some(snapshot)
    }

    fn planner_capability(&self) -> Option<NodeCapability> {
        Some(NodeCapability {
            compute_score: self.compute_score?,
            available_ram_mb: self.available_ram_mb?,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct WorkerRegistryEntry {
    peer_id: PeerId,
    state: WorkerConnectionState,
    capability: Option<WorkerCapabilitySnapshot>,
    /// Phase 2: structured `WorkerCapability` from a heartbeat
    /// `CapabilityPing`. The planner reads this when present.
    wire_capability: Option<crate::WorkerCapability>,
    pending_challenge: Option<crate::Challenge>,
    last_seen_at: Instant,
    /// Phase 4: live telemetry from the most recently completed
    /// generation step. `None` until this peer has handled at least
    /// one activation hop.
    layers_assigned: Option<u32>,
    tokens_per_sec: Option<f32>,
    util_pct: Option<f32>,
}

#[derive(Default)]
struct WorkerRegistry {
    entries: HashMap<PeerId, WorkerRegistryEntry>,
}

impl WorkerRegistry {
    fn mark_connected(&mut self, peer_id: PeerId) {
        self.upsert(peer_id, WorkerConnectionState::Connected, None);
    }

    fn mark_authenticated(
        &mut self,
        peer_id: PeerId,
        capability: Option<WorkerCapabilitySnapshot>,
    ) {
        self.upsert(peer_id, WorkerConnectionState::Authenticated, capability);
    }

    fn mark_disconnected(&mut self, peer_id: PeerId) {
        self.upsert(peer_id, WorkerConnectionState::Disconnected, None);
    }

    fn mark_expired(&mut self, peer_id: PeerId) {
        self.upsert(peer_id, WorkerConnectionState::Expired, None);
    }

    #[cfg(test)]
    fn get(&self, peer_id: &PeerId) -> Option<&WorkerRegistryEntry> {
        self.entries.get(peer_id)
    }

    fn capability_for(&self, peer_id: &PeerId) -> Option<WorkerCapabilitySnapshot> {
        self.entries
            .get(peer_id)
            .and_then(|entry| entry.capability.clone())
    }

    /// Structured (Phase 2) capability snapshot for the peer, if
    /// the worker has sent a `CapabilityPing` heartbeat.
    pub fn wire_capability(&self, peer_id: &PeerId) -> Option<crate::WorkerCapability> {
        self.entries
            .get(peer_id)
            .and_then(|entry| entry.wire_capability.clone())
    }

    /// Update the structured capability for a peer. Used by the
    /// `CapabilityPing` heartbeat path.
    pub fn update_capability(&mut self, peer_id: PeerId, cap: crate::WorkerCapability) {
        if let Some(entry) = self.entries.get_mut(&peer_id) {
            entry.wire_capability = Some(cap);
            entry.last_seen_at = Instant::now();
        }
    }

    /// Phase 4: record live per-device telemetry for a worker from the
    /// generation step that just completed. No-op if the peer isn't
    /// registered (e.g. it disconnected mid-step).
    fn update_telemetry(
        &mut self,
        peer_id: PeerId,
        layers_assigned: u32,
        tokens_per_sec: f32,
        util_pct: f32,
    ) {
        if let Some(entry) = self.entries.get_mut(&peer_id) {
            entry.layers_assigned = Some(layers_assigned);
            entry.tokens_per_sec = Some(tokens_per_sec);
            entry.util_pct = Some(util_pct);
        }
    }

    fn upsert(
        &mut self,
        peer_id: PeerId,
        state: WorkerConnectionState,
        capability: Option<WorkerCapabilitySnapshot>,
    ) {
        let now = Instant::now();
        let entry = self
            .entries
            .entry(peer_id)
            .or_insert_with(|| WorkerRegistryEntry {
                peer_id,
                state,
                capability: None,
                wire_capability: None,
                pending_challenge: None,
                last_seen_at: now,
                layers_assigned: None,
                tokens_per_sec: None,
                util_pct: None,
            });
        entry.peer_id = peer_id;
        entry.state = state;
        if state == WorkerConnectionState::Authenticated {
            entry.pending_challenge = None;
        }
        if let Some(capability) = capability {
            entry.capability = Some(capability);
        }
        entry.last_seen_at = now;
    }

    fn set_challenge(&mut self, peer_id: PeerId, challenge: crate::Challenge) {
        if let Some(entry) = self.entries.get_mut(&peer_id) {
            entry.pending_challenge = Some(challenge);
        }
    }

    fn get_challenge(&self, peer_id: &PeerId) -> Option<crate::Challenge> {
        self.entries
            .get(peer_id)
            .and_then(|e| e.pending_challenge.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WorkerRouteHop {
    peer_id: PeerId,
    layer_range: Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GenerationRoutePlan {
    master_layer_range: Range<usize>,
    worker_hops: Vec<WorkerRouteHop>,
}

/// Public, serializable view of a `GenerationRoutePlan` for UIs.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RoutePlanView {
    pub master_layer_range: std::ops::Range<usize>,
    pub worker_hops: Vec<RouteHopView>,
}

/// Public, serializable view of a single `WorkerRouteHop`.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RouteHopView {
    pub peer_id: String,
    pub layer_range: std::ops::Range<usize>,
}

/// Live snapshot of the node's networking state, published by the run
/// loop through a `watch` channel so the GUI can render real peers
/// instead of placeholder data. Cheap to clone (small Vec + a couple of
/// Options); the `watch` channel only ever retains the latest value.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct NodeStatusSnapshot {
    /// Every worker the registry knows about, with connection state.
    pub peers: Vec<PeerSummary>,
    /// Number of workers that have fully authenticated (challenge-response).
    pub authenticated_count: usize,
    /// The active generation route, if a turn is in progress.
    pub route_plan: Option<RoutePlanView>,
    /// Reconnection status (worker role only).
    pub redial: Option<RedialStatus>,
    /// Stable role label: "master" | "worker" | "relay".
    pub role: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct InFlightActivationHop {
    pub(super) sequence_position: usize,
    pub(super) hop_index: usize,
    pub(super) trace_id: String,
    pub(super) dispatch_time: Option<Instant>,
    /// Which compute pass this hop belongs to; workers need it to pick
    /// the right KV-cache strategy (replace / single append / block append).
    pub(super) pass: ActivationPass,
}

/// How a hidden-state activation should be executed against the KV cache.
///
/// Derived from the request's `sequence_position` and the activation's
/// sequence length:
/// - position 0 → [`ActivationPass::Prefill`] (cache replace, full prompt)
/// - later positions with a single token → [`ActivationPass::Decode`]
/// - later positions with multiple tokens → [`ActivationPass::Speculative`]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ActivationPass {
    Prefill,
    Decode,
    Speculative,
}

impl ActivationPass {
    fn for_sequence_position(sequence_position: usize) -> Self {
        if sequence_position == 0 {
            Self::Prefill
        } else {
            Self::Decode
        }
    }

    /// Classify an inbound `ActivationRequest`. Multi-token activations at
    /// non-zero sequence positions are speculative verification passes that
    /// must append to (not replace) the worker's KV caches.
    pub(crate) fn for_request(sequence_position: usize, seq_len: usize) -> Self {
        if sequence_position == 0 {
            Self::Prefill
        } else if seq_len <= 1 {
            Self::Decode
        } else {
            Self::Speculative
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Speculative => "speculative",
        }
    }
}

/// A speculative verification round in flight.
///
/// The master dispatched `[last_committed_token, draft_1, ..., draft_k]`
/// through the route and is awaiting the multi-position activation output
/// to verify the drafts. On verification the cache must hold exactly
/// `cache_base_len + accepted + 1` positions — anything beyond that came
/// from rejected drafts and is truncated away.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SpeculativeRound {
    /// Drafted tokens `[d1..dk]` being verified.
    pub(crate) draft_tokens: Vec<u32>,
    /// Master/worker KV cache length *before* this round appended its
    /// verification block.
    pub(crate) cache_base_len: usize,
}

// ─── Batching types ──────────────────────────────────────────────────────────

/// A prompt request submitted to the batch queue.
#[derive(Debug)]
pub struct BatchPromptRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub event_tx: mpsc::Sender<ChatEvent>,
}

/// Tracks a single in-flight sequence within a batch.
struct InFlightRequest {
    /// Monotonic request ID for routing.
    request_id: u64,
    /// Token IDs accumulated so far.
    token_ids: Vec<u32>,
    /// Max tokens for this request.
    max_tokens: usize,
    /// Per-request event channel back to the originating client.
    event_tx: mpsc::Sender<ChatEvent>,
}

pub struct SynapseNode {
    role: SynapseRole,
    model: ModelIdentity,
    swarm: Swarm<SynapseBehaviour>,
    dialed_peers: HashSet<PeerId>,
    greeted_peers: HashSet<PeerId>,
    authenticated_workers: HashSet<PeerId>,
    worker_peers: Vec<PeerId>,
    worker_registry: WorkerRegistry,
    local_capability: Option<WorkerCapabilitySnapshot>,
    exchange_complete: bool,
    /// Shared model reference — loaded after download, before network exchange.
    pipeline: Option<Arc<ToyLlama>>,
    /// Tokenizer for encoding/decoding text.
    tokenizer: Option<Arc<SimpleTokenizer>>,
    /// Cached master range end for the active route plan.
    split_point: usize,
    total_layers: usize,
    // ─── Master generation state ─────────────────────────────────────────
    /// Current token sequence (grows with each generated token).
    token_ids: Vec<u32>,
    /// How many new tokens we've generated so far in the current turn.
    tokens_generated: usize,
    /// Whether a generation round is currently active for the master.
    generation_active: bool,
    /// Whether master is waiting for worker ClearCacheAck before first activation.
    awaiting_clear_cache_ack: bool,
    /// Trace id for the in-flight clear-cache request (if any).
    pending_cache_reset_trace_id: Option<String>,
    /// Workers still expected to acknowledge the in-flight cache-reset request.
    pending_cache_reset_peers: HashSet<PeerId>,
    /// Route plan locked for the active generation turn.
    generation_route_plan: Option<GenerationRoutePlan>,
    /// In-flight worker hop awaiting ActivationResponse.
    in_flight_activation_hop: Option<InFlightActivationHop>,
    /// Monotonic nonce for clear-cache trace ids.
    cache_reset_nonce: u64,
    // ─── Chat channels ───────────────────────────────────────────────────
    /// Receives prompts from the CLI (master only).
    prompt_rx: Option<mpsc::Receiver<String>>,
    /// Sends chat events back to the CLI (master only).
    event_tx: Option<mpsc::Sender<ChatEvent>>,
    // ─── Continuous Batching (Phase 8) ───────────────────────────────────
    /// Unbounded queue for incoming prompt requests from multiple clients.
    batch_queue_rx: Option<mpsc::UnboundedReceiver<BatchPromptRequest>>,
    /// Active in-flight requests being processed as a batch.
    active_batch: VecDeque<InFlightRequest>,
    /// Monotonic counter for assigning request IDs.
    next_request_id: u64,
    /// Max tokens override for the currently-processing batch request.
    batch_active_max_tokens: Option<usize>,
    // ─── Self-Healing (Phase 9) ──────────────────────────────────────────
    /// Whether we are in self-healing mode (replaying prefill through new route).
    healing_in_progress: bool,
    /// Token IDs saved before healing replay (restored on completion).
    healing_saved_tokens: Option<Vec<u32>>,
    /// Generation step saved before healing replay.
    healing_saved_step: usize,
    // ─── Speculative Decoding (Phase 10) ─────────────────────────────────
    /// N-gram draft cache for predicting next tokens.
    draft_cache: speculation::NgramDraftCache,
    /// In-flight speculative draft tokens awaiting verification.
    pending_draft: Option<Vec<u32>>,
    /// Whether speculative decoding is enabled for this session.
    speculative_enabled: bool,
    // ─── Security ────────────────────────────────────────────────────────
    /// Rate limiter for pairing code brute-force protection.
    pairing_rate_limiter: PairingRateLimiter,
    // ─── Reconnection ────────────────────────────────────────────────────
    /// Number of consecutive redial attempts to the manual master.
    redial_attempts: u32,
    /// When the next redial attempt is allowed (backoff).
    next_redial_at: Option<Instant>,
}

/// Public summary of a worker peer for UIs / dashboards.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PeerSummary {
    pub peer_id: String,
    pub state: String,
    pub compute_score: Option<u64>,
    pub available_ram_mb: Option<u64>,
    pub last_seen_unix_ms: u64,
}

/// Reconnection state for a worker waiting for its master.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RedialStatus {
    pub attempts: u32,
    pub next_redial_in_ms: Option<u64>,
}

impl SynapseNode {
    /// Snapshot of the worker registry, suitable for display in a GUI.
    pub fn peers(&self) -> Vec<PeerSummary> {
        let now = Instant::now();
        let unix_epoch_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.worker_registry
            .entries
            .values()
            .map(|entry| {
                let last_seen_unix_ms = unix_epoch_ms.saturating_sub(
                    now.saturating_duration_since(entry.last_seen_at)
                        .as_millis() as u64,
                );
                PeerSummary {
                    peer_id: entry.peer_id.to_string(),
                    state: match entry.state {
                        WorkerConnectionState::Connected => "connected",
                        WorkerConnectionState::Authenticated => "authenticated",
                        WorkerConnectionState::Disconnected => "disconnected",
                        WorkerConnectionState::Expired => "expired",
                    }
                    .to_owned(),
                    compute_score: entry.capability.as_ref().and_then(|c| c.compute_score),
                    available_ram_mb: entry.capability.as_ref().and_then(|c| c.available_ram_mb),
                    last_seen_unix_ms,
                }
            })
            .collect()
    }

    /// The current generation route plan (master only), if one is active.
    pub fn route_plan(&self) -> Option<RoutePlanView> {
        self.generation_route_plan
            .as_ref()
            .map(|plan| RoutePlanView {
                master_layer_range: plan.master_layer_range.clone(),
                worker_hops: plan
                    .worker_hops
                    .iter()
                    .map(|hop| RouteHopView {
                        peer_id: hop.peer_id.to_string(),
                        layer_range: hop.layer_range.clone(),
                    })
                    .collect(),
            })
    }

    /// Current reconnection status (worker only).
    pub fn redial_status(&self) -> RedialStatus {
        let next_in_ms = self.next_redial_at.and_then(|when| {
            let now = Instant::now();
            if when <= now {
                None
            } else {
                Some((when - now).as_millis() as u64)
            }
        });
        RedialStatus {
            attempts: self.redial_attempts,
            next_redial_in_ms: next_in_ms,
        }
    }

    /// Return the role this node was created with, as a stable string.
    pub fn role_label(&self) -> &'static str {
        match self.role {
            SynapseRole::Master { .. } => "master",
            SynapseRole::Worker { .. } => "worker",
            SynapseRole::RelayServer => "relay",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WorkerCapabilitySnapshot, WorkerConnectionState, WorkerRegistry};
    use libp2p::PeerId;

    #[test]
    fn worker_registry_inserts_and_updates_capabilities() {
        let mut registry = WorkerRegistry::default();
        let peer_id = PeerId::random();

        registry.mark_connected(peer_id);
        let connected = registry
            .get(&peer_id)
            .expect("connected worker should exist")
            .clone();
        assert_eq!(connected.peer_id, peer_id);
        assert_eq!(connected.state, WorkerConnectionState::Connected);
        assert!(connected.capability.is_none());

        registry.mark_authenticated(
            peer_id,
            Some(WorkerCapabilitySnapshot {
                compute_score: Some(1700),
                available_ram_mb: Some(8192),
            }),
        );
        let authenticated = registry
            .get(&peer_id)
            .expect("authenticated worker should exist")
            .clone();
        assert_eq!(authenticated.state, WorkerConnectionState::Authenticated);
        assert_eq!(
            authenticated.capability,
            Some(WorkerCapabilitySnapshot {
                compute_score: Some(1700),
                available_ram_mb: Some(8192),
            })
        );
        assert!(authenticated.last_seen_at >= connected.last_seen_at);

        registry.mark_authenticated(
            peer_id,
            Some(WorkerCapabilitySnapshot {
                compute_score: Some(1900),
                available_ram_mb: Some(12288),
            }),
        );
        let refreshed = registry
            .get(&peer_id)
            .expect("refreshed worker should still exist");
        assert_eq!(
            refreshed.capability,
            Some(WorkerCapabilitySnapshot {
                compute_score: Some(1900),
                available_ram_mb: Some(12288),
            })
        );
        assert!(refreshed.last_seen_at >= authenticated.last_seen_at);
    }

    #[test]
    fn worker_registry_marks_workers_stale_on_disconnect() {
        let mut registry = WorkerRegistry::default();
        let peer_id = PeerId::random();

        registry.mark_authenticated(
            peer_id,
            Some(WorkerCapabilitySnapshot {
                compute_score: Some(900),
                available_ram_mb: Some(4096),
            }),
        );
        registry.mark_disconnected(peer_id);

        let disconnected = registry
            .get(&peer_id)
            .expect("disconnected worker should remain in registry");
        assert_eq!(disconnected.state, WorkerConnectionState::Disconnected);
        assert_eq!(
            disconnected.capability,
            Some(WorkerCapabilitySnapshot {
                compute_score: Some(900),
                available_ram_mb: Some(4096),
            })
        );
    }
}
