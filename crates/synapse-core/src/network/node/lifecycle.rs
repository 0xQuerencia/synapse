use futures::StreamExt;
use libp2p::{
    Multiaddr, SwarmBuilder, autonat, dcutr, identify, kad, mdns, ping, relay,
    request_response::{self, ProtocolSupport},
    swarm::behaviour::toggle::Toggle,
};
use tokio::sync::mpsc;
use tracing::{info, warn};

use super::{
    BatchPromptRequest, ChatEvent, DEFAULT_REQUEST_TIMEOUT, EVENT_CHANNEL_CAPACITY, NetworkError,
    PROMPT_CHANNEL_CAPACITY, SynapseBehaviour, SynapseNode, SynapseRole, TIMEOUT_CHECK_INTERVAL,
    WorkerRegistry,
    helpers::{
        extract_peer_id, local_worker_capability_snapshot, measure_relay_latencies,
        recv_batch_prompt, recv_prompt, select_best_relay,
    },
};
use crate::{ModelIdentity, NetConfig, NodeIdentity, PairingRateLimiter};
use crate::{
    compute_score::blend_compute_score,
    inference::{
        ShardedModel, SimpleTokenizer, ToyLlama, ToyLlamaConfig, ToyLlamaGGUF, download_model,
        gguf_layer_count, verify_file_sha256,
    },
};
use std::collections::VecDeque;

impl SynapseNode {
    pub async fn new(
        role: SynapseRole,
        model: ModelIdentity,
        listen_addr: Multiaddr,
    ) -> Result<Self, NetworkError> {
        let identity = NodeIdentity::load_or_create()
            .map_err(|e| NetworkError::Transport(format!("identity error: {e}")))?;
        info!(peer_id = %identity.peer_id(), "node identity loaded");

        let is_relay_server = matches!(role, SynapseRole::RelayServer);
        let mut swarm = SwarmBuilder::with_existing_identity(identity.keypair().clone())
            .with_tokio()
            .with_tcp(
                libp2p::tcp::Config::default(),
                libp2p::noise::Config::new,
                libp2p::yamux::Config::default,
            )
            .map_err(|error| NetworkError::Transport(error.to_string()))?
            .with_quic()
            // Relay client transport — REQUIRED for circuit dialing/listening.
            // Previously `relay::client::new()` was called and its transport
            // discarded, so circuits never routed. The builder wires the
            // transport in properly.
            .with_relay_client(libp2p::noise::Config::new, libp2p::yamux::Config::default)
            .map_err(|error| NetworkError::Transport(error.to_string()))?
            .with_behaviour(|keypair, relay_client| {
                let local_peer_id = libp2p::PeerId::from(keypair.public());

                // Identify — exchanges peer info, required by relay/DCUtR.
                let identify = identify::Behaviour::new(
                    identify::Config::new(crate::PROTOCOL_VERSION.to_string(), keypair.public())
                        .with_agent_version(format!("synapse/{}", env!("CARGO_PKG_VERSION"))),
                );

                // Relay client behaviour — paired with the transport above.
                let relay = relay_client;

                // Relay server — only active for a dedicated public relay node.
                let relay_server = Toggle::from(
                    is_relay_server
                        .then(|| relay::Behaviour::new(local_peer_id, relay::Config::default())),
                );

                // DCUtR — hole punches through NATs after relay connection.
                let dcutr = dcutr::Behaviour::new(local_peer_id);

                // AutoNAT — detects if we're behind a NAT.
                let autonat = autonat::Behaviour::new(local_peer_id, autonat::Config::default());

                // Kademlia DHT — internet-wide peer discovery.
                let kad_store = kad::store::MemoryStore::new(local_peer_id);
                let mut kad = kad::Behaviour::new(local_peer_id, kad_store);
                kad.set_mode(Some(kad::Mode::Server));

                // Ping — keepalive and liveness checks.
                let ping = ping::Behaviour::new(ping::Config::default());

                // mDNS — LAN fast-path discovery.
                let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)?;

                // Request-response — the core Synapse protocol.
                let request_response = request_response::Behaviour::with_codec(
                    super::super::ProtocolCodec,
                    [(super::super::SynapseProtocol, ProtocolSupport::Full)],
                    request_response::Config::default()
                        .with_request_timeout(DEFAULT_REQUEST_TIMEOUT)
                        .with_max_concurrent_streams(128), // Increased for pipelining support
                );

                Ok(SynapseBehaviour {
                    identify,
                    relay,
                    relay_server,
                    dcutr,
                    autonat,
                    kad,
                    ping,
                    mdns,
                    request_response,
                })
            })
            .map_err(|error| NetworkError::Transport(error.to_string()))?
            .with_swarm_config(|cfg| {
                cfg.with_idle_connection_timeout(std::time::Duration::from_secs(5 * 60))
            })
            .build();

        swarm
            .listen_on(listen_addr)
            .map_err(|error| NetworkError::Listen(error.to_string()))?;

        let local_capability = local_worker_capability_snapshot();
        let peer_cache = super::super::PeerCache::load();

        // Live status channel: the run loop publishes peer/route snapshots
        // here so a GUI can render real peer data without owning the node.
        let (status_tx, _status_rx) =
            tokio::sync::watch::channel(super::NodeStatusSnapshot::default());

        Ok(Self {
            role,
            model,
            swarm,
            dialed_peers: std::collections::HashSet::new(),
            greeted_peers: std::collections::HashSet::new(),
            authenticated_workers: std::collections::HashSet::new(),
            worker_peers: Vec::new(),
            worker_registry: WorkerRegistry::default(),
            local_capability,
            peer_cache,
            exchange_complete: false,
            pipeline: None,
            tokenizer: None,
            split_point: 0,
            total_layers: 0,
            token_ids: Vec::new(),
            tokens_generated: 0,
            generation_active: false,
            awaiting_clear_cache_ack: false,
            pending_cache_reset_trace_id: None,
            pending_cache_reset_peers: std::collections::HashSet::new(),
            generation_route_plan: None,
            in_flight_activation_hop: None,
            cache_reset_nonce: 0,
            truncate_nonce: 0,
            prompt_rx: None,
            event_tx: None,
            // Phase 8: Continuous Batching
            batch_queue_rx: None,
            active_batch: VecDeque::new(),
            next_request_id: 0,
            batch_active_max_tokens: None,
            // Phase 9: Self-Healing
            healing_in_progress: false,
            healing_saved_tokens: None,
            healing_saved_step: 0,
            // Phase 10: Speculative Decoding
            draft_cache: super::NgramDraftCache::default(),
            spec_round: None,
            speculative_enabled: true,
            kv_cache_len: 0,
            awaiting_truncate_ack: false,
            pending_truncate_trace_id: None,
            pending_truncate_peers: std::collections::HashSet::new(),
            pairing_rate_limiter: PairingRateLimiter::default(),
            redial_attempts: 0,
            next_redial_at: None,
            status_tx,
            measured_compute_score: None,
            step_started_at: None,
            step_hop_durations_ms: std::collections::HashMap::new(),
            worker_latencies: std::collections::HashMap::new(),
            hop_latencies: std::collections::HashMap::new(),
            // Phase 5: per-worker batch calibration starts empty.
            batch_profiles: std::collections::HashMap::new(),
            // Phase 6: pre-warmed connection pool.
            connection_pool: super::super::ConnectionPool::with_capacity(64),
        })
    }

    /// Create chat channels for interactive mode (master only).
    ///
    /// Returns `(prompt_sender, event_receiver)`. The CLI sends prompts via
    /// `prompt_sender`, and reads `ChatEvent`s from `event_receiver`.
    pub fn chat_channels(&mut self) -> (mpsc::Sender<String>, mpsc::Receiver<ChatEvent>) {
        let (prompt_tx, prompt_rx) = mpsc::channel(PROMPT_CHANNEL_CAPACITY);
        let (event_tx, event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        self.prompt_rx = Some(prompt_rx);
        self.event_tx = Some(event_tx);
        (prompt_tx, event_rx)
    }

    /// Create a batch prompt queue for concurrent multi-client usage.
    ///
    /// Returns an unbounded sender that any client (CLI, HTTP, WebSocket) can
    /// use to submit prompt requests. Each request carries its own event channel
    /// for receiving per-token streaming responses.
    pub fn batch_queue_sender(&mut self) -> mpsc::UnboundedSender<BatchPromptRequest> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.batch_queue_rx = Some(rx);
        tx
    }

    /// Load the model weights and tokenizer. Must be called before `run()`.
    pub async fn load_model(&mut self) -> Result<(), NetworkError> {
        let files = download_model(&self.model.id)
            .await
            .map_err(|error| NetworkError::Inference(error.to_string()))?;

        if let Some(expected_sha256) = &self.model.sha256 {
            verify_file_sha256(&files.weights, expected_sha256).map_err(|error| {
                NetworkError::Inference(format!("model weight verification failed: {error}"))
            })?;
        }

        let is_gguf = files.weights.extension().is_some_and(|ext| ext == "gguf");

        // GGUF files carry all model metadata in their header; safetensors
        // models use the HF `config.json`. Only the layer count is needed
        // here to compute the static shard plan.
        let config = if is_gguf {
            None
        } else {
            Some(
                ToyLlamaConfig::from_file(&files.config)
                    .map_err(|error| NetworkError::Inference(error.to_string()))?,
            )
        };
        let total = match &config {
            Some(config) => config.num_hidden_layers,
            None => gguf_layer_count(&files.weights)
                .map_err(|error| NetworkError::Inference(error.to_string()))?,
        };
        // The master loads the full layer span. Generation replans shards
        // dynamically (capability-weighted) as workers join and leave, so a
        // fixed placeholder split here would leave the master holding fewer
        // layers than a later route plan assigns it. A full-span load is
        // valid for any plan; workers slice their own range on join.
        let split = match &self.role {
            SynapseRole::Master { .. } => total,
            SynapseRole::Worker { .. } | SynapseRole::RelayServer => total,
        };

        let device = crate::inference::get_best_device()
            .map_err(|error| NetworkError::Inference(error.to_string()))?;

        // Slice the model by role: the master holds embed + all layers +
        // the LM head (full span, see note above); workers hold the upper
        // layers. RelayServer never loads a model (empty range).
        let layer_range = match &self.role {
            SynapseRole::Master { .. } => 0..split,
            SynapseRole::Worker { .. } => split..total,
            SynapseRole::RelayServer => 0..0,
        };

        let model: std::sync::Arc<dyn ShardedModel> = if is_gguf {
            std::sync::Arc::new(
                ToyLlamaGGUF::load_range(&files.weights, layer_range, &device)
                    .map_err(|error| NetworkError::Inference(error.to_string()))?,
            )
        } else {
            std::sync::Arc::new(
                ToyLlama::load_range(
                    config.as_ref().expect("safetensors config loaded above"),
                    &files.weights,
                    layer_range,
                )
                .map_err(|error| NetworkError::Inference(error.to_string()))?,
            )
        };

        let tokenizer = SimpleTokenizer::from_file(&files.tokenizer)
            .map_err(|error| NetworkError::Inference(error.to_string()))?;

        self.pipeline = Some(model);
        self.tokenizer = Some(std::sync::Arc::new(tokenizer));
        self.split_point = split;
        self.total_layers = total;

        info!(
            total_layers = total,
            split_point = split,
            "model and tokenizer loaded, shard plan computed"
        );

        // ─── Compute real compute score (Phase 1) ─────────────────
        // Run the micro-benchmark once and cache the result so the
        // worker advertises its real throughput, not cpu_threads*100.
        let measured = crate::inference::run_micro_benchmark(&device);
        let probe = crate::hardware::HardwareProbe::detect();
        let ram_free_mb = super::helpers::detect_available_ram_mb().unwrap_or(0);
        let score = blend_compute_score(measured, ram_free_mb, &probe);
        self.measured_compute_score = Some(score);
        info!(
            measured,
            ram_free_mb,
            blended_score = score,
            backend = %probe.backend.as_str(),
            gpu = probe.has_accelerator,
            "compute score set from real benchmark"
        );

        // Also update the local capability snapshot so the master's
        // own peer view shows the measured score, not cpu_threads*100.
        if let Some(snapshot) = self.local_capability.as_mut() {
            snapshot.compute_score = Some(score);
            snapshot.available_ram_mb = Some(ram_free_mb);
        }

        Ok(())
    }

    pub fn local_peer_id(&self) -> libp2p::PeerId {
        *self.swarm.local_peer_id()
    }

    pub async fn run(mut self) -> Result<(), NetworkError> {
        info!(peer_id = %self.local_peer_id(), "Synapse swarm started");

        // Load bootstrap + relay peers from config/*.toml (no hardcoded panics).
        let net_config = NetConfig::load_default();
        for addr in &net_config.bootstrap {
            if let Some(peer_id) = extract_peer_id(addr) {
                info!(%peer_id, %addr, "adding DHT bootstrap peer");
                self.swarm
                    .behaviour_mut()
                    .kad
                    .add_address(&peer_id, addr.clone());
            }
        }

        if let Err(e) = self.swarm.behaviour_mut().kad.bootstrap() {
            warn!(
                "DHT bootstrap failed: {} (no bootstrap peers configured?)",
                e
            );
        }

        // Dial relays and (for NAT'd masters) reserve a circuit so remote peers
        // can reach us through the relay. The relay server itself skips this.
        if !matches!(self.role, SynapseRole::RelayServer) {
            // Geographic relay selection: measure latency and prefer fastest.
            let relay_infos = measure_relay_latencies(&mut self.swarm, &net_config.relays);

            // Masters listen on circuit address via the best relay.
            if matches!(self.role, SynapseRole::Master { .. }) {
                if let Some(best_relay) = select_best_relay(&relay_infos) {
                    let circuit: Multiaddr = best_relay
                        .addr
                        .clone()
                        .with(libp2p::multiaddr::Protocol::P2pCircuit);
                    match self.swarm.listen_on(circuit.clone()) {
                        Ok(_) => {
                            info!(
                               circuit = %circuit,
                                 region = ?best_relay.region,
                                latency_ms = ?best_relay.measured_latency_ms,
                            "listening on best relay circuit"
                               );
                        }
                        Err(error) => {
                            warn!(%circuit, %error, "failed to listen on circuit");
                        }
                    }
                } else {
                    warn!("no relay available for circuit listening");
                }
            }
        }

        let model_key = libp2p::kad::RecordKey::new(&self.model.id.as_bytes());

        match &self.role {
            // Masters advertise that they serve this model so workers can find them.
            SynapseRole::Master { .. } => {
                if let Err(e) = self
                    .swarm
                    .behaviour_mut()
                    .kad
                    .start_providing(model_key.clone())
                {
                    warn!("DHT start_providing failed: {}", e);
                }
            }
            // Workers without a manual master discover one via the DHT.
            SynapseRole::Worker {
                manual_master: None,
                ..
            } => {
                info!("searching DHT for masters serving model {}", self.model.id);
                self.swarm.behaviour_mut().kad.get_providers(model_key);
            }
            _ => {}
        }

        if let SynapseRole::Worker {
            manual_master: Some(addr),
            ..
        } = &self.role
        {
            info!(address = %addr, "dialing manual master address");
            self.swarm
                .dial(addr.clone())
                .map_err(|error| NetworkError::Dial(error.to_string()))?;
        }

        let mut timeout_interval = tokio::time::interval(TIMEOUT_CHECK_INTERVAL);
        timeout_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Build a combined shutdown signal (SIGINT + SIGTERM on Unix).
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
        let shutdown_tx_ctrlc = shutdown_tx.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            info!("SIGINT received");
            let _ = shutdown_tx_ctrlc.send(()).await;
        });
        #[cfg(unix)]
        {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to register SIGTERM handler");
            tokio::spawn(async move {
                sigterm.recv().await;
                info!("SIGTERM received");
                let _ = shutdown_tx.send(()).await;
            });
        }

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("initiating graceful shutdown");
                    self.shutdown_cleanup();
                    return Ok(());
                }
                event = self.swarm.select_next_some() => {
                    self.handle_swarm_event(event).await?;

                    if self.exchange_complete {
                        match &self.role {
                            SynapseRole::Master { .. } => {
                                info!("master shutting down");
                            }
                            SynapseRole::Worker { .. } => {
                                info!("worker session ended");
                            }
                            SynapseRole::RelayServer => {
                                info!("relay server stopping");
                            }
                        }
                        return Ok(());
                    }
                }
                // Listen for prompts from the CLI REPL (master only)
                Some(prompt) = recv_prompt(&mut self.prompt_rx) => {
                    self.handle_new_prompt(prompt);
                }
                // Listen for batch queue prompts from multiple clients
                Some(batch_req) = recv_batch_prompt(&mut self.batch_queue_rx) => {
                    self.handle_batch_prompt_request(batch_req);
                }
                // Periodic timeout check for self-healing (Phase 9)
                _ = timeout_interval.tick() => {
                    self.check_worker_timeouts();
                }
            }
        }
    }
}
