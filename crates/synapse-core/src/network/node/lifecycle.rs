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
use crate::inference::{
    ShardedModel, SimpleTokenizer, ToyLlama, ToyLlamaConfig, ToyLlamaGGUF, gguf_layer_count,
};
use crate::{ModelIdentity, NetConfig, NodeIdentity, PairingRateLimiter};
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

        let revocations = crate::identity::RevocationList::load(
            &identity.key_path().with_extension("revocations.json"),
        )
        .map_err(|error| NetworkError::Transport(error.to_string()))?;
        let memory_limit_bytes = local_capability
            .as_ref()
            .and_then(|cap| cap.available_ram_mb)
            .unwrap_or(0)
            .saturating_mul(1024 * 1024)
            / 2;
        Ok(Self {
            identity,
            revocations,
            pending_coordinator_challenges: Default::default(),
            model_files: None,
            model_config: None,
            generation_lease: None,
            shard_readiness: None,
            worker_assignment: None,
            route_epoch: 0,
            memory_limit_bytes,
            coordinator_only: false,
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

    /// Resolve and verify a model, retaining metadata until a complete route
    /// assigns a concrete range. This never loads a hidden full host model.
    pub async fn load_model(&mut self) -> Result<(), NetworkError> {
        let selected_path = std::path::Path::new(&self.model.id);
        let files = if selected_path.is_dir() {
            crate::inference::model_loader::ModelFiles::from_directory(selected_path)
        } else {
            crate::inference::model_loader::download_model_revision(&self.model).await
        }
        .map_err(|error| NetworkError::Inference(error.to_string()))?;
        self.install_model_metadata(files)
    }

    pub fn load_model_from_directory(
        &mut self,
        directory: &std::path::Path,
    ) -> Result<(), NetworkError> {
        let files = crate::inference::model_loader::ModelFiles::from_directory(directory)
            .map_err(|error| NetworkError::Inference(error.to_string()))?;
        self.install_model_metadata(files)
    }

    fn install_model_metadata(
        &mut self,
        files: crate::inference::model_loader::ModelFiles,
    ) -> Result<(), NetworkError> {
        let config = ToyLlamaConfig::from_file(&files.config)
            .map_err(|error| NetworkError::Inference(error.to_string()))?;
        let is_gguf = files
            .weights
            .extension()
            .is_some_and(|extension| extension == "gguf");
        let total = if is_gguf {
            gguf_layer_count(&files.weights)
                .map_err(|error| NetworkError::Inference(error.to_string()))?
        } else {
            config.num_hidden_layers
        };
        let tokenizer = SimpleTokenizer::from_file(&files.tokenizer)
            .map_err(|error| NetworkError::Inference(error.to_string()))?;
        let digest = files
            .manifest
            .digest()
            .map_err(|error| NetworkError::Inference(error.to_string()))?;
        self.model = ModelIdentity {
            id: files.manifest.model_id.clone(),
            revision: Some(files.manifest.revision.clone()),
            sha256: Some(digest),
        };
        self.pipeline = None;
        self.tokenizer = Some(std::sync::Arc::new(tokenizer));
        self.total_layers = total;
        self.model_config = Some(config);
        self.model_files = Some(files);
        let device = crate::inference::get_best_device()
            .map_err(|error| NetworkError::Inference(error.to_string()))?;
        let score = crate::inference::run_micro_benchmark(&device);
        self.measured_compute_score = Some(score);
        if let Some(capability) = self.local_capability.as_mut() {
            capability.compute_score = Some(score);
            capability.available_ram_mb = Some(self.memory_limit_bytes / (1024 * 1024));
        }
        info!(total_layers=total, model_digest=?self.model.sha256, "model artifacts verified; awaiting shard assignment");
        Ok(())
    }

    /// Set an explicit inference budget before run; zero/unknown memory fails
    /// closed. The budget includes weights, cache and activation scratch.
    pub fn set_memory_limit_bytes(&mut self, bytes: u64) -> Result<(), NetworkError> {
        if bytes == 0 {
            return Err(NetworkError::Inference(
                "memory budget must be positive".into(),
            ));
        }
        if self.generation_active || self.worker_assignment.is_some() {
            return Err(NetworkError::Inference(
                "cannot change budget during active work".into(),
            ));
        }
        self.memory_limit_bytes = bytes;
        if let Some(capability) = self.local_capability.as_mut() {
            capability.available_ram_mb = Some(bytes / (1024 * 1024));
        }
        Ok(())
    }

    pub fn set_coordinator_only(&mut self, enabled: bool) -> Result<(), NetworkError> {
        if self.generation_active || !matches!(self.role, SynapseRole::Master { .. }) {
            return Err(NetworkError::Inference(
                "coordinator-only mode is a pre-run host setting".into(),
            ));
        }
        self.coordinator_only = enabled;
        Ok(())
    }

    pub(super) fn load_assigned_range(
        &mut self,
        range: std::ops::Range<usize>,
    ) -> anyhow::Result<()> {
        let files = self
            .model_files
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model metadata not loaded"))?;
        let config = self
            .model_config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model config not loaded"))?;
        anyhow::ensure!(
            range.start <= range.end && range.end <= self.total_layers,
            "invalid model range"
        );
        // Release old storage before loading a replacement, avoiding a hidden
        // second copy during replanning.
        self.pipeline = None;
        let roles = if matches!(self.role, SynapseRole::Master { .. }) {
            crate::inference::pipeline::TensorRoles::COORDINATOR
        } else {
            crate::inference::pipeline::TensorRoles::WORKER
        };
        let model: std::sync::Arc<dyn ShardedModel> = if files
            .weights
            .extension()
            .is_some_and(|extension| extension == "gguf")
        {
            let device = crate::inference::get_best_device()?;
            std::sync::Arc::new(ToyLlamaGGUF::load_range_with_roles(
                &files.weights,
                range.clone(),
                &device,
                roles,
            )?)
        } else {
            std::sync::Arc::new(ToyLlama::load_range_with_roles(
                config,
                &files.weights,
                range.clone(),
                roles,
            )?)
        };
        anyhow::ensure!(
            model.layer_offset() == range.start && model.owned_layers() == range.len(),
            "resident layer range differs from assignment"
        );
        anyhow::ensure!(
            model.resident_weight_bytes() <= self.memory_limit_bytes,
            "resident weights exceed memory cap"
        );
        info!(
            layer_start = range.start,
            layer_end = range.end,
            resident_weight_bytes = model.resident_weight_bytes(),
            "assigned shard resident"
        );
        self.pipeline = Some(model);
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
                // Check peer cache first before DHT query.
                let cached_providers = self.peer_cache.get_providers(&self.model.id);
                if !cached_providers.is_empty() {
                    info!(
                        count = cached_providers.len(),
                        "found cached providers, dialing directly"
                    );
                    let cached_ids: Vec<_> = cached_providers.iter().map(|(p, _)| *p).collect();
                    // Phase 6: pre-warm connection pool with cached masters.
                    self.connection_pool.prewarm(&cached_ids);
                    for (peer_id, addrs) in cached_providers {
                        for addr in addrs {
                            info!(%peer_id, %addr, "dialing cached master");
                            if let Err(error) = self.swarm.dial(addr.clone()) {
                                warn!(%peer_id, %addr, %error, "failed to dial cached master");
                            }
                        }
                    }
                } else {
                    info!(
                        "no cached providers, searching DHT for masters serving model {}",
                        self.model.id
                    );
                    self.swarm.behaviour_mut().kad.get_providers(model_key);
                }
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

                    // Peer/auth state changes happen inside swarm events;
                    // publish a fresh snapshot so the GUI updates promptly.
                    self.publish_status();
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
                    // Periodic refresh: catches redial countdowns and keeps
                    // the GUI's peer panel live even when idle.
                    self.publish_status();
                }
            }
        }
    }
}
