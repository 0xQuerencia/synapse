use libp2p::{
    PeerId, identify, kad, mdns, ping,
    request_response::{self, Event as RequestResponseEvent, Message as RequestResponseMessage},
    swarm::SwarmEvent,
};
use tracing::{debug, info, warn};

use super::{
    ChatEvent, NetworkError, StopReason, SynapseBehaviourEvent, SynapseNode, SynapseRole,
    WorkerCapabilitySnapshot,
};
use crate::{NodeHello, NodeRole};
use std::time::{Duration, Instant};

impl SynapseNode {
    pub(super) async fn handle_swarm_event(
        &mut self,
        event: SwarmEvent<SynapseBehaviourEvent>,
    ) -> Result<(), NetworkError> {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                info!(peer_id = %self.local_peer_id(), %address, "listening on QUIC address");
            }
            SwarmEvent::ConnectionEstablished {
                peer_id, endpoint, ..
            } => {
                if self.revocations.contains(&peer_id) {
                    let _ = self.swarm.disconnect_peer_id(peer_id);
                    return Ok(());
                }
                let is_outgoing = endpoint.is_dialer();
                info!(%peer_id, %is_outgoing, "connection established");

                if matches!(self.role, SynapseRole::Master { .. }) {
                    self.worker_registry.mark_connected(peer_id);
                }

                // Workers only send NodeHello on outgoing connections (i.e.,
                // connections the worker initiated to a master). This prevents
                // leaking the pairing code to other workers or unrelated peers
                // discovered via mDNS.
                if matches!(self.role, SynapseRole::Worker { .. }) && is_outgoing {
                    self.send_worker_hello(peer_id);
                }
            }
            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                info!(%peer_id, ?cause, "connection closed");
                self.transport_peer_disconnected(peer_id);
                if matches!(self.role, SynapseRole::Worker { .. }) {
                    let was_active_master = self.authenticated_workers.remove(&peer_id)
                        || self.greeted_peers.remove(&peer_id);
                    self.dialed_peers.remove(&peer_id);
                    self.pending_coordinator_challenges.remove(&peer_id);
                    if self
                        .worker_assignment
                        .as_ref()
                        .is_some_and(|active| active.owner == peer_id)
                    {
                        self.release_worker_assignment();
                    }

                    if was_active_master {
                        info!(
                            %peer_id,
                            "master disconnected — worker staying online and waiting for reconnect"
                        );
                    }
                    self.redial_manual_master();
                }
                if matches!(self.role, SynapseRole::Master { .. }) {
                    self.authenticated_workers.remove(&peer_id);
                    self.worker_peers.retain(|peer| *peer != peer_id);
                    self.worker_registry.mark_disconnected(peer_id);
                    if self.generation_active && self.peer_required_for_active_generation(peer_id) {
                        warn!(%peer_id, "required worker disconnected — attempting self-healing");
                        self.initiate_healing(peer_id);
                    }
                }
            }
            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                warn!(?peer_id, %error, "outgoing connection failed");
            }
            SwarmEvent::IncomingConnectionError { error, .. } => {
                warn!(%error, "incoming connection failed");
            }
            SwarmEvent::Behaviour(SynapseBehaviourEvent::Identify(event)) => {
                self.handle_identify_event(event);
            }
            SwarmEvent::Behaviour(SynapseBehaviourEvent::Relay(event)) => {
                debug!(?event, "relay client event");
            }
            SwarmEvent::Behaviour(SynapseBehaviourEvent::RelayServer(event)) => {
                info!(?event, "relay server event");
            }
            SwarmEvent::Behaviour(SynapseBehaviourEvent::Dcutr(event)) => {
                debug!(?event, "dcutr hole-punch event");
            }
            SwarmEvent::Behaviour(SynapseBehaviourEvent::Autonat(event)) => {
                if let libp2p::autonat::Event::StatusChanged { new, .. } = &event {
                    if let libp2p::autonat::NatStatus::Public(addr) = new {
                        info!(%addr, "AutoNAT: node is publicly reachable, advertising address");
                        self.swarm.add_external_address(addr.clone());
                    } else {
                        debug!(?new, "AutoNAT status changed");
                    }
                }
            }
            SwarmEvent::Behaviour(SynapseBehaviourEvent::Kad(event)) => {
                self.handle_kad_event(event);
            }
            SwarmEvent::Behaviour(SynapseBehaviourEvent::Ping(event)) => {
                self.handle_ping_event(event);
            }
            SwarmEvent::Behaviour(SynapseBehaviourEvent::Mdns(event)) => {
                self.handle_mdns_event(event)?;
            }
            SwarmEvent::Behaviour(SynapseBehaviourEvent::RequestResponse(event)) => {
                self.handle_request_response_event(event);
            }
            other => {
                debug!(?other, "swarm event");
            }
        }

        Ok(())
    }

    fn handle_mdns_event(&mut self, event: mdns::Event) -> Result<(), NetworkError> {
        match event {
            mdns::Event::Discovered(peers) => {
                for (peer_id, address) in peers {
                    if peer_id == self.local_peer_id()
                        || !self
                            .net_config
                            .discovery
                            .allows(&peer_id, crate::netconfig::DiscoverySource::Lan)
                    {
                        continue;
                    }

                    info!(%peer_id, %address, "mDNS discovered peer");

                    if matches!(self.role, SynapseRole::Worker { .. })
                        && self.dialed_peers.insert(peer_id)
                    {
                        info!(%peer_id, %address, "worker dialing discovered peer");
                        self.swarm
                            .dial(address)
                            .map_err(|error| NetworkError::Dial(error.to_string()))?;
                    }
                }
            }
            mdns::Event::Expired(peers) => {
                for (peer_id, address) in peers {
                    info!(%peer_id, %address, "mDNS peer expired");
                    if matches!(self.role, SynapseRole::Worker { .. }) {
                        self.dialed_peers.remove(&peer_id);
                    }
                    if matches!(self.role, SynapseRole::Master { .. }) {
                        self.worker_registry.mark_expired(peer_id);
                    }
                }
            }
        }

        Ok(())
    }

    fn handle_request_response_event(
        &mut self,
        event: RequestResponseEvent<super::super::RequestMessage, super::super::ResponseMessage>,
    ) {
        match event {
            RequestResponseEvent::Message { peer, message, .. } => match message {
                RequestResponseMessage::Request {
                    request, channel, ..
                } => self.handle_inbound_request(peer, request, channel),
                RequestResponseMessage::Response {
                    request_id,
                    response,
                } => {
                    if self.accept_tracked_response(request_id, peer, &response) {
                        self.handle_inbound_response(peer, response);
                    }
                }
            },
            RequestResponseEvent::OutboundFailure {
                peer,
                request_id,
                error,
                ..
            } => {
                warn!(%peer, %request_id, %error, "outbound request failed");
                self.handle_tracked_failure(request_id, peer);
            }
            RequestResponseEvent::InboundFailure {
                peer,
                request_id,
                error,
                ..
            } => {
                warn!(%peer, %request_id, %error, "inbound request failed");
            }
            RequestResponseEvent::ResponseSent {
                peer, request_id, ..
            } => {
                debug!(%peer, %request_id, "response sent");
            }
        }
    }

    fn handle_identify_event(&mut self, event: identify::Event) {
        match event {
            identify::Event::Received { peer_id, info, .. } => {
                debug!(
                    %peer_id,
                    protocol_version = %info.protocol_version,
                    agent_version = %info.agent_version,
                    observed_addr = %info.observed_addr,
                    "identify: received peer info"
                );
                // Add the peer's listen addresses to the DHT for discovery.
                for addr in &info.listen_addrs {
                    self.swarm
                        .behaviour_mut()
                        .kad
                        .add_address(&peer_id, addr.clone());
                }

                // Cache authenticated workers for faster reconnection
                if matches!(self.role, SynapseRole::Master { .. })
                    && self.authenticated_workers.contains(&peer_id)
                {
                    self.peer_cache.add_provider(
                        self.model.id.clone(),
                        peer_id,
                        info.listen_addrs.clone(),
                    );
                    self.peer_cache.save();
                }
            }
            identify::Event::Sent { peer_id, .. } => {
                debug!(%peer_id, "identify: sent our info");
            }
            identify::Event::Pushed { peer_id, .. } => {
                debug!(%peer_id, "identify: pushed info");
            }
            identify::Event::Error { peer_id, error, .. } => {
                warn!(%peer_id, %error, "identify: error");
            }
        }
    }

    fn handle_kad_event(&mut self, event: kad::Event) {
        match event {
            kad::Event::OutboundQueryProgressed { result, .. } => match result {
                kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders {
                    key,
                    providers,
                })) => {
                    info!(key = ?String::from_utf8_lossy(key.as_ref()), peers = providers.len(), "DHT: found providers for model");

                    // Dial discovered peers that run the same model.
                    // Dial discovered peers that run the same model.
                    for peer in providers {
                        if peer != self.local_peer_id()
                            && !self.dialed_peers.contains(&peer)
                            && self
                                .net_config
                                .discovery
                                .allows(&peer, crate::netconfig::DiscoverySource::Dht)
                        {
                            info!(%peer, "DHT: dialing discovered peer");
                            if let Err(e) = self.swarm.dial(peer) {
                                warn!(%peer, %e, "DHT: failed to dial peer");
                            }
                            self.dialed_peers.insert(peer);
                        }
                    }
                }
                kad::QueryResult::GetProviders(Ok(
                    kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. },
                )) => {
                    debug!("DHT: provider search finished");
                }
                kad::QueryResult::GetProviders(Err(e)) => {
                    warn!(?e, "DHT: provider search failed");
                }
                kad::QueryResult::StartProviding(Ok(kad::AddProviderOk { key })) => {
                    info!(key = ?String::from_utf8_lossy(key.as_ref()), "DHT: now providing model");
                }
                kad::QueryResult::StartProviding(Err(e)) => {
                    warn!(?e, "DHT: failed to start providing");
                }
                other => {
                    debug!(?other, "DHT: query progress");
                }
            },
            kad::Event::RoutingUpdated { peer, .. } => {
                debug!(%peer, "DHT: routing table updated");
            }
            other => {
                debug!(?other, "DHT event");
            }
        }
    }

    fn handle_ping_event(&mut self, event: ping::Event) {
        match event.result {
            Ok(duration) => {
                debug!(peer = %event.peer, rtt_ms = duration.as_millis(), "ping: ok");
            }
            Err(error) => {
                debug!(peer = %event.peer, %error, "ping: failed");
            }
        }
    }

    fn handle_inbound_request(
        &mut self,
        peer: PeerId,
        request: super::super::RequestMessage,
        channel: request_response::ResponseChannel<super::super::ResponseMessage>,
    ) {
        if !self.charge_inbound_request(peer, &request) {
            self.reject(channel, "peer traffic quota exceeded");
            return;
        }
        if self.revocations.contains(&peer) {
            self.reject(channel, "peer revoked");
            return;
        }
        match request {
            super::super::RequestMessage::Challenge(_) => {
                self.reject(channel, "worker does not accept challenges");
            }
            super::super::RequestMessage::CapabilityPing(capability) => {
                if !matches!(self.role, SynapseRole::Master { .. })
                    || !self.authenticated_workers.contains(&peer)
                    || capability.compute_score == 0
                    || capability.available_ram_mb == 0
                {
                    self.reject(channel, "unauthorized or invalid capability update");
                    return;
                }
                self.worker_registry.update_capability(peer, capability);
                let newly_ready = !self.worker_peers.contains(&peer);
                if newly_ready {
                    self.worker_peers.push(peer);
                }
                self.send_response(
                    channel,
                    super::super::ResponseMessage::CapabilityAck { accepted: true },
                );
                if newly_ready {
                    self.send_health_ping(peer);
                    self.emit(ChatEvent::WorkerConnected);
                }
            }
            super::super::RequestMessage::PrepareShard(assignment) => {
                self.handle_prepare_shard(peer, assignment, channel)
            }
            super::super::RequestMessage::ReleaseShard(key) => {
                self.handle_release_shard(peer, key, channel)
            }
            super::super::RequestMessage::NodeHello {
                hello,
                pairing_code,
            } => self.handle_node_hello(peer, hello, pairing_code, channel),
            super::super::RequestMessage::ChallengeResponse(response) => {
                self.handle_challenge_response(peer, response, channel);
            }
            super::super::RequestMessage::WeightRequest(..) => {
                self.reject(channel, "weight transfer not implemented yet");
            }
            super::super::RequestMessage::ActivationRequest(request) => {
                self.handle_activation_request(peer, request, channel);
            }
            super::super::RequestMessage::ClearCache(request) => {
                self.handle_clear_cache_request(peer, request, channel);
            }
            super::super::RequestMessage::KvTruncate(request) => {
                self.handle_kv_truncate_request(peer, request, channel);
            }
            super::super::RequestMessage::HealthPing => {
                self.handle_health_ping(peer, channel);
            }
        }
    }

    fn handle_node_hello(
        &mut self,
        peer: PeerId,
        hello: NodeHello,
        _pairing_code: String,
        channel: request_response::ResponseChannel<super::super::ResponseMessage>,
    ) {
        let SynapseRole::Master { .. } = &self.role else {
            warn!(%peer, "worker received NodeHello request; rejecting");
            self.reject(channel, "NodeHello requests are only accepted by masters");
            return;
        };

        if let Err(error) = hello.validate() {
            warn!(%peer, %error, "NodeHello validation failed");
            self.reject(channel, "invalid NodeHello");
            return;
        }

        if hello.role != NodeRole::Worker {
            warn!(%peer, role = ?hello.role, "NodeHello did not come from worker");
            self.reject(channel, "expected worker role");
            return;
        }

        if hello.model != self.model {
            warn!(%peer, local_model = %self.model.id, remote_model = %hello.model.id, "model mismatch");
            self.reject(channel, "model mismatch");
            return;
        }

        // Rate-limit pairing attempts to prevent brute-force.
        let peer_key = peer.to_string();
        if let Err(remaining) = self.pairing_rate_limiter.check(&peer_key) {
            warn!(%peer, ?remaining, "pairing code attempt rate-limited");
            self.reject(
                channel,
                &format!(
                    "too many failed attempts; try again in {}s",
                    remaining.as_secs()
                ),
            );
            return;
        }

        // Phase 4: Instead of authenticating immediately, we issue a cryptographic challenge.
        // We still record the capability snapshot, but state remains Connected until ChallengeResponse.
        let capability_snapshot = WorkerCapabilitySnapshot::from_hello(&hello);
        self.worker_registry.upsert(
            peer,
            super::WorkerConnectionState::Connected,
            capability_snapshot,
        );

        let challenge = crate::Challenge::generate();
        self.worker_registry.set_challenge(peer, challenge.clone());

        info!(%peer, "NodeHello accepted, issuing challenge");
        self.send_response(
            channel,
            super::super::ResponseMessage::ChallengeIssued(challenge),
        );
    }

    fn handle_challenge_response(
        &mut self,
        peer: PeerId,
        response: crate::ChallengeResponse,
        channel: request_response::ResponseChannel<super::super::ResponseMessage>,
    ) {
        let SynapseRole::Master {
            expected_pairing_code,
            ..
        } = &self.role
        else {
            warn!(%peer, "worker received ChallengeResponse; rejecting");
            self.reject(channel, "only masters accept ChallengeResponse");
            return;
        };

        // Consume on every attempt, including invalid signatures and replay.
        let Some(pending_challenge) = self.worker_registry.take_challenge(&peer) else {
            self.reject(channel, "no pending authentication challenge");
            return;
        };
        let peer_key = peer.to_string();
        if crate::pairing::verify_challenge_response(
            &pending_challenge,
            &response,
            expected_pairing_code,
            self.local_peer_id(),
            peer,
            super::assignments::unix_now(),
        )
        .is_err()
        {
            self.pairing_rate_limiter.record_failure(&peer_key);
            self.reject(channel, "authentication failed");
            return;
        }
        let proof = match crate::pairing::coordinator_proof(
            &pending_challenge,
            expected_pairing_code,
            self.local_peer_id(),
            peer,
        ) {
            Ok(proof) => proof,
            Err(_) => {
                self.reject(channel, "coordinator proof failed");
                return;
            }
        };
        self.pairing_rate_limiter.record_success(&peer_key);

        self.authenticated_workers.insert(peer);
        self.worker_registry.mark_authenticated(peer, None);
        self.send_response(
            channel,
            super::super::ResponseMessage::PairingAccepted(proof),
        );
        // The worker confirms receipt using CapabilityPing. Only then is it a
        // route candidate, preventing prepare messages racing mutual auth.
    }

    fn redial_manual_master(&mut self) {
        const MAX_REDIAL_ATTEMPTS: u32 = 10;
        const REDIAL_BASE_DELAY: Duration = Duration::from_secs(1);
        const REDIAL_MAX_DELAY: Duration = Duration::from_secs(30);

        let SynapseRole::Worker {
            manual_master: Some(addr),
            ..
        } = &self.role
        else {
            return;
        };

        // Check if we're still in backoff period.
        if let Some(next_at) = self.next_redial_at
            && Instant::now() < next_at
        {
            debug!(
                address = %addr,
                remaining_ms = (next_at - Instant::now()).as_millis(),
                "redial suppressed by backoff"
            );
            return;
        }

        self.redial_attempts += 1;

        if self.redial_attempts > MAX_REDIAL_ATTEMPTS {
            warn!(
                address = %addr,
                attempts = self.redial_attempts,
                "max redial attempts reached, giving up"
            );
            return;
        }

        // Exponential backoff: 1s, 2s, 4s, 8s, ... capped at 30s.
        let delay = REDIAL_BASE_DELAY
            .saturating_mul(2u32.saturating_pow(self.redial_attempts.saturating_sub(1)))
            .min(REDIAL_MAX_DELAY);
        self.next_redial_at = Some(Instant::now() + delay);

        info!(
            address = %addr,
            attempt = self.redial_attempts,
            next_delay_secs = delay.as_secs(),
            "attempting redial to manual master address"
        );
        if let Err(error) = self.swarm.dial(addr.clone()) {
            warn!(%error, "manual master redial failed");
        }
    }

    /// Reset redial backoff state after a successful connection.
    pub(super) fn reset_redial_backoff(&mut self) {
        self.redial_attempts = 0;
        self.next_redial_at = None;
    }

    /// Notify connected workers and clean up state before shutdown.
    pub(super) fn shutdown_cleanup(&mut self) {
        self.release_generation_assignments();
        self.release_worker_assignment();
        self.pipeline = None;

        if self.generation_active {
            self.fail_generation("master shutting down");
        }

        while let Some(req) = self.active_batch.pop_front() {
            let _ = req.event_tx.try_send(ChatEvent::Token(
                "[shutdown] server is shutting down".to_owned(),
            ));
            let _ = req.event_tx.try_send(ChatEvent::GenerationDone {
                tokens_generated: 0,
                reason: StopReason::Error,
            });
        }
    }
}
