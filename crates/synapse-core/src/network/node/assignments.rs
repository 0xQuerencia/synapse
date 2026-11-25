//! Real shard prepare/ready/release orchestration for a private session.
use anyhow::{Result, ensure};
use libp2p::{PeerId, request_response};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tracing::{info, warn};

use super::{GenerationRoutePlan, NetworkError, SynapseNode, SynapseRole};
use crate::inference::assignment::{LeaseKey, ReadinessBarrier, ShardAssignment, ShardReady};
use crate::network::{RequestMessage, ResponseMessage};

pub(super) struct WorkerAssignment {
    pub owner: PeerId,
    pub assignment: ShardAssignment,
    pub expires_at: Instant,
    pub cursor: crate::inference::admission::CacheCursor,
}

pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

impl SynapseNode {
    pub(super) fn begin_route_preparation(&mut self, route: &GenerationRoutePlan) -> Result<()> {
        self.route_epoch = crate::network::session::EpochCounter::advance(
            &self.identity.key_path().with_extension("epochs.json"),
        )?;
        let key = LeaseKey {
            session_id: format!("{:016x}", rand::random::<u64>()),
            route_epoch: self.route_epoch,
            model_digest: self
                .model_files
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("model metadata unavailable"))?
                .manifest
                .digest()?,
        };
        let admission = self
            .admission
            .clone()
            .ok_or_else(|| anyhow::anyhow!("local memory capacity is unknown"))?;
        admission.reserve(
            key.clone(),
            &self.local_peer_id().to_string(),
            self.memory_limit_bytes,
            Duration::from_secs(300),
            Instant::now(),
        )?;
        self.generation_lease = Some(key.clone());
        self.load_assigned_range(route.master_layer_range.clone())?;
        admission.mark_ready(&key, &self.local_peer_id().to_string(), Instant::now())?;
        let context_tokens = self
            .token_ids
            .len()
            .checked_add(self.current_max_tokens())
            .ok_or_else(|| anyhow::anyhow!("context overflow"))?;
        let mut expected = BTreeMap::new();
        for hop in &route.worker_hops {
            let capability = self
                .worker_registry
                .capability_for(&hop.peer_id)
                .and_then(|capability| capability.available_ram_mb)
                .unwrap_or(0);
            let assignment = ShardAssignment {
                protocol_version: 1,
                key: key.clone(),
                layer_start: hop.layer_range.start.try_into()?,
                layer_end: hop.layer_range.end.try_into()?,
                total_layers: self.total_layers.try_into()?,
                context_tokens: context_tokens.try_into()?,
                memory_limit_bytes: capability
                    .checked_mul(1024 * 1024)
                    .ok_or_else(|| anyhow::anyhow!("memory cap overflow"))?,
                ttl_ms: 300_000,
            };
            assignment.validate()?;
            expected.insert(hop.peer_id.to_string(), assignment);
        }
        let barrier =
            ReadinessBarrier::new(expected.clone(), Instant::now(), Duration::from_secs(30))?;
        self.shard_readiness = Some(barrier);
        for hop in &route.worker_hops {
            let assignment = expected
                .get(&hop.peer_id.to_string())
                .expect("constructed assignment")
                .clone();
            self.send_tracked_request(hop.peer_id, RequestMessage::PrepareShard(assignment))?;
        }
        if route.worker_hops.is_empty() {
            self.shard_readiness = None;
            self.send_next_activation();
        }
        Ok(())
    }

    pub(super) fn handle_prepare_shard(
        &mut self,
        peer: PeerId,
        assignment: ShardAssignment,
        channel: request_response::ResponseChannel<ResponseMessage>,
    ) {
        let result = self.prepare_worker_shard(peer, assignment.clone());
        match result {
            Ok(ready) => self.send_response(channel, ResponseMessage::ShardReady(ready)),
            Err(error) => {
                warn!(%peer,%error,"shard preparation rejected");
                self.reject(channel, &error.to_string());
            }
        }
    }

    fn prepare_worker_shard(
        &mut self,
        peer: PeerId,
        assignment: ShardAssignment,
    ) -> Result<ShardReady> {
        ensure!(
            matches!(self.role, SynapseRole::Worker { .. }),
            "only workers accept shard assignments"
        );
        ensure!(
            self.authenticated_workers.contains(&peer) && !self.revocations.contains(&peer),
            "unauthorized assignment"
        );
        ensure!(
            self.worker_controller.allows_assignment(),
            "worker is paused, draining or stopped"
        );
        assignment.validate()?;
        ensure!(
            assignment.total_layers as usize == self.total_layers,
            "model depth mismatch"
        );
        let files = self
            .model_files
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model metadata not loaded"))?;
        ensure!(
            assignment.key.model_digest == files.manifest.digest()?,
            "model manifest mismatch"
        );
        ensure!(
            assignment.memory_limit_bytes <= self.memory_limit_bytes,
            "assignment exceeds worker memory policy"
        );
        ensure!(
            self.model_config
                .as_ref()
                .is_some_and(|config| assignment.context_tokens as usize <= config.context_limit()),
            "context exceeds model limit"
        );
        let now = Instant::now();
        if let Some(active) = &self.worker_assignment {
            if active.expires_at > now {
                ensure!(
                    active.owner == peer,
                    "worker is leased to another coordinator"
                );
                if active.assignment == assignment {
                    return Ok(ShardReady {
                        assignment,
                        resident_weight_bytes: self
                            .pipeline
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("ready shard was unloaded"))?
                            .resident_weight_bytes(),
                    });
                }
                ensure!(
                    active.assignment.key.session_id == assignment.key.session_id
                        && assignment.key.route_epoch > active.assignment.key.route_epoch,
                    "worker already has an active session"
                );
            }
        }
        self.release_worker_assignment();
        let admission = self
            .admission
            .clone()
            .ok_or_else(|| anyhow::anyhow!("worker capacity unavailable"))?;
        admission.reserve(
            assignment.key.clone(),
            &peer.to_string(),
            assignment.memory_limit_bytes,
            Duration::from_millis(assignment.ttl_ms as u64),
            now,
        )?;
        if let Err(error) = self.load_assigned_range(assignment.layer_range()) {
            admission.release(&assignment.key, &peer.to_string())?;
            self.pipeline = None;
            return Err(error);
        }
        let bytes = self
            .pipeline
            .as_ref()
            .expect("loaded above")
            .resident_weight_bytes();
        if bytes > assignment.memory_limit_bytes
            || admission
                .mark_ready(&assignment.key, &peer.to_string(), Instant::now())
                .is_err()
        {
            admission.release(&assignment.key, &peer.to_string())?;
            self.pipeline = None;
            anyhow::bail!("loaded weights exceed reservation or preparation lease expired");
        }
        self.worker_assignment = Some(WorkerAssignment {
            owner: peer,
            expires_at: now + Duration::from_millis(assignment.ttl_ms as u64),
            assignment: assignment.clone(),
            cursor: crate::inference::admission::CacheCursor::default(),
        });
        info!(%peer,session=%assignment.key.session_id,epoch=assignment.key.route_epoch,
            layer_start=assignment.layer_start,layer_end=assignment.layer_end,resident_weight_bytes=bytes,"shard ready");
        Ok(ShardReady {
            assignment,
            resident_weight_bytes: bytes,
        })
    }

    pub(super) fn handle_shard_ready(&mut self, peer: PeerId, ready: ShardReady) {
        if !self.authenticated_workers.contains(&peer) || !self.generation_active {
            return;
        }
        let Some(barrier) = &mut self.shard_readiness else {
            return;
        };
        match barrier.acknowledge(&peer.to_string(), &ready, Instant::now()) {
            Ok(true) => {
                self.shard_readiness = None;
                let peers: Vec<_> = self
                    .generation_route_plan
                    .as_ref()
                    .map(|route| route.worker_hops.iter().map(|hop| hop.peer_id).collect())
                    .unwrap_or_default();
                self.send_clear_cache_requests(&peers);
            }
            Ok(false) => {}
            Err(error) => self.fail_generation(format!("shard readiness failed: {error}")),
        }
    }

    pub(super) fn handle_release_shard(
        &mut self,
        peer: PeerId,
        key: LeaseKey,
        channel: request_response::ResponseChannel<ResponseMessage>,
    ) {
        if key.validate().is_err()
            || !self.authenticated_workers.contains(&peer)
            || self.revocations.contains(&peer)
        {
            self.reject(channel, "unauthorized shard release");
            return;
        }
        if self
            .worker_assignment
            .as_ref()
            .is_some_and(|active| active.owner == peer && active.assignment.key == key)
        {
            self.release_worker_assignment();
        }
        self.send_response(channel, ResponseMessage::ShardReleased(key));
    }

    pub(super) fn authorize_worker_trace(&self, peer: PeerId, trace_id: &str) -> bool {
        self.authenticated_workers.contains(&peer)
            && !self.revocations.contains(&peer)
            && self.worker_assignment.as_ref().is_some_and(|active| {
                active.owner == peer
                    && active.expires_at > Instant::now()
                    && active.assignment.key.owns_trace(trace_id)
                    && self.admission.as_ref().is_some_and(|pool| {
                        pool.is_ready(&active.assignment.key, &peer.to_string(), Instant::now())
                    })
            })
    }

    pub(super) fn release_generation_assignments(&mut self) {
        if let Some(key) = self.generation_lease.take() {
            if let Some(admission) = &self.admission {
                let _ = admission.release(&key, &self.local_peer_id().to_string());
            }
            self.pending_requests.cancel_session(&key.session_id);
            let peers: Vec<_> = self
                .generation_route_plan
                .as_ref()
                .map(|route| route.worker_hops.iter().map(|hop| hop.peer_id).collect())
                .unwrap_or_default();
            for peer in peers {
                if let Err(error) =
                    self.send_tracked_request(peer, RequestMessage::ReleaseShard(key.clone()))
                {
                    warn!(%peer, %error, "release request failed; remote lease expires independently");
                }
            }
        }
        self.shard_readiness = None;
        if matches!(self.role, SynapseRole::Master { .. }) {
            self.pipeline = None;
        }
    }

    pub(super) fn expire_assignments(&mut self) {
        let now = Instant::now();
        let expired = self
            .admission
            .as_ref()
            .and_then(|pool| pool.expire(now).ok())
            .unwrap_or_default();
        if self
            .generation_lease
            .as_ref()
            .is_some_and(|key| expired.contains(key))
        {
            self.fail_generation("session capacity lease expired");
        }
        if self
            .worker_assignment
            .as_ref()
            .is_some_and(|active| active.expires_at <= now)
        {
            self.release_worker_assignment();
            info!("expired worker lease released");
        }
        if self
            .shard_readiness
            .as_ref()
            .is_some_and(|barrier| barrier.is_expired(now))
        {
            self.fail_generation("timed out preparing complete shard route");
        }
        if self.generation_active && self.event_tx.as_ref().is_some_and(|tx| tx.is_closed()) {
            self.fail_generation("requester disconnected");
        }
    }

    /// Persist revocation before disconnecting or allowing new work.
    pub fn revoke_peer(&mut self, peer: PeerId) -> Result<(), NetworkError> {
        self.revocations
            .revoke(
                peer,
                &self.identity.key_path().with_extension("revocations.json"),
            )
            .map_err(|error| NetworkError::Transport(error.to_string()))?;
        self.authenticated_workers.remove(&peer);
        self.pending_coordinator_challenges.remove(&peer);
        if self.peer_required_for_active_generation(peer) {
            self.fail_generation("assigned peer revoked");
        }
        if self
            .worker_assignment
            .as_ref()
            .is_some_and(|active| active.owner == peer)
        {
            self.release_worker_assignment();
        }
        let _ = self.swarm.disconnect_peer_id(peer);
        Ok(())
    }

    pub(super) fn release_worker_assignment(&mut self) {
        if let Some(active) = self.worker_assignment.take()
            && let Some(admission) = &self.admission
        {
            let _ = admission.release(&active.assignment.key, &active.owner.to_string());
        }
        if matches!(self.role, SynapseRole::Worker { .. }) {
            self.pipeline = None;
        }
    }
}
