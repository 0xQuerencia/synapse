//! Runtime integration of transport correlation, quotas and acknowledged local
//! worker control. Only this module sends tracked requests to libp2p.
use super::{DEFAULT_REQUEST_TIMEOUT, NetworkError, SynapseNode, SynapseRole};
use crate::inference::assignment::LeaseKey;
use crate::network::stream::{ResponseKind, ResponseScope};
use crate::network::worker::{
    WorkerControlHandle, WorkerControlRequest, WorkerSnapshot, worker_control_channel,
};
use crate::network::{RequestMessage, ResponseMessage};
use libp2p::{PeerId, request_response::OutboundRequestId};
use std::time::Instant;
use tracing::warn;

fn base_scope(peer: PeerId, kind: ResponseKind, trace_id: String) -> ResponseScope {
    ResponseScope {
        peer,
        kind,
        session_id: "control".into(),
        route_epoch: 0,
        trace_id,
    }
}
fn lease_scope(peer: PeerId, kind: ResponseKind, key: &LeaseKey) -> ResponseScope {
    ResponseScope {
        peer,
        kind,
        session_id: key.session_id.clone(),
        route_epoch: key.route_epoch,
        trace_id: key.trace_prefix(),
    }
}
fn trace_scope(peer: PeerId, kind: ResponseKind, trace: &str) -> ResponseScope {
    let mut scope = base_scope(peer, kind, trace.to_owned());
    // LeaseKey::trace_prefix: s<16hex>e<epoch>-. Non-generation control traces
    // remain in a separate scope and can never fail an active generation.
    if trace.is_ascii()
        && trace.len() > 19
        && trace.starts_with('s')
        && trace.as_bytes().get(17) == Some(&b'e')
        && trace[1..17].bytes().all(|byte| byte.is_ascii_hexdigit())
        && let Some((epoch, _)) = trace[18..].split_once('-')
        && let Ok(epoch) = epoch.parse::<u64>()
        && epoch > 0
    {
        scope.session_id = trace[1..17].to_owned();
        scope.route_epoch = epoch;
    }
    scope
}
fn artifact_trace(model: &str, filename: &str, offset: u64) -> String {
    let key = format!(
        "{}:{model}{}:{filename}:{offset}",
        model.len(),
        filename.len()
    );
    format!(
        "artifact-{}",
        &crate::hash::sha256_hex(key.as_bytes())[..32]
    )
}
fn expected_scope(peer: PeerId, request: &RequestMessage) -> Result<ResponseScope, NetworkError> {
    let scope = match request {
        RequestMessage::PrepareShard(value) => {
            lease_scope(peer, ResponseKind::ShardReady, &value.key)
        }
        RequestMessage::ReleaseShard(value) => {
            lease_scope(peer, ResponseKind::ShardReleased, value)
        }
        RequestMessage::NodeHello { .. } => base_scope(peer, ResponseKind::Hello, "hello".into()),
        RequestMessage::ChallengeResponse(value) => {
            base_scope(peer, ResponseKind::Pairing, value.nonce.clone())
        }
        RequestMessage::CapabilityPing(_) => {
            base_scope(peer, ResponseKind::Capability, "capability".into())
        }
        RequestMessage::HealthPing => base_scope(peer, ResponseKind::Health, "health".into()),
        RequestMessage::WeightRequest(value) => base_scope(
            peer,
            ResponseKind::Artifact,
            artifact_trace(&value.model_id, &value.filename, value.offset),
        ),
        RequestMessage::ActivationRequest(value) => {
            trace_scope(peer, ResponseKind::Activation, &value.trace_id)
        }
        RequestMessage::ClearCache(value) => {
            trace_scope(peer, ResponseKind::CacheClear, &value.trace_id)
        }
        RequestMessage::KvTruncate(value) => {
            trace_scope(peer, ResponseKind::Truncate, &value.trace_id)
        }
        RequestMessage::Challenge(_) => {
            return Err(NetworkError::Transport(
                "standalone challenge requests are unsupported".into(),
            ));
        }
    };
    if !scope.validate() {
        return Err(NetworkError::Transport("invalid request scope".into()));
    }
    Ok(scope)
}
fn received_scope(
    peer: PeerId,
    response: &ResponseMessage,
    expected: &ResponseScope,
) -> Option<ResponseScope> {
    Some(match response {
        ResponseMessage::ShardReady(value) => {
            lease_scope(peer, ResponseKind::ShardReady, &value.assignment.key)
        }
        ResponseMessage::ShardReleased(value) => {
            lease_scope(peer, ResponseKind::ShardReleased, value)
        }
        ResponseMessage::PairingAccepted(value) => {
            base_scope(peer, ResponseKind::Pairing, value.nonce.clone())
        }
        ResponseMessage::ChallengeIssued(_) => {
            base_scope(peer, ResponseKind::Hello, "hello".into())
        }
        ResponseMessage::CapabilityAck { .. } => {
            base_scope(peer, ResponseKind::Capability, "capability".into())
        }
        ResponseMessage::HealthPong(_) => base_scope(peer, ResponseKind::Health, "health".into()),
        ResponseMessage::WeightTransfer(value) => base_scope(
            peer,
            ResponseKind::Artifact,
            artifact_trace(&value.model_id, &value.filename, value.offset),
        ),
        ResponseMessage::ActivationResponse(value) => {
            trace_scope(peer, ResponseKind::Activation, &value.trace_id)
        }
        ResponseMessage::ClearCacheAck(value) => {
            trace_scope(peer, ResponseKind::CacheClear, &value.trace_id)
        }
        ResponseMessage::KvTruncateAck(value) => {
            trace_scope(peer, ResponseKind::Truncate, &value.trace_id)
        }
        ResponseMessage::Rejected { .. } => ResponseScope {
            peer,
            ..expected.clone()
        },
        ResponseMessage::Authenticated => return None,
    })
}

impl SynapseNode {
    pub(super) fn send_tracked_request(
        &mut self,
        peer: PeerId,
        request: RequestMessage,
    ) -> Result<OutboundRequestId, NetworkError> {
        request
            .validate_wire()
            .map_err(|e| NetworkError::Transport(e.to_string()))?;
        if self.pending_requests.is_full() {
            return Err(NetworkError::Transport(
                "outbound request capacity exhausted".into(),
            ));
        }
        let scope = expected_scope(peer, &request)?;
        let now = Instant::now();
        let id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer, request);
        // No await or competing mutation between the capacity check and
        // registration. libp2p request IDs are unique within this behaviour.
        self.pending_requests
            .register(id, scope, now + DEFAULT_REQUEST_TIMEOUT, now)
            .map_err(|e| NetworkError::Transport(e.to_string()))?;
        Ok(id)
    }

    fn scope_is_active_generation(&self, scope: &ResponseScope) -> bool {
        !matches!(scope.kind, ResponseKind::ShardReleased)
            && self.generation_active
            && self.generation_lease.as_ref().is_some_and(|key| {
                key.session_id == scope.session_id && key.route_epoch == scope.route_epoch
            })
    }

    /// Caller must pass request_id from libp2p's Response event. Returns false
    /// for stale, unsolicited, cross-peer or unrelated control errors.
    pub(super) fn accept_tracked_response(
        &mut self,
        id: OutboundRequestId,
        peer: PeerId,
        response: &ResponseMessage,
    ) -> bool {
        let Some(expected) = self.pending_requests.scope(id).cloned() else {
            return false;
        };
        let Some(actual) = received_scope(peer, response, &expected) else {
            return false;
        };
        if response.validate_wire().is_err()
            || self
                .pending_requests
                .complete(id, &actual, Instant::now())
                .is_err()
        {
            warn!(%peer, "rejected unmatched network response");
            return false;
        }
        if matches!(response, ResponseMessage::Rejected { .. })
            && !self.scope_is_active_generation(&expected)
        {
            if matches!(expected.kind, ResponseKind::Hello | ResponseKind::Pairing) {
                self.pending_coordinator_challenges.remove(&peer);
                self.greeted_peers.remove(&peer);
            }
            warn!(%peer, "peer rejected a control request");
            return false;
        }
        true
    }

    pub(super) fn handle_tracked_failure(&mut self, id: OutboundRequestId, peer: PeerId) {
        if let Some(scope) = self.pending_requests.remove(id)
            && scope.peer == peer
            && self.scope_is_active_generation(&scope)
        {
            self.fail_generation("assigned worker request failed");
        }
    }

    pub(super) fn charge_inbound_request(
        &mut self,
        peer: PeerId,
        request: &RequestMessage,
    ) -> bool {
        bincode::serialized_size(request).ok().is_some_and(|bytes| {
            self.traffic_quota
                .check_and_charge(peer, bytes, Instant::now())
                .is_ok()
        })
    }

    pub(super) fn transport_tick(&mut self) {
        let now = Instant::now();
        let expired = self.pending_requests.expire(now);
        if expired
            .iter()
            .any(|scope| self.scope_is_active_generation(scope))
        {
            self.fail_generation("assigned worker response deadline expired");
        }
        let active_jobs = usize::from(self.worker_assignment.is_some());
        if let Some(effect) = self.worker_controller.tick(active_jobs, now) {
            if effect.release_active {
                self.release_worker_assignment();
            }
            self.publish_status();
        }
    }

    pub(super) fn transport_peer_disconnected(&mut self, peer: PeerId) {
        self.pending_requests.disconnect(peer);
        self.pending_coordinator_challenges.remove(&peer);
        if self
            .worker_assignment
            .as_ref()
            .is_some_and(|active| active.owner == peer)
        {
            self.release_worker_assignment();
        }
    }

    /// Obtain a local acknowledged control channel before spawning run().
    pub fn worker_control(&mut self) -> Result<WorkerControlHandle, NetworkError> {
        if !matches!(self.role, SynapseRole::Worker { .. }) {
            return Err(NetworkError::Transport(
                "worker controls require a worker node".into(),
            ));
        }
        if let Some(handle) = &self.worker_control_handle {
            return Ok(handle.clone());
        }
        let (handle, receiver) = worker_control_channel();
        self.worker_control_handle = Some(handle.clone());
        self.worker_control_rx = Some(receiver);
        Ok(handle)
    }

    pub fn worker_snapshot(&self) -> WorkerSnapshot {
        self.worker_controller
            .snapshot(usize::from(self.worker_assignment.is_some()))
    }

    /// Returns true after applying Stop so the event loop exits immediately.
    pub(super) fn apply_worker_control(&mut self, request: WorkerControlRequest) -> bool {
        let active_jobs = usize::from(self.worker_assignment.is_some());
        match self
            .worker_controller
            .apply(request.action, active_jobs, Instant::now())
        {
            Ok(effect) => {
                if effect.release_active {
                    self.release_worker_assignment();
                }
                if effect.exit {
                    self.shutdown_cleanup();
                }
                self.publish_status();
                // Derive the acknowledgement from state after actual cleanup.
                request.respond(Ok(self.worker_snapshot()));
                effect.exit
            }
            Err(error) => {
                request.respond(Err(error));
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn activation_scope_extracts_exact_session_epoch_and_keeps_token_trace() {
        let peer = PeerId::random();
        let scope = trace_scope(
            peer,
            ResponseKind::Activation,
            "s0123456789abcdefe12-token-7",
        );
        assert_eq!(scope.session_id, "0123456789abcdef");
        assert_eq!(scope.route_epoch, 12);
        assert_eq!(scope.trace_id, "s0123456789abcdefe12-token-7");
        assert_eq!(
            trace_scope(peer, ResponseKind::Health, "health").route_epoch,
            0
        );
    }
    #[test]
    fn negative_response_keeps_request_scope_but_binds_actual_peer() {
        let expected = base_scope(PeerId::random(), ResponseKind::Health, "health".into());
        let attacker = PeerId::random();
        let actual = received_scope(attacker, &ResponseMessage::rejected("no"), &expected).unwrap();
        assert_ne!(actual, expected);
        assert_eq!(actual.peer, attacker);
    }
    #[test]
    fn wrong_response_kind_is_not_correlated_to_activation() {
        let peer = PeerId::random();
        let expected = trace_scope(
            peer,
            ResponseKind::Activation,
            "s0123456789abcdefe1-token-0",
        );
        let response = ResponseMessage::ClearCacheAck(crate::ClearCache {
            trace_id: expected.trace_id.clone(),
        });
        assert_ne!(
            received_scope(peer, &response, &expected).unwrap().kind,
            expected.kind
        );
    }
}
