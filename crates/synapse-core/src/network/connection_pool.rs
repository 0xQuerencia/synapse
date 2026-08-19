//! Phase 6: Connection Pooling
//!
//! Pre-warmed connection pool to top workers for zero-latency request dispatch.
//! Eliminates connection establishment overhead (20-40ms per request).

use libp2p::PeerId;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

/// Number of top workers to maintain warm connections to.
pub const WARM_POOL_SIZE: usize = 5;

/// Health check interval for connection monitoring.
pub const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(30);

/// Max time without activity before considering connection stale.
const CONNECTION_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// Connection state for a pooled peer.
#[derive(Debug, Clone)]
pub struct ConnectionState {
    pub peer_id: PeerId,
    /// Last time this connection was used for a request.
    pub last_used: Instant,
    /// Last successful health check (ping response).
    pub last_health_check: Instant,
    /// Whether the peer is considered healthy.
    pub is_healthy: bool,
    /// Number of consecutive failed health checks.
    pub failed_checks: u32,
    /// Total requests sent through this connection.
    pub request_count: u64,
}

impl ConnectionState {
    fn new(peer_id: PeerId) -> Self {
        let now = Instant::now();
        Self {
            peer_id,
            last_used: now,
            last_health_check: now,
            is_healthy: true,
            failed_checks: 0,
            request_count: 0,
        }
    }

    /// Mark connection as used (resets idle timer).
    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.request_count += 1;
    }

    /// Update health check status.
    pub fn update_health(&mut self, success: bool) {
        self.last_health_check = Instant::now();
        if success {
            self.is_healthy = true;
            self.failed_checks = 0;
        } else {
            self.failed_checks += 1;
            if self.failed_checks >= 3 {
                self.is_healthy = false;
            }
        }
    }

    /// Check if connection is idle and should be removed.
    pub fn is_idle(&self) -> bool {
        self.last_used.elapsed() > CONNECTION_IDLE_TIMEOUT
    }
}

/// Connection pool for top worker peers.
///
/// Maintains persistent connections to the fastest N workers to eliminate
/// connection establishment overhead on request dispatch.
#[derive(Debug)]
pub struct ConnectionPool {
    /// Map of peer_id to connection state.
    warm_connections: HashMap<PeerId, ConnectionState>,
    /// Maximum number of connections to maintain.
    max_pool_size: usize,
    /// When the last health check sweep was performed.
    last_health_check: Instant,
}

impl ConnectionPool {
    /// Create a new connection pool with default size.
    pub fn new() -> Self {
        Self::with_capacity(WARM_POOL_SIZE)
    }

    /// Create a connection pool with custom capacity.
    pub fn with_capacity(max_pool_size: usize) -> Self {
        Self {
            warm_connections: HashMap::with_capacity(max_pool_size),
            max_pool_size,
            last_health_check: Instant::now(),
        }
    }

    /// Pre-warm connections to top N peers.
    ///
    /// This should be called when the worker ranking changes (after heartbeats)
    /// to ensure connections to the fastest workers are ready.
    pub fn prewarm(&mut self, peers: &[PeerId]) {
        let target_peers: Vec<PeerId> = peers.iter().take(self.max_pool_size).copied().collect();

        // Add new peers to pool
        for peer_id in &target_peers {
            if !self.warm_connections.contains_key(peer_id) {
                info!(%peer_id, "prewarming connection to top worker");
                self.warm_connections
                    .insert(*peer_id, ConnectionState::new(*peer_id));
            }
        }

        // Remove peers no longer in top N
        let to_remove: Vec<PeerId> = self
            .warm_connections
            .keys()
            .filter(|peer_id| !target_peers.contains(peer_id))
            .copied()
            .collect();

        for peer_id in to_remove {
            debug!(%peer_id, "removing peer from warm pool (no longer in top workers)");
            self.warm_connections.remove(&peer_id);
        }
    }

    /// Check if a peer is in the warm connection pool.
    pub fn has_warm_connection(&self, peer_id: &PeerId) -> bool {
        self.warm_connections
            .get(peer_id)
            .map(|state| state.is_healthy)
            .unwrap_or(false)
    }

    /// Mark a pooled connection as used.
    pub fn mark_connection_used(&mut self, peer_id: &PeerId) {
        if let Some(state) = self.warm_connections.get_mut(peer_id) {
            state.mark_used();
            debug!(%peer_id, count = state.request_count, "marked pooled connection as used");
        }
    }

    /// Update health status for a peer (e.g., after ping response).
    pub fn update_peer_health(&mut self, peer_id: &PeerId, success: bool) {
        if let Some(state) = self.warm_connections.get_mut(peer_id) {
            state.update_health(success);
            if !success {
                warn!(
                        %peer_id,
                failed_checks = state.failed_checks,
                             "health check failed for pooled connection"
                        );
            }
        }
    }

    /// Get all peers in the warm pool.
    pub fn pooled_peers(&self) -> Vec<PeerId> {
        self.warm_connections.keys().copied().collect()
    }

    /// Get healthy peers in the warm pool.
    pub fn healthy_peers(&self) -> Vec<PeerId> {
        self.warm_connections
            .values()
            .filter(|state| state.is_healthy)
            .map(|state| state.peer_id)
            .collect()
    }

    /// Run periodic health check maintenance.
    ///
    /// Returns true if health checks should be triggered (30s interval).
    pub fn should_run_health_check(&mut self) -> bool {
        if self.last_health_check.elapsed() >= HEALTH_CHECK_INTERVAL {
            self.last_health_check = Instant::now();
            true
        } else {
            false
        }
    }

    /// Remove idle and unhealthy connections.
    pub fn prune_stale_connections(&mut self) {
        let to_remove: Vec<PeerId> = self
            .warm_connections
            .iter()
            .filter(|(_, state)| !state.is_healthy || state.is_idle())
            .map(|(peer_id, _)| *peer_id)
            .collect();

        for peer_id in to_remove {
            let state = self.warm_connections.remove(&peer_id).unwrap();
            if !state.is_healthy {
                info!(%peer_id, "removed unhealthy connection from pool");
            } else if state.is_idle() {
                debug!(%peer_id, "removed idle connection from pool");
            }
        }
    }

    /// Get statistics about the connection pool.
    pub fn stats(&self) -> PoolStats {
        let healthy = self.healthy_peers().len();
        let total = self.warm_connections.len();
        let total_requests: u64 = self
            .warm_connections
            .values()
            .map(|state| state.request_count)
            .sum();

        PoolStats {
            total_connections: total,
            healthy_connections: healthy,
            max_pool_size: self.max_pool_size,
            total_requests,
        }
    }
}

impl Default for ConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection pool statistics.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub total_connections: usize,
    pub healthy_connections: usize,
    pub max_pool_size: usize,
    pub total_requests: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
