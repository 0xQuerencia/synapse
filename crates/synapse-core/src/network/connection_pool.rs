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
