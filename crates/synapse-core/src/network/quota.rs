//! Per-peer and global traffic accounting before application dispatch.
//! Identity rotation cannot reset the global window; rejected traffic does not
//! evict another peer's accounting entry.
use libp2p::PeerId;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug)]
pub struct TrafficLimits {
    pub bytes_per_peer: u64,
    pub bytes_total: u64,
    pub messages_per_peer: u32,
    pub messages_total: u32,
    pub max_peers: usize,
    pub window: Duration,
}
impl Default for TrafficLimits {
    fn default() -> Self {
        Self {
            bytes_per_peer: 256 * 1024 * 1024,
            bytes_total: 512 * 1024 * 1024,
            messages_per_peer: 4096,
            messages_total: 16384,
            max_peers: 128,
            window: Duration::from_secs(60),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum QuotaError {
    #[error("invalid traffic limits")]
    InvalidLimits,
    #[error("traffic quota exhausted")]
    Exhausted,
}
#[derive(Clone, Copy, Debug, Default)]
struct Usage {
    bytes: u64,
    messages: u32,
}
#[derive(Debug)]
pub struct TrafficQuota {
    limits: TrafficLimits,
    window_started: Instant,
    peers: HashMap<PeerId, Usage>,
    total: Usage,
}
impl TrafficQuota {
    pub fn new(limits: TrafficLimits, now: Instant) -> Result<Self, QuotaError> {
        if limits.bytes_per_peer == 0
            || limits.bytes_total < limits.bytes_per_peer
            || limits.messages_per_peer == 0
            || limits.messages_total < limits.messages_per_peer
            || limits.max_peers == 0
            || limits.max_peers > 4096
            || limits.window.is_zero()
        {
            return Err(QuotaError::InvalidLimits);
        }
        Ok(Self {
            limits,
            window_started: now,
            peers: HashMap::new(),
            total: Usage::default(),
        })
    }
    pub fn check_and_charge(
        &mut self,
        peer: PeerId,
        bytes: u64,
        now: Instant,
    ) -> Result<(), QuotaError> {
        if now
            .checked_duration_since(self.window_started)
            .is_some_and(|elapsed| elapsed >= self.limits.window)
        {
            self.window_started = now;
            self.peers.clear();
            self.total = Usage::default();
        }
        if !self.peers.contains_key(&peer) && self.peers.len() >= self.limits.max_peers {
            return Err(QuotaError::Exhausted);
        }
        let used = self.peers.get(&peer).copied().unwrap_or_default();
        let peer_bytes = used.bytes.checked_add(bytes).ok_or(QuotaError::Exhausted)?;
        let total_bytes = self
            .total
            .bytes
            .checked_add(bytes)
            .ok_or(QuotaError::Exhausted)?;
        let peer_messages = used.messages.checked_add(1).ok_or(QuotaError::Exhausted)?;
        let total_messages = self
            .total
            .messages
            .checked_add(1)
            .ok_or(QuotaError::Exhausted)?;
        if peer_bytes > self.limits.bytes_per_peer
            || total_bytes > self.limits.bytes_total
            || peer_messages > self.limits.messages_per_peer
            || total_messages > self.limits.messages_total
        {
            return Err(QuotaError::Exhausted);
        }
        self.peers.insert(
            peer,
            Usage {
                bytes: peer_bytes,
                messages: peer_messages,
            },
        );
        self.total = Usage {
            bytes: total_bytes,
            messages: total_messages,
        };
        Ok(())
    }
    pub fn retry_after(&self, now: Instant) -> Duration {
        self.limits
            .window
            .saturating_sub(now.saturating_duration_since(self.window_started))
    }
}
impl Default for TrafficQuota {
    fn default() -> Self {
        Self::new(TrafficLimits::default(), Instant::now()).expect("valid static traffic limits")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn limits() -> TrafficLimits {
        TrafficLimits {
            bytes_per_peer: 16,
            bytes_total: 24,
            messages_per_peer: 2,
            messages_total: 3,
            max_peers: 2,
            window: Duration::from_secs(10),
        }
    }
    #[test]
    fn per_peer_and_global_limits_cannot_be_bypassed_by_identity_rotation() {
        let now = Instant::now();
        let mut quota = TrafficQuota::new(limits(), now).unwrap();
        let first = PeerId::random();
        quota.check_and_charge(first, 16, now).unwrap();
        assert_eq!(
            quota.check_and_charge(first, 1, now),
            Err(QuotaError::Exhausted)
        );
        quota.check_and_charge(PeerId::random(), 8, now).unwrap();
        assert_eq!(
            quota.check_and_charge(PeerId::random(), 1, now),
            Err(QuotaError::Exhausted)
        );
        quota
            .check_and_charge(first, 16, now + Duration::from_secs(10))
            .unwrap();
    }
    #[test]
    fn oversized_and_overflow_requests_do_not_consume_budget() {
        let now = Instant::now();
        let mut quota = TrafficQuota::new(limits(), now).unwrap();
        let peer = PeerId::random();
        assert!(quota.check_and_charge(peer, u64::MAX, now).is_err());
        quota.check_and_charge(peer, 16, now).unwrap();
        assert!(quota.check_and_charge(peer, u64::MAX, now).is_err());
        assert_eq!(quota.total.bytes, 16);
        assert_eq!(quota.total.messages, 1);
    }
    #[test]
    fn tiny_message_flood_hits_message_limit_and_backwards_time_does_not_reset() {
        let now = Instant::now();
        let mut quota = TrafficQuota::new(limits(), now).unwrap();
        let peer = PeerId::random();
        quota.check_and_charge(peer, 0, now).unwrap();
        quota.check_and_charge(peer, 0, now).unwrap();
        assert!(
            quota
                .check_and_charge(peer, 0, now - Duration::from_secs(1))
                .is_err()
        );
        assert_eq!(quota.retry_after(now), Duration::from_secs(10));
    }
}
