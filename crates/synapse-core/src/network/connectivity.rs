//! Bounded private WAN connectivity policies. Transport authentication remains
//! end-to-end when a Circuit Relay v2 server forwards a connection.

use libp2p::{Multiaddr, PeerId, multiaddr::Protocol};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RelayLimits {
    pub max_reservations: usize,
    pub max_reservations_per_peer: usize,
    pub max_circuits: usize,
    pub max_circuits_per_peer: usize,
    pub reservation_duration_secs: u64,
    pub max_circuit_duration_secs: u64,
    pub max_circuit_bytes: u64,
}

impl Default for RelayLimits {
    fn default() -> Self {
        Self {
            max_reservations: 32,
            max_reservations_per_peer: 2,
            max_circuits: 8,
            max_circuits_per_peer: 2,
            reservation_duration_secs: 15 * 60,
            max_circuit_duration_secs: 10 * 60,
            max_circuit_bytes: 256 * 1024 * 1024,
        }
    }
}

impl RelayLimits {
    pub fn to_libp2p_config(&self) -> Result<libp2p::relay::Config, String> {
        if self.max_reservations == 0
            || self.max_reservations > 1024
            || self.max_reservations_per_peer == 0
            || self.max_reservations_per_peer > self.max_reservations
            || self.max_circuits == 0
            || self.max_circuits > 128
            || self.max_circuits_per_peer == 0
            || self.max_circuits_per_peer > self.max_circuits
            || self.reservation_duration_secs == 0
            || self.reservation_duration_secs > 3600
            || self.max_circuit_duration_secs == 0
            || self.max_circuit_duration_secs > 3600
            || self.max_circuit_bytes == 0
            || self.max_circuit_bytes > 4 * 1024 * 1024 * 1024
        {
            return Err("relay limits exceed the supported private-network bounds".into());
        }
        Ok(libp2p::relay::Config {
            max_reservations: self.max_reservations,
            max_reservations_per_peer: self.max_reservations_per_peer,
            max_circuits: self.max_circuits,
            max_circuits_per_peer: self.max_circuits_per_peer,
            reservation_duration: Duration::from_secs(self.reservation_duration_secs),
            max_circuit_duration: Duration::from_secs(self.max_circuit_duration_secs),
            max_circuit_bytes: self.max_circuit_bytes,
            ..Default::default()
        })
    }
}

/// Build a route through an explicitly configured relay, pinned to the final
/// destination. No arbitrary nested circuits or unpinned peer IDs are accepted.
pub fn circuit_address(relay: &Multiaddr, destination: PeerId) -> Result<Multiaddr, String> {
    let relay_peer =
        crate::netconfig::validate_direct_peer_addr(relay).map_err(|e| e.to_string())?;
    if relay_peer == destination {
        return Err("relay cannot also be the circuit destination".into());
    }
    Ok(relay
        .clone()
        .with(Protocol::P2pCircuit)
        .with(Protocol::P2p(destination)))
}

/// Prefer actual RTT samples; absent samples remain unknown and fall back to
/// the administrator's configured order, never fabricated enqueue timings.
pub fn relay_candidates(relays: &[crate::RelayInfo], limit: usize) -> Vec<Multiaddr> {
    let mut candidates: Vec<_> = relays
        .iter()
        .enumerate()
        .filter(|(_, info)| crate::netconfig::validate_direct_peer_addr(&info.addr).is_ok())
        .collect();
    candidates.sort_by_key(|(index, info)| {
        (
            info.measured_latency_ms.is_none(),
            info.measured_latency_ms.unwrap_or(u64::MAX),
            *index,
        )
    });
    candidates
        .into_iter()
        .take(limit.min(2))
        .map(|(_, info)| info.addr.clone())
        .collect()
}

/// Bounded retry state, driven from the event loop's timer. A failed dial does
/// not recursively redial or wait for a ConnectionClosed event that may never
/// happen. Delays are monotonic and exhaustion is explicit.
#[derive(Clone, Debug)]
pub struct DialBackoff {
    failures: u32,
    next_attempt: Option<Instant>,
    max_attempts: u32,
}

impl Default for DialBackoff {
    fn default() -> Self {
        Self {
            failures: 0,
            next_attempt: None,
            max_attempts: 10,
        }
    }
}

impl DialBackoff {
    pub fn failed(&mut self, now: Instant) -> Option<Duration> {
        self.failures = self.failures.saturating_add(1);
        if self.failures >= self.max_attempts {
            self.next_attempt = None;
            return None;
        }
        let delay = Duration::from_secs(1_u64 << self.failures.saturating_sub(1).min(5))
            .min(Duration::from_secs(30));
        self.next_attempt = now.checked_add(delay);
        Some(delay)
    }
    pub fn is_due(&self, now: Instant) -> bool {
        !self.exhausted() && self.next_attempt.is_some_and(|at| at <= now)
    }
    pub fn exhausted(&self) -> bool {
        self.failures >= self.max_attempts
    }
    pub fn connected(&mut self) {
        self.failures = 0;
        self.next_attempt = None;
    }
}

pub fn connection_limits() -> libp2p::connection_limits::Behaviour {
    use libp2p::connection_limits::{Behaviour, ConnectionLimits};
    Behaviour::new(
        ConnectionLimits::default()
            .with_max_pending_incoming(Some(16))
            .with_max_pending_outgoing(Some(16))
            .with_max_established(Some(32))
            .with_max_established_per_peer(Some(2)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_limits_are_enforced_by_libp2p_configuration() {
        let limits = RelayLimits::default();
        let config = limits.to_libp2p_config().unwrap();
        assert_eq!(config.max_circuits, 8);
        assert_eq!(config.max_circuits_per_peer, 2);
        assert_eq!(config.max_circuit_bytes, 256 * 1024 * 1024);
        assert!(!config.reservation_rate_limiters.is_empty());
        assert!(
            RelayLimits {
                max_circuits_per_peer: 9,
                ..limits
            }
            .to_libp2p_config()
            .is_err()
        );
    }

    #[test]
    fn relayed_route_preserves_relay_and_destination_identity() {
        let relay = PeerId::random();
        let target = PeerId::random();
        let addr: Multiaddr = format!("/ip4/127.0.0.1/tcp/4001/p2p/{relay}")
            .parse()
            .unwrap();
        assert_eq!(
            circuit_address(&addr, target).unwrap().to_string(),
            format!("{addr}/p2p-circuit/p2p/{target}")
        );
        assert!(circuit_address(&addr, relay).is_err());
        assert!(circuit_address(&addr.clone().with(Protocol::P2pCircuit), target).is_err());
    }

    #[test]
    fn unknown_rtt_is_never_presented_as_zero_latency() {
        let a = crate::RelayInfo::new(
            format!("/ip4/127.0.0.1/tcp/4001/p2p/{}", PeerId::random())
                .parse()
                .unwrap(),
        );
        let mut b = crate::RelayInfo::new(
            format!("/ip4/127.0.0.1/tcp/4002/p2p/{}", PeerId::random())
                .parse()
                .unwrap(),
        );
        b.measured_latency_ms = Some(25);
        assert_eq!(
            relay_candidates(&[a.clone(), b.clone()], 2),
            vec![b.addr, a.addr]
        );
    }

    #[test]
    fn retries_are_bounded_timer_driven_and_reset_after_connect() {
        let now = Instant::now();
        let mut retry = DialBackoff::default();
        assert_eq!(retry.failed(now), Some(Duration::from_secs(1)));
        assert!(!retry.is_due(now));
        assert!(retry.is_due(now + Duration::from_secs(1)));
        for _ in 1..10 {
            retry.failed(now);
        }
        assert!(retry.exhausted());
        assert!(!retry.is_due(now + Duration::from_secs(100)));
        retry.connected();
        assert!(!retry.exhausted());
    }
}
