//! Versioned shard preparation and the coordinator's all-worker ready barrier.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeaseKey {
    pub session_id: String,
    pub route_epoch: u64,
    pub model_digest: String,
}

impl LeaseKey {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.session_id.len() == 16 && self.session_id.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid session id"
        );
        ensure!(self.route_epoch > 0, "zero route epoch");
        ensure!(
            self.model_digest.len() == 64
                && self.model_digest.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid model manifest digest"
        );
        Ok(())
    }

    pub fn trace_prefix(&self) -> String {
        format!("s{}e{}-", self.session_id, self.route_epoch)
    }

    pub fn owns_trace(&self, trace_id: &str) -> bool {
        trace_id.starts_with(&self.trace_prefix())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardAssignment {
    pub protocol_version: u16,
    pub key: LeaseKey,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    pub context_tokens: u32,
    pub memory_limit_bytes: u64,
    pub ttl_ms: u32,
}

impl ShardAssignment {
    pub fn validate(&self) -> Result<()> {
        self.key.validate()?;
        ensure!(
            self.protocol_version == 1,
            "unsupported shard lifecycle version"
        );
        ensure!(
            self.total_layers > 0
                && self.total_layers <= 1024
                && self.layer_start < self.layer_end
                && self.layer_end <= self.total_layers,
            "invalid assigned layer range"
        );
        ensure!(
            self.context_tokens > 0 && self.context_tokens <= 4096,
            "invalid context cap"
        );
        ensure!(self.memory_limit_bytes > 0, "missing memory cap");
        ensure!(
            (1000..=300_000).contains(&self.ttl_ms),
            "lease TTL outside allowed bounds"
        );
        Ok(())
    }

    pub fn layer_range(&self) -> std::ops::Range<usize> {
        self.layer_start as usize..self.layer_end as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardReady {
    pub assignment: ShardAssignment,
    pub resident_weight_bytes: u64,
}

impl ShardReady {
    pub fn validate(&self) -> Result<()> {
        self.assignment.validate()?;
        ensure!(
            self.resident_weight_bytes > 0
                && self.resident_weight_bytes <= self.assignment.memory_limit_bytes,
            "invalid resident weight count"
        );
        Ok(())
    }
}

/// A stage is usable only after the exact assignment has been acknowledged.
/// Duplicate acknowledgments do not advance the barrier twice.
pub struct ReadinessBarrier {
    expected: BTreeMap<String, ShardAssignment>,
    ready: BTreeSet<String>,
    deadline: Instant,
}

impl ReadinessBarrier {
    pub fn new(
        expected: BTreeMap<String, ShardAssignment>,
        now: Instant,
        timeout: Duration,
    ) -> Result<Self> {
        ensure!(expected.len() <= 64, "too many route stages");
        ensure!(
            !timeout.is_zero() && timeout <= Duration::from_secs(300),
            "invalid readiness timeout"
        );
        for assignment in expected.values() {
            assignment.validate()?;
        }
        Ok(Self {
            expected,
            ready: BTreeSet::new(),
            deadline: now + timeout,
        })
    }

    pub fn acknowledge(&mut self, peer: &str, ready: &ShardReady, now: Instant) -> Result<bool> {
        ensure!(now < self.deadline, "shard preparation timed out");
        ensure!(
            self.expected.get(peer) == Some(&ready.assignment),
            "unexpected peer, epoch, model or resident range"
        );
        ensure!(
            ready.resident_weight_bytes > 0
                && ready.resident_weight_bytes <= ready.assignment.memory_limit_bytes,
            "worker resident weights exceed assignment cap"
        );
        self.ready.insert(peer.to_owned());
        Ok(self.is_complete())
    }

    pub fn is_complete(&self) -> bool {
        self.ready.len() == self.expected.len()
    }
    pub fn is_expired(&self, now: Instant) -> bool {
        now >= self.deadline
    }
    pub fn required(&self, peer: &str) -> bool {
        self.expected.contains_key(peer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assignment() -> ShardAssignment {
        ShardAssignment {
            protocol_version: 1,
            key: LeaseKey {
                session_id: "0123456789abcdef".into(),
                route_epoch: 1,
                model_digest: "a".repeat(64),
            },
            layer_start: 0,
            layer_end: 2,
            total_layers: 2,
            context_tokens: 128,
            memory_limit_bytes: 1024,
            ttl_ms: 30_000,
        }
    }

    #[test]
    fn barrier_requires_all_exact_resident_assignments() {
        let now = Instant::now();
        let expected = BTreeMap::from([("a".into(), assignment()), ("b".into(), assignment())]);
        let mut barrier = ReadinessBarrier::new(expected, now, Duration::from_secs(1)).unwrap();
        let ready = ShardReady {
            assignment: assignment(),
            resident_weight_bytes: 512,
        };
        assert!(!barrier.acknowledge("a", &ready, now).unwrap());
        assert!(!barrier.acknowledge("a", &ready, now).unwrap());
        let mut bad = ready.clone();
        bad.assignment.key.route_epoch += 1;
        assert!(barrier.acknowledge("b", &bad, now).is_err());
        assert!(barrier.acknowledge("stranger", &ready, now).is_err());
        assert!(barrier.acknowledge("b", &ready, now).unwrap());
    }

    #[test]
    fn timed_out_or_over_budget_worker_never_becomes_ready() {
        let now = Instant::now();
        let mut barrier = ReadinessBarrier::new(
            BTreeMap::from([("a".into(), assignment())]),
            now,
            Duration::from_secs(1),
        )
        .unwrap();
        let mut ready = ShardReady {
            assignment: assignment(),
            resident_weight_bytes: 2048,
        };
        assert!(barrier.acknowledge("a", &ready, now).is_err());
        ready.resident_weight_bytes = 512;
        assert!(
            barrier
                .acknowledge("a", &ready, now + Duration::from_secs(1))
                .is_err()
        );
        assert!(!barrier.is_complete());
    }
}
