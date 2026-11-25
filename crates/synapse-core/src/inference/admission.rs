//! Atomic local memory/concurrency reservations, independent of discovery hints.
//! Every operation accepts a monotonic clock instant for deterministic tests.

use super::assignment::LeaseKey;
use anyhow::{Result, ensure};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseState {
    Preparing,
    Ready,
}

#[derive(Clone, Debug)]
pub struct CapacityLease {
    pub key: LeaseKey,
    pub owner: String,
    pub bytes: u64,
    pub expires_at: Instant,
    pub state: LeaseState,
}

#[derive(Default)]
struct Entries {
    leases: BTreeMap<LeaseKey, CapacityLease>,
}

#[derive(Clone)]
pub struct CapacityPool {
    capacity_bytes: u64,
    max_jobs: usize,
    entries: Arc<Mutex<Entries>>,
}

impl CapacityPool {
    pub fn new(capacity_bytes: u64, max_jobs: usize) -> Result<Self> {
        ensure!(
            capacity_bytes > 0 && (1..=64).contains(&max_jobs),
            "invalid worker capacity"
        );
        Ok(Self {
            capacity_bytes,
            max_jobs,
            entries: Arc::new(Mutex::new(Entries::default())),
        })
    }

    pub fn reserve(
        &self,
        key: LeaseKey,
        owner: &str,
        bytes: u64,
        ttl: Duration,
        now: Instant,
    ) -> Result<CapacityLease> {
        key.validate()?;
        ensure!(
            !owner.is_empty() && owner.len() <= 128 && bytes > 0,
            "invalid lease owner or reservation"
        );
        ensure!(
            !ttl.is_zero() && ttl <= Duration::from_secs(300),
            "invalid lease TTL"
        );
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| anyhow::anyhow!("admission lock poisoned"))?;
        entries.leases.retain(|_, lease| lease.expires_at > now);
        if let Some(existing) = entries.leases.get(&key) {
            ensure!(
                existing.owner == owner && existing.bytes == bytes,
                "conflicting duplicate lease"
            );
            return Ok(existing.clone()); // Retries never extend expiration.
        }
        ensure!(
            entries.leases.len() < self.max_jobs,
            "worker concurrency limit reached"
        );
        let used = entries
            .leases
            .values()
            .try_fold(0_u64, |total, lease| total.checked_add(lease.bytes))
            .ok_or_else(|| anyhow::anyhow!("reservation total overflow"))?;
        ensure!(
            used.checked_add(bytes)
                .is_some_and(|total| total <= self.capacity_bytes),
            "insufficient unreserved memory"
        );
        let lease = CapacityLease {
            key: key.clone(),
            owner: owner.to_owned(),
            bytes,
            expires_at: now + ttl,
            state: LeaseState::Preparing,
        };
        entries.leases.insert(key, lease.clone());
        Ok(lease)
    }

    pub fn mark_ready(&self, key: &LeaseKey, owner: &str, now: Instant) -> Result<()> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| anyhow::anyhow!("admission lock poisoned"))?;
        let lease = entries
            .leases
            .get_mut(key)
            .ok_or_else(|| anyhow::anyhow!("unknown lease"))?;
        ensure!(
            lease.owner == owner && lease.expires_at > now,
            "unauthorized or expired lease"
        );
        lease.state = LeaseState::Ready;
        Ok(())
    }

    pub fn is_ready(&self, key: &LeaseKey, owner: &str, now: Instant) -> bool {
        self.entries.lock().ok().is_some_and(|entries| {
            entries.leases.get(key).is_some_and(|lease| {
                lease.owner == owner && lease.expires_at > now && lease.state == LeaseState::Ready
            })
        })
    }

    /// Missing keys are idempotent; a different owner can never free a lease.
    pub fn release(&self, key: &LeaseKey, owner: &str) -> Result<bool> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| anyhow::anyhow!("admission lock poisoned"))?;
        if let Some(lease) = entries.leases.get(key) {
            ensure!(lease.owner == owner, "lease belongs to another coordinator");
        }
        Ok(entries.leases.remove(key).is_some())
    }

    pub fn expire(&self, now: Instant) -> Result<Vec<LeaseKey>> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| anyhow::anyhow!("admission lock poisoned"))?;
        let expired: Vec<_> = entries
            .leases
            .iter()
            .filter(|(_, lease)| lease.expires_at <= now)
            .map(|(key, _)| key.clone())
            .collect();
        for key in &expired {
            entries.leases.remove(key);
        }
        Ok(expired)
    }

    pub fn reserved_bytes(&self) -> Result<u64> {
        let entries = self
            .entries
            .lock()
            .map_err(|_| anyhow::anyhow!("admission lock poisoned"))?;
        entries
            .leases
            .values()
            .try_fold(0_u64, |total, lease| total.checked_add(lease.bytes))
            .ok_or_else(|| anyhow::anyhow!("reservation total overflow"))
    }
}

/// Sequence accounting is scoped to one lease. It prevents duplicate decode
/// frames from appending the same KV state twice and rejects cross-session work.
#[derive(Clone, Debug, Default)]
pub struct CacheCursor {
    pub generated_step: usize,
    pub cached_tokens: usize,
    reset_trace: Option<String>,
}

impl CacheCursor {
    /// True means the caller must clear the cache; false is an idempotent retry.
    pub fn reset(&mut self, trace: &str) -> Result<bool> {
        if let Some(previous) = &self.reset_trace {
            ensure!(
                previous == trace,
                "a new reset requires a new session lease"
            );
            return Ok(false);
        }
        ensure!(self.cached_tokens == 0, "cannot reset an active session");
        self.reset_trace = Some(trace.to_owned());
        Ok(true)
    }

    pub fn validate_step(
        &self,
        position: usize,
        token_count: usize,
        context_limit: usize,
    ) -> Result<()> {
        ensure!(
            self.reset_trace.is_some(),
            "cache readiness has not been acknowledged"
        );
        ensure!(
            position == self.generated_step,
            "duplicate, missing or reordered activation step"
        );
        ensure!(
            token_count > 0 && (position == 0 || token_count == 1),
            "distributed speculative blocks require a separate verified protocol"
        );
        ensure!(
            self.cached_tokens
                .checked_add(token_count)
                .is_some_and(|total| total <= context_limit),
            "activation exceeds reserved context"
        );
        Ok(())
    }

    pub fn commit_step(&mut self, token_count: usize) {
        self.cached_tokens += token_count;
        self.generated_step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(number: u64) -> LeaseKey {
        LeaseKey {
            session_id: format!("{number:016x}"),
            route_epoch: 1,
            model_digest: "a".repeat(64),
        }
    }

    #[test]
    fn simultaneous_requests_cannot_overcommit_capacity() {
        let pool = CapacityPool::new(32, 4).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(16));
        let handles: Vec<_> = (0..16)
            .map(|number| {
                let pool = pool.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    pool.reserve(
                        key(number),
                        "owner",
                        8,
                        Duration::from_secs(10),
                        Instant::now(),
                    )
                    .is_ok()
                })
            })
            .collect();
        let successes = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .filter(|success| *success)
            .count();
        assert_eq!(successes, 4);
        assert_eq!(pool.reserved_bytes().unwrap(), 32);
    }

    #[test]
    fn retry_ready_expiry_and_owner_isolation() {
        let pool = CapacityPool::new(128, 1).unwrap();
        let now = Instant::now();
        let first = pool
            .reserve(key(1), "a", 64, Duration::from_secs(2), now)
            .unwrap();
        let retry = pool
            .reserve(
                key(1),
                "a",
                64,
                Duration::from_secs(200),
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(first.expires_at, retry.expires_at);
        assert!(
            pool.reserve(key(1), "b", 64, Duration::from_secs(1), now)
                .is_err()
        );
        assert!(
            pool.reserve(key(1), "a", 65, Duration::from_secs(1), now)
                .is_err()
        );
        assert!(!pool.is_ready(&key(1), "a", now));
        pool.mark_ready(&key(1), "a", now).unwrap();
        assert!(pool.is_ready(&key(1), "a", now));
        assert!(!pool.is_ready(&key(1), "b", now));
        assert!(pool.release(&key(1), "b").is_err());
        assert_eq!(
            pool.expire(now + Duration::from_secs(2)).unwrap(),
            vec![key(1)]
        );
        assert!(!pool.is_ready(&key(1), "a", now + Duration::from_secs(2)));
        assert!(!pool.release(&key(1), "a").unwrap());
    }

    #[test]
    fn reservation_arithmetic_does_not_wrap() {
        let pool = CapacityPool::new(u64::MAX, 2).unwrap();
        let now = Instant::now();
        pool.reserve(key(1), "a", u64::MAX - 1, Duration::from_secs(1), now)
            .unwrap();
        assert!(
            pool.reserve(key(2), "a", 2, Duration::from_secs(1), now)
                .is_err()
        );
    }

    #[test]
    fn duplicate_decode_and_reset_cannot_mutate_live_kv_state() {
        let mut cursor = CacheCursor::default();
        assert!(cursor.validate_step(0, 3, 8).is_err());
        assert!(cursor.reset("reset-1").unwrap());
        cursor.validate_step(0, 3, 8).unwrap();
        cursor.commit_step(3);
        assert!(!cursor.reset("reset-1").unwrap());
        assert!(cursor.reset("reset-2").is_err());
        assert!(cursor.validate_step(0, 3, 8).is_err());
        cursor.validate_step(1, 1, 8).unwrap();
        cursor.commit_step(1);
        assert!(cursor.validate_step(1, 1, 8).is_err());
        assert!(cursor.validate_step(2, 2, 8).is_err());
        assert!(cursor.validate_step(2, 1, 4).is_err());
    }
}
