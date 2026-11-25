//! Authenticated, signed, expiring evidence that an assigned range is resident.
//! Discovery hints never enter this registry. No record grants compute access;
//! every activation still requires the current owner-bound admission lease.
use super::assignment::{LeaseKey, ShardReady};
use anyhow::{Result, ensure};
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MAX_RECORDS: usize = 64;
const MAX_CLOCK_SKEW_SECS: u64 = 5;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedAvailability {
    pub schema_version: u16,
    pub peer_id: String,
    pub key: LeaseKey,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    pub backend: String,
    pub resident_weight_bytes: u64,
    pub sequence: u64,
    pub issued_at_unix: u64,
    pub expires_at_unix: u64,
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}
impl SignedAvailability {
    fn signing_bytes(&self) -> Result<Vec<u8>> {
        // Struct field order plus integer/string/byte-array primitives form a
        // deterministic versioned transcript. Signature is deliberately empty.
        let mut unsigned = self.clone();
        unsigned.signature.clear();
        let mut bytes = b"synapse-ready-range-v1\0".to_vec();
        bytes.extend(serde_json::to_vec(&unsigned)?);
        Ok(bytes)
    }
    pub fn sign(
        identity: &crate::NodeIdentity,
        ready: &ShardReady,
        backend: &str,
        sequence: u64,
        now: u64,
        expires_at_unix: u64,
    ) -> Result<Self> {
        let a = &ready.assignment;
        let mut record = Self {
            schema_version: 1,
            peer_id: identity.peer_id().to_string(),
            key: a.key.clone(),
            layer_start: a.layer_start,
            layer_end: a.layer_end,
            total_layers: a.total_layers,
            backend: backend.into(),
            resident_weight_bytes: ready.resident_weight_bytes,
            sequence,
            issued_at_unix: now,
            expires_at_unix,
            public_key: identity.public_key_bytes(),
            signature: Vec::new(),
        };
        record.signature = identity.sign(&record.signing_bytes()?)?;
        record.validate()?;
        Ok(record)
    }
    pub fn validate(&self) -> Result<()> {
        self.key.validate()?;
        ensure!(
            self.schema_version == 1
                && self.peer_id.len() <= 128
                && self.peer_id.parse::<PeerId>().is_ok(),
            "invalid availability version or peer"
        );
        ensure!(
            self.total_layers > 0
                && self.total_layers <= 1024
                && self.layer_start < self.layer_end
                && self.layer_end <= self.total_layers,
            "invalid available range"
        );
        ensure!(
            matches!(self.backend.as_str(), "llama-safetensors" | "llama-gguf"),
            "unsupported available backend"
        );
        ensure!(
            self.resident_weight_bytes > 0 && self.sequence > 0,
            "missing residency or sequence"
        );
        ensure!(
            self.expires_at_unix > self.issued_at_unix
                && self.expires_at_unix - self.issued_at_unix <= 300,
            "invalid availability lifetime"
        );
        ensure!(
            !self.public_key.is_empty()
                && self.public_key.len() <= 128
                && self.signature.len() == 64,
            "invalid availability signature size"
        );
        Ok(())
    }
    pub fn matches_ready(&self, ready: &ShardReady) -> bool {
        let a = &ready.assignment;
        self.key == a.key
            && self.layer_start == a.layer_start
            && self.layer_end == a.layer_end
            && self.total_layers == a.total_layers
            && self.resident_weight_bytes == ready.resident_weight_bytes
    }
    pub fn verify(&self, peer: PeerId, expected_model_digest: &str, now: u64) -> Result<()> {
        self.validate()?;
        ensure!(
            self.peer_id == peer.to_string() && self.key.model_digest == expected_model_digest,
            "availability identity/model mismatch"
        );
        ensure!(
            self.issued_at_unix <= now.saturating_add(MAX_CLOCK_SKEW_SECS)
                && self.expires_at_unix > now,
            "expired or future availability"
        );
        crate::NodeIdentity::verify_remote_peer(
            peer,
            &self.public_key,
            &self.signing_bytes()?,
            &self.signature,
        )?;
        Ok(())
    }
}

#[derive(Default)]
pub struct ReadyRegistry {
    records: BTreeMap<String, SignedAvailability>,
    // Keep watermarks after disconnect/release to reject replays. Bounded by
    // the private pool size; churn cannot grow an unbounded tombstone map.
    highest_sequence: BTreeMap<String, u64>,
}
impl ReadyRegistry {
    pub fn accept(
        &mut self,
        peer: PeerId,
        record: SignedAvailability,
        authenticated_member: bool,
        expected_model_digest: &str,
        now: u64,
    ) -> Result<()> {
        ensure!(
            authenticated_member,
            "availability publisher is not a private member"
        );
        record.verify(peer, expected_model_digest, now)?;
        self.expire(now);
        let id = peer.to_string();
        ensure!(
            self.highest_sequence.contains_key(&id) || self.highest_sequence.len() < MAX_RECORDS,
            "ready registry private pool capacity exceeded"
        );
        if let Some(previous) = self.highest_sequence.get(&id) {
            ensure!(
                record.sequence > *previous,
                "replayed or out-of-order availability"
            );
        }
        self.highest_sequence.insert(id.clone(), record.sequence);
        self.records.insert(id, record);
        Ok(())
    }
    pub fn ranges(&self, model_digest: &str, now: u64) -> Vec<SignedAvailability> {
        self.records
            .values()
            .filter(|r| r.key.model_digest == model_digest && r.expires_at_unix > now)
            .cloned()
            .collect()
    }
    pub fn contains(
        &self,
        peer: PeerId,
        key: &LeaseKey,
        range: &std::ops::Range<usize>,
        now: u64,
    ) -> bool {
        self.records.get(&peer.to_string()).is_some_and(|r| {
            &r.key == key
                && r.layer_start as usize == range.start
                && r.layer_end as usize == range.end
                && r.expires_at_unix > now
        })
    }
    pub fn remove_peer(&mut self, peer: PeerId) {
        self.records.remove(&peer.to_string());
    }
    pub fn release(&mut self, key: &LeaseKey) {
        self.records.retain(|_, r| &r.key != key);
    }
    pub fn expire(&mut self, now: u64) {
        self.records.retain(|_, r| r.expires_at_unix > now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::assignment::ShardAssignment;
    fn identity() -> (tempfile::TempDir, crate::NodeIdentity) {
        let d = tempfile::tempdir().unwrap();
        let identity =
            crate::NodeIdentity::load_or_create_at(&d.path().join("identity.key")).unwrap();
        (d, identity)
    }
    fn ready() -> ShardReady {
        ShardReady {
            assignment: ShardAssignment {
                protocol_version: 1,
                key: LeaseKey {
                    session_id: "0123456789abcdef".into(),
                    route_epoch: 1,
                    model_digest: "a".repeat(64),
                },
                layer_start: 0,
                layer_end: 2,
                total_layers: 2,
                context_tokens: 11,
                memory_limit_bytes: 65536,
                ttl_ms: 30_000,
            },
            resident_weight_bytes: 20_736,
            availability: None,
        }
    }
    #[test]
    fn signature_binds_peer_model_range_epoch_and_membership() {
        let (_d, id) = identity();
        let r = SignedAvailability::sign(&id, &ready(), "llama-safetensors", 1, 100, 130).unwrap();
        let mut registry = ReadyRegistry::default();
        assert!(
            registry
                .accept(id.peer_id(), r.clone(), false, &"a".repeat(64), 100)
                .is_err()
        );
        assert!(
            registry
                .accept(PeerId::random(), r.clone(), true, &"a".repeat(64), 100)
                .is_err()
        );
        for mutate in 0..3 {
            let mut bad = r.clone();
            match mutate {
                0 => bad.key.route_epoch += 1,
                1 => bad.layer_end = 1,
                _ => bad.key.model_digest = "b".repeat(64),
            };
            assert!(
                registry
                    .accept(id.peer_id(), bad, true, &"a".repeat(64), 100)
                    .is_err()
            );
        }
        registry
            .accept(id.peer_id(), r.clone(), true, &"a".repeat(64), 100)
            .unwrap();
        assert!(registry.contains(id.peer_id(), &r.key, &(0..2), 100));
        assert_eq!(registry.ranges(&"a".repeat(64), 100).len(), 1);
    }
    #[test]
    fn expired_future_replayed_and_disconnected_records_are_ineligible() {
        let (_d, id) = identity();
        let r = SignedAvailability::sign(&id, &ready(), "llama-safetensors", 3, 100, 130).unwrap();
        assert!(r.verify(id.peer_id(), &"a".repeat(64), 94).is_err());
        assert!(r.verify(id.peer_id(), &"a".repeat(64), 130).is_err());
        let mut registry = ReadyRegistry::default();
        registry
            .accept(id.peer_id(), r.clone(), true, &"a".repeat(64), 100)
            .unwrap();
        registry.remove_peer(id.peer_id());
        assert!(registry.ranges(&"a".repeat(64), 100).is_empty());
        assert!(
            registry
                .accept(id.peer_id(), r, true, &"a".repeat(64), 100)
                .is_err()
        );
        let next =
            SignedAvailability::sign(&id, &ready(), "llama-safetensors", 4, 101, 131).unwrap();
        registry
            .accept(id.peer_id(), next, true, &"a".repeat(64), 101)
            .unwrap();
        registry.expire(131);
        assert!(registry.ranges(&"a".repeat(64), 131).is_empty());
    }
    #[test]
    fn concurrent_refresh_never_regresses_the_signed_sequence() {
        let (_d, id) = identity();
        let mut records = Vec::new();
        for seq in 1..17 {
            records.push(
                SignedAvailability::sign(&id, &ready(), "llama-safetensors", seq, 100, 130)
                    .unwrap(),
            );
        }
        let registry = std::sync::Arc::new(std::sync::Mutex::new(ReadyRegistry::default()));
        let handles: Vec<_> = records
            .into_iter()
            .map(|r| {
                let registry = registry.clone();
                let peer = id.peer_id();
                std::thread::spawn(move || {
                    let _ = registry
                        .lock()
                        .unwrap()
                        .accept(peer, r, true, &"a".repeat(64), 100);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            registry.lock().unwrap().ranges(&"a".repeat(64), 100)[0].sequence,
            16
        );
    }
}
