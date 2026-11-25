use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::protocol::ProtocolError;
use rand::{RngCore, rngs::OsRng};
const PAIRING_CODE_LEN: usize = 12;
const MIN_PAIRING_CODE_LEN: usize = 8;
const MAX_PAIRING_CODE_LEN: usize = 32;
const ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

pub fn generate_pairing_code() -> String {
    let mut random = [0_u8; PAIRING_CODE_LEN];
    OsRng.fill_bytes(&mut random);

    random
        .iter()
        .map(|byte| ALPHABET[*byte as usize % ALPHABET.len()] as char)
        .collect()
}

pub fn hash_pairing_code(code: &str) -> String {
    let normalised = normalize_pairing_code(code);
    crate::hash::sha256_hex(normalised.as_bytes())
}

pub fn normalize_pairing_code(code: &str) -> String {
    code.chars()
        .filter(|c| !c.is_ascii_whitespace() && *c != '-')
        .flat_map(|c| c.to_uppercase())
        .collect()
}

pub fn verify_pairing_code(expected: &str, presented: &str) -> Result<(), ProtocolError> {
    let expected = normalize_pairing_code(expected);
    let presented = normalize_pairing_code(presented);

    validate_pairing_code(&expected)?;
    validate_pairing_code(&presented)?;

    if !constant_time_eq(expected.as_bytes(), presented.as_bytes()) {
        return Err(ProtocolError::AuthenticationFailed);
    }

    Ok(())
}

pub fn verify_pairing_code_hash(
    expected_code: &str,
    presented_hash: &str,
) -> Result<(), ProtocolError> {
    let expected_hash = hash_pairing_code(expected_code);
    if !constant_time_eq(expected_hash.as_bytes(), presented_hash.as_bytes()) {
        return Err(ProtocolError::AuthenticationFailed);
    }
    Ok(())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut diff = left.len() ^ right.len();
    for (&l, &r) in left.iter().zip(right.iter()) {
        diff |= usize::from(l ^ r);
    }
    diff == 0
}

pub fn validate_pairing_code(code: &str) -> Result<(), ProtocolError> {
    if code.len() < MIN_PAIRING_CODE_LEN || code.len() > MAX_PAIRING_CODE_LEN {
        return Err(ProtocolError::InvalidField {
            field: "pairing code",
            reason: "length outside accepted range",
        });
    }

    if !code.bytes().all(|byte| ALPHABET.contains(&byte)) {
        return Err(ProtocolError::InvalidField {
            field: "pairing code",
            reason: "contains unsupported characters",
        });
    }

    Ok(())
}

const MAX_PAIRING_ATTEMPTS: u8 = 5;
const LOCKOUT_BASE_DURATION: Duration = Duration::from_secs(30);

#[derive(Debug, Default)]
pub struct PairingRateLimiter {
    attempts: HashMap<String, PairingAttemptState>,
}

#[derive(Debug, Default)]
struct PairingAttemptState {
    count: u8,
    locked_until: Option<Instant>,
}

impl PairingRateLimiter {
    pub fn check(&self, peer_key: &str) -> Result<(), Duration> {
        if let Some(state) = self.attempts.get(peer_key)
            && let Some(locked_until) = state.locked_until
        {
            let now = Instant::now();
            if now < locked_until {
                return Err(locked_until - now);
            }
        }
        Ok(())
    }

    pub fn record_failure(&mut self, peer_key: &str) {
        let state = self.attempts.entry(peer_key.to_owned()).or_default();

        state.count = state.count.saturating_add(1);

        if state.count >= MAX_PAIRING_ATTEMPTS {
            let overflow = state.count - MAX_PAIRING_ATTEMPTS;
            let lockout = LOCKOUT_BASE_DURATION * 2u32.saturating_pow(overflow.into());
            state.locked_until = Some(Instant::now() + lockout);
        }
    }

    pub fn record_success(&mut self, peer_key: &str) {
        self.attempts.remove(peer_key);
    }

    pub fn cleanup(&mut self) {
        let now = Instant::now();
        self.attempts.retain(|_, state| {
            state.count > 0 && state.locked_until.is_none_or(|until| until > now)
        });
    }
}

/// Pairing codes never travel on the wire. This keyed proof is fresh for a
/// single challenge and bound to both authenticated transport identities.
fn authentication_payload(
    challenge: &crate::Challenge,
    coordinator: libp2p::PeerId,
    worker: libp2p::PeerId,
) -> Vec<u8> {
    let mut bytes = b"synapse:private-pairing:v1\0".to_vec();
    append_field(&mut bytes, crate::PROTOCOL_VERSION.as_bytes());
    append_field(&mut bytes, &coordinator.to_bytes());
    append_field(&mut bytes, &worker.to_bytes());
    append_field(&mut bytes, challenge.nonce.as_bytes());
    bytes.extend_from_slice(&challenge.timestamp.to_be_bytes());
    bytes
}

fn append_field(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u32).to_be_bytes());
    target.extend_from_slice(value);
}

fn pairing_mac(code: &str, payload: &[u8]) -> Result<hmac::Hmac<sha2::Sha256>, ProtocolError> {
    use hmac::Mac;
    let normalized = normalize_pairing_code(code);
    validate_pairing_code(&normalized)?;
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(normalized.as_bytes())
        .map_err(|_| ProtocolError::AuthenticationFailed)?;
    mac.update(payload);
    Ok(mac)
}

fn decode_hex_digest(value: &str) -> Result<[u8; 32], ProtocolError> {
    if value.len() != 64 || !value.is_ascii() {
        return Err(ProtocolError::AuthenticationFailed);
    }
    let mut result = [0_u8; 32];
    for (i, slot) in result.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[i * 2..i * 2 + 2], 16)
            .map_err(|_| ProtocolError::AuthenticationFailed)?;
    }
    Ok(result)
}

pub fn respond_to_challenge(
    identity: &crate::NodeIdentity,
    challenge: &crate::Challenge,
    pairing_code: &str,
    coordinator: libp2p::PeerId,
    now: u64,
) -> Result<crate::ChallengeResponse, ProtocolError> {
    use hmac::Mac;
    challenge.validate_at(now)?;
    let mut payload = authentication_payload(challenge, coordinator, identity.peer_id());
    let proof = pairing_mac(pairing_code, &payload)?.finalize().into_bytes();
    let pairing_code_hash = proof.iter().map(|byte| format!("{byte:02x}")).collect();
    payload.extend_from_slice(&proof);
    Ok(crate::ChallengeResponse {
        pairing_code_hash,
        nonce: challenge.nonce.clone(),
        signature: identity
            .sign(&payload)
            .map_err(|_| ProtocolError::AuthenticationFailed)?,
        public_key: identity.public_key_bytes(),
    })
}

/// Call only for a pending challenge, and consume that challenge on every
/// attempt. A valid proof cannot enroll a different peer or a different host.
pub fn verify_challenge_response(
    challenge: &crate::Challenge,
    response: &crate::ChallengeResponse,
    pairing_code: &str,
    coordinator: libp2p::PeerId,
    worker: libp2p::PeerId,
    now: u64,
) -> Result<(), ProtocolError> {
    use hmac::Mac;
    challenge.validate_at(now)?;
    response.validate()?;
    if response.nonce != challenge.nonce {
        return Err(ProtocolError::AuthenticationFailed);
    }
    let mut payload = authentication_payload(challenge, coordinator, worker);
    let proof = decode_hex_digest(&response.pairing_code_hash)?;
    pairing_mac(pairing_code, &payload)?
        .verify_slice(&proof)
        .map_err(|_| ProtocolError::AuthenticationFailed)?;
    payload.extend_from_slice(&proof);
    crate::NodeIdentity::verify_remote_peer(
        worker,
        &response.public_key,
        &payload,
        &response.signature,
    )
    .map_err(|_| ProtocolError::AuthenticationFailed)
}

/// Coordinator proof completes mutual authentication. A transport identity
/// alone must not authorize an unrelated discovered peer to use a worker.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoordinatorProof {
    pub nonce: String,
    pub proof: String,
}

pub fn coordinator_proof(
    challenge: &crate::Challenge,
    pairing_code: &str,
    coordinator: libp2p::PeerId,
    worker: libp2p::PeerId,
) -> Result<CoordinatorProof, ProtocolError> {
    use hmac::Mac;
    challenge.validate()?;
    let mut payload = authentication_payload(challenge, coordinator, worker);
    payload.extend_from_slice(b"\0coordinator-accepted");
    let proof = pairing_mac(pairing_code, &payload)?.finalize().into_bytes();
    Ok(CoordinatorProof {
        nonce: challenge.nonce.clone(),
        proof: proof.iter().map(|byte| format!("{byte:02x}")).collect(),
    })
}

pub fn verify_coordinator_proof(
    challenge: &crate::Challenge,
    proof: &CoordinatorProof,
    pairing_code: &str,
    coordinator: libp2p::PeerId,
    worker: libp2p::PeerId,
    now: u64,
) -> Result<(), ProtocolError> {
    use hmac::Mac;
    challenge.validate_at(now)?;
    if proof.nonce != challenge.nonce {
        return Err(ProtocolError::AuthenticationFailed);
    }
    let mut payload = authentication_payload(challenge, coordinator, worker);
    payload.extend_from_slice(b"\0coordinator-accepted");
    pairing_mac(pairing_code, &payload)?
        .verify_slice(&decode_hex_digest(&proof.proof)?)
        .map_err(|_| ProtocolError::AuthenticationFailed)
}

pub const INVITE_MAX_TTL_SECS: u64 = 24 * 60 * 60;
const INVITE_MAX_REDEMPTIONS: usize = 4096;

/// A recipient-bound, signed, one-time private-pool invitation. It contains no
/// pairing secret. Identity possession still requires an authenticated channel.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedInvite {
    pub version: u16,
    pub pool: String,
    pub issuer: String,
    pub recipient: String,
    pub nonce: String,
    pub issued_at: u64,
    pub expires_at: u64,
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Copy)]
pub struct InviteScope<'a> {
    pub pool: &'a str,
    pub issuer: libp2p::PeerId,
    pub recipient: libp2p::PeerId,
}

impl SignedInvite {
    pub fn issue(
        issuer: &crate::NodeIdentity,
        pool: &str,
        recipient: libp2p::PeerId,
        now: u64,
        ttl_secs: u64,
    ) -> Result<Self, ProtocolError> {
        if ttl_secs == 0 || ttl_secs > INVITE_MAX_TTL_SECS {
            return Err(ProtocolError::AuthenticationFailed);
        }
        let mut nonce = [0_u8; 32];
        OsRng.fill_bytes(&mut nonce);
        let mut invite = Self {
            version: 1,
            pool: pool.to_owned(),
            issuer: issuer.peer_id().to_string(),
            recipient: recipient.to_string(),
            nonce: nonce.iter().map(|byte| format!("{byte:02x}")).collect(),
            issued_at: now,
            expires_at: now
                .checked_add(ttl_secs)
                .ok_or(ProtocolError::AuthenticationFailed)?,
            public_key: issuer.public_key_bytes(),
            signature: Vec::new(),
        };
        invite.validate_fields()?;
        invite.signature = issuer
            .sign(&invite.signing_payload())
            .map_err(|_| ProtocolError::AuthenticationFailed)?;
        Ok(invite)
    }

    fn validate_fields(&self) -> Result<(), ProtocolError> {
        if self.version != 1
            || self.pool.is_empty()
            || self.pool.len() > 64
            || !self
                .pool
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
            || self.issuer.parse::<libp2p::PeerId>().is_err()
            || self.recipient.parse::<libp2p::PeerId>().is_err()
            || self.nonce.len() != 64
            || !self
                .nonce
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || self.expires_at <= self.issued_at
            || self.expires_at - self.issued_at > INVITE_MAX_TTL_SECS
            || self.public_key.len() > crate::protocol::MAX_PUBKEY_LEN
            || self.signature.len() > crate::protocol::MAX_SIGNATURE_LEN
        {
            return Err(ProtocolError::AuthenticationFailed);
        }
        Ok(())
    }

    /// Canonical binary encoding: fixed domain, big-endian integers, u32
    /// length-prefixed UTF-8 fields. Signature and key bytes are not signed.
    pub fn signing_payload(&self) -> Vec<u8> {
        let mut payload = b"synapse:private-invite:v1\0".to_vec();
        payload.extend_from_slice(&self.version.to_be_bytes());
        for field in [&self.pool, &self.issuer, &self.recipient, &self.nonce] {
            append_field(&mut payload, field.as_bytes());
        }
        payload.extend_from_slice(&self.issued_at.to_be_bytes());
        payload.extend_from_slice(&self.expires_at.to_be_bytes());
        payload
    }

    pub fn verify(&self, scope: InviteScope<'_>, now: u64) -> Result<(), ProtocolError> {
        self.validate_fields()?;
        if self.pool != scope.pool
            || self.issuer != scope.issuer.to_string()
            || self.recipient != scope.recipient.to_string()
            || now < self.issued_at
            || now >= self.expires_at
        {
            return Err(ProtocolError::AuthenticationFailed);
        }
        crate::NodeIdentity::verify_remote_peer(
            scope.issuer,
            &self.public_key,
            &self.signing_payload(),
            &self.signature,
        )
        .map_err(|_| ProtocolError::AuthenticationFailed)
    }
}

/// Replay ledger survives restarts. Expired invite IDs can be safely removed;
/// the signed expiry prevents them becoming valid again after pruning.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InviteRedemptions {
    redeemed: std::collections::BTreeMap<String, u64>,
}

impl InviteRedemptions {
    pub fn load(path: &std::path::Path) -> Result<Self, crate::identity::IdentityError> {
        use std::io::Read;
        let file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(error.into()),
        };
        let mut bytes = Vec::new();
        file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
        if bytes.len() > 1024 * 1024 {
            return Err(crate::identity::IdentityError::InvalidKey(
                "invite ledger exceeds limit".into(),
            ));
        }
        let ledger: Self = serde_json::from_slice(&bytes)
            .map_err(|e| crate::identity::IdentityError::InvalidKey(e.to_string()))?;
        if ledger.redeemed.len() > INVITE_MAX_REDEMPTIONS {
            return Err(crate::identity::IdentityError::InvalidKey(
                "too many invite redemptions".into(),
            ));
        }
        Ok(ledger)
    }

    pub fn redeem(
        &mut self,
        invite: &SignedInvite,
        scope: InviteScope<'_>,
        now: u64,
        path: &std::path::Path,
        revoked: &crate::identity::RevocationList,
    ) -> Result<(), ProtocolError> {
        invite.verify(scope, now)?;
        if revoked.contains(&scope.issuer) || revoked.contains(&scope.recipient) {
            return Err(ProtocolError::AuthenticationFailed);
        }
        let _lock = crate::identity::lock_private_state(path)
            .map_err(|_| ProtocolError::AuthenticationFailed)?;
        let mut next = Self::load(path).map_err(|_| ProtocolError::AuthenticationFailed)?;
        // Include the issuer so equal nonces from different pools cannot
        // interfere; the same issuer may never redeem the same nonce twice.
        let id = format!("{}:{}", invite.issuer, invite.nonce);
        if next.redeemed.contains_key(&id) {
            return Err(ProtocolError::AuthenticationFailed);
        }
        next.redeemed.retain(|_, expiry| *expiry > now);
        if next.redeemed.len() >= INVITE_MAX_REDEMPTIONS {
            return Err(ProtocolError::AuthenticationFailed);
        }
        next.redeemed.insert(id, invite.expires_at);
        let bytes = serde_json::to_vec(&next).map_err(|_| ProtocolError::AuthenticationFailed)?;
        crate::identity::write_private_state(path, &bytes)
            .map_err(|_| ProtocolError::AuthenticationFailed)?;
        *self = next;
        Ok(())
    }
}

#[cfg(test)]
mod security_tests {
    use super::*;
    use libp2p::PeerId;

    fn identities() -> (tempfile::TempDir, crate::NodeIdentity, crate::NodeIdentity) {
        let dir = tempfile::tempdir().unwrap();
        let host = crate::NodeIdentity::load_or_create_at(&dir.path().join("host.key")).unwrap();
        let worker =
            crate::NodeIdentity::load_or_create_at(&dir.path().join("worker.key")).unwrap();
        (dir, host, worker)
    }

    #[test]
    fn challenge_proof_rejects_expiry_wrong_secret_wrong_peer_and_replay() {
        let (_dir, host, worker) = identities();
        let challenge = crate::Challenge {
            nonce: "ab".repeat(16),
            timestamp: 100,
        };
        let response =
            respond_to_challenge(&worker, &challenge, "ABCDEFGHJKLM", host.peer_id(), 101).unwrap();
        assert!(
            verify_challenge_response(
                &challenge,
                &response,
                "ABCDEFGHJKLM",
                host.peer_id(),
                worker.peer_id(),
                101
            )
            .is_ok()
        );
        assert!(
            verify_challenge_response(
                &challenge,
                &response,
                "ABCDEFGHJKLM",
                host.peer_id(),
                worker.peer_id(),
                161
            )
            .is_err()
        );
        assert!(
            verify_challenge_response(
                &challenge,
                &response,
                "ABCDEFGHJKLN",
                host.peer_id(),
                worker.peer_id(),
                101
            )
            .is_err()
        );
        assert!(
            verify_challenge_response(
                &challenge,
                &response,
                "ABCDEFGHJKLM",
                PeerId::random(),
                worker.peer_id(),
                101
            )
            .is_err()
        );
        assert!(
            verify_challenge_response(
                &challenge,
                &response,
                "ABCDEFGHJKLM",
                host.peer_id(),
                PeerId::random(),
                101
            )
            .is_err()
        );
        let other = crate::Challenge {
            nonce: "cd".repeat(16),
            timestamp: 100,
        };
        assert!(
            verify_challenge_response(
                &other,
                &response,
                "ABCDEFGHJKLM",
                host.peer_id(),
                worker.peer_id(),
                101
            )
            .is_err()
        );
        assert_ne!(
            response.pairing_code_hash,
            hash_pairing_code("ABCDEFGHJKLM")
        );
        let proof = coordinator_proof(&challenge, "ABCDEFGHJKLM", host.peer_id(), worker.peer_id())
            .unwrap();
        assert!(
            verify_coordinator_proof(
                &challenge,
                &proof,
                "ABCDEFGHJKLM",
                host.peer_id(),
                worker.peer_id(),
                101
            )
            .is_ok()
        );
        assert!(
            verify_coordinator_proof(
                &challenge,
                &proof,
                "ABCDEFGHJKLN",
                host.peer_id(),
                worker.peer_id(),
                101
            )
            .is_err()
        );
        let reflection = CoordinatorProof {
            nonce: challenge.nonce.clone(),
            proof: response.pairing_code_hash,
        };
        assert!(
            verify_coordinator_proof(
                &challenge,
                &reflection,
                "ABCDEFGHJKLM",
                host.peer_id(),
                worker.peer_id(),
                101
            )
            .is_err()
        );
    }

    #[test]
    fn invite_redeems_once_and_stays_consumed_after_restart() {
        let (dir, host, worker) = identities();
        let invite =
            SignedInvite::issue(&host, "my-private-pool", worker.peer_id(), 100, 600).unwrap();
        let scope = InviteScope {
            pool: "my-private-pool",
            issuer: host.peer_id(),
            recipient: worker.peer_id(),
        };
        let path = dir.path().join("redeemed.json");
        let revoked = crate::identity::RevocationList::default();
        let mut ledger = InviteRedemptions::default();
        ledger.redeem(&invite, scope, 101, &path, &revoked).unwrap();
        assert!(ledger.redeem(&invite, scope, 102, &path, &revoked).is_err());
        let mut loaded = InviteRedemptions::load(&path).unwrap();
        assert!(loaded.redeem(&invite, scope, 103, &path, &revoked).is_err());
    }

    #[test]
    fn stolen_expired_wrong_pool_and_tampered_invites_fail() {
        let (_dir, host, worker) = identities();
        let mut invite = SignedInvite::issue(&host, "private", worker.peer_id(), 100, 600).unwrap();
        let scope = InviteScope {
            pool: "private",
            issuer: host.peer_id(),
            recipient: worker.peer_id(),
        };
        assert!(invite.verify(scope, 700).is_err());
        assert!(invite.verify(scope, 99).is_err());
        assert!(
            invite
                .verify(
                    InviteScope {
                        recipient: PeerId::random(),
                        ..scope
                    },
                    101
                )
                .is_err()
        );
        assert!(
            invite
                .verify(
                    InviteScope {
                        pool: "other",
                        ..scope
                    },
                    101
                )
                .is_err()
        );
        invite.expires_at += 1;
        assert!(invite.verify(scope, 101).is_err());
    }

    #[test]
    fn revoked_recipient_and_failed_persistence_cannot_enroll() {
        let (dir, host, worker) = identities();
        let invite = SignedInvite::issue(&host, "private", worker.peer_id(), 100, 600).unwrap();
        let scope = InviteScope {
            pool: "private",
            issuer: host.peer_id(),
            recipient: worker.peer_id(),
        };
        let mut revoked = crate::identity::RevocationList::default();
        let mut ledger = InviteRedemptions::default();
        assert!(
            ledger
                .redeem(&invite, scope, 101, dir.path(), &revoked)
                .is_err()
        );
        assert!(ledger.redeemed.is_empty());
        revoked
            .revoke(worker.peer_id(), &dir.path().join("revoked.json"))
            .unwrap();
        assert!(
            ledger
                .redeem(
                    &invite,
                    scope,
                    101,
                    &dir.path().join("ledger.json"),
                    &revoked
                )
                .is_err()
        );
    }

    #[test]
    fn separate_redeemers_cannot_spend_the_same_invite() {
        let (dir, host, worker) = identities();
        let invite = SignedInvite::issue(&host, "private", worker.peer_id(), 100, 600).unwrap();
        let scope = InviteScope {
            pool: "private",
            issuer: host.peer_id(),
            recipient: worker.peer_id(),
        };
        let path = dir.path().join("ledger.json");
        let mut first = InviteRedemptions::default();
        let mut second = InviteRedemptions::default();
        let revoked = crate::identity::RevocationList::default();
        first.redeem(&invite, scope, 101, &path, &revoked).unwrap();
        assert!(second.redeem(&invite, scope, 102, &path, &revoked).is_err());
    }
}
