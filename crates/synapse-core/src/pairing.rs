use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::protocol::ProtocolError;
use rand::{RngCore, rngs::OsRng};
const PAIRING_CODE_LEN: usize = 8;
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
    if code.len() < PAIRING_CODE_LEN || code.len() > MAX_PAIRING_CODE_LEN {
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
