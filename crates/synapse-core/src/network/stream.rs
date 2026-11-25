//! Correlate WAN responses with exactly one live request. Authentication alone
//! is insufficient: an unrelated or delayed response must not advance a route.
use libp2p::PeerId;
use std::{collections::HashMap, hash::Hash, time::Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseKind {
    Activation,
    CacheClear,
    Truncate,
    ShardReady,
    ShardReleased,
    Pairing,
    Health,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponseScope {
    pub peer: PeerId,
    pub kind: ResponseKind,
    pub session_id: String,
    pub route_epoch: u64,
    pub trace_id: String,
}
impl ResponseScope {
    fn validate(&self) -> bool {
        [&self.session_id, &self.trace_id].iter().all(|value| {
            !value.is_empty()
                && value.len() <= crate::protocol::MAX_TRACE_ID_LEN
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_".contains(&byte))
        })
    }
}

#[derive(Clone, Debug)]
struct PendingResponse {
    scope: ResponseScope,
    deadline: Instant,
}

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum StreamError {
    #[error("too many pending network requests")]
    Capacity,
    #[error("request ID was already registered")]
    Duplicate,
    #[error("invalid response scope or deadline")]
    Invalid,
    #[error("response has no pending request")]
    Unsolicited,
    #[error("response arrived after its deadline")]
    Expired,
    #[error("response does not match peer, kind, session, epoch or trace")]
    ScopeMismatch,
}

#[derive(Debug)]
pub struct PendingRequests<K> {
    entries: HashMap<K, PendingResponse>,
    capacity: usize,
}
impl<K: Copy + Eq + Hash> PendingRequests<K> {
    pub fn new(capacity: usize) -> Result<Self, StreamError> {
        if capacity == 0 || capacity > 4096 {
            return Err(StreamError::Capacity);
        }
        Ok(Self {
            entries: HashMap::new(),
            capacity,
        })
    }
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }
    pub fn register(
        &mut self,
        id: K,
        scope: ResponseScope,
        deadline: Instant,
        now: Instant,
    ) -> Result<(), StreamError> {
        if !scope.validate() || deadline <= now {
            return Err(StreamError::Invalid);
        }
        if self.entries.contains_key(&id) {
            return Err(StreamError::Duplicate);
        }
        if self.is_full() {
            return Err(StreamError::Capacity);
        }
        self.entries.insert(id, PendingResponse { scope, deadline });
        Ok(())
    }
    /// Validate without consuming on a mismatched response, then remove on
    /// success. Replay has no live ID left to match and is rejected.
    pub fn complete(
        &mut self,
        id: K,
        scope: &ResponseScope,
        now: Instant,
    ) -> Result<(), StreamError> {
        let pending = self.entries.get(&id).ok_or(StreamError::Unsolicited)?;
        if now >= pending.deadline {
            self.entries.remove(&id);
            return Err(StreamError::Expired);
        }
        if &pending.scope != scope {
            return Err(StreamError::ScopeMismatch);
        }
        self.entries.remove(&id);
        Ok(())
    }
    pub fn expire(&mut self, now: Instant) -> Vec<ResponseScope> {
        let mut expired = Vec::new();
        self.entries.retain(|_, pending| {
            if pending.deadline <= now {
                expired.push(pending.scope.clone());
                false
            } else {
                true
            }
        });
        expired
    }
    pub fn cancel_session(&mut self, session: &str) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, pending| pending.scope.session_id != session);
        before - self.entries.len()
    }
    pub fn disconnect(&mut self, peer: PeerId) -> Vec<ResponseScope> {
        let mut removed = Vec::new();
        self.entries.retain(|_, pending| {
            if pending.scope.peer == peer {
                removed.push(pending.scope.clone());
                false
            } else {
                true
            }
        });
        removed
    }
}
impl<K: Copy + Eq + Hash> Default for PendingRequests<K> {
    fn default() -> Self {
        Self::new(128).expect("valid static request capacity")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    fn scope() -> ResponseScope {
        ResponseScope {
            peer: PeerId::random(),
            kind: ResponseKind::Activation,
            session_id: "session-1".into(),
            route_epoch: 1,
            trace_id: "token-0".into(),
        }
    }
    #[test]
    fn stale_unrelated_and_duplicate_responses_cannot_advance_a_request() {
        let now = Instant::now();
        let scope = scope();
        let mut requests = PendingRequests::new(2).unwrap();
        requests
            .register(1, scope.clone(), now + Duration::from_secs(5), now)
            .unwrap();
        for changed in [
            ResponseScope {
                route_epoch: 2,
                ..scope.clone()
            },
            ResponseScope {
                peer: PeerId::random(),
                ..scope.clone()
            },
            ResponseScope {
                kind: ResponseKind::Health,
                ..scope.clone()
            },
            ResponseScope {
                trace_id: "token-1".into(),
                ..scope.clone()
            },
        ] {
            assert_eq!(
                requests.complete(1, &changed, now),
                Err(StreamError::ScopeMismatch)
            );
        }
        requests.complete(1, &scope, now).unwrap();
        assert_eq!(
            requests.complete(1, &scope, now),
            Err(StreamError::Unsolicited)
        );
    }
    #[test]
    fn cancellation_and_timeout_clean_scoped_requests_without_cross_session_loss() {
        let now = Instant::now();
        let first = scope();
        let other = ResponseScope {
            session_id: "other".into(),
            ..first.clone()
        };
        let mut requests = PendingRequests::new(2).unwrap();
        requests
            .register(1, first.clone(), now + Duration::from_secs(1), now)
            .unwrap();
        requests
            .register(2, other.clone(), now + Duration::from_secs(2), now)
            .unwrap();
        assert_eq!(requests.cancel_session(&first.session_id), 1);
        assert_eq!(
            requests.complete(1, &first, now),
            Err(StreamError::Unsolicited)
        );
        assert_eq!(requests.expire(now + Duration::from_secs(2)), vec![other]);
        assert!(!requests.is_full());
    }
    #[test]
    fn full_queue_and_duplicate_id_never_replace_live_requests() {
        let now = Instant::now();
        let scope = scope();
        let mut requests = PendingRequests::new(1).unwrap();
        requests
            .register(1, scope.clone(), now + Duration::from_secs(5), now)
            .unwrap();
        assert_eq!(
            requests.register(1, scope.clone(), now + Duration::from_secs(5), now),
            Err(StreamError::Duplicate)
        );
        assert_eq!(
            requests.register(2, scope.clone(), now + Duration::from_secs(5), now),
            Err(StreamError::Capacity)
        );
        assert_eq!(
            requests.complete(1, &scope, now + Duration::from_secs(5)),
            Err(StreamError::Expired)
        );
    }
}
