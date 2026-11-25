//! Durable route epochs and bounded recovery progress. This module stores no
//! prompts, tensor bytes or KV state; the admission book owns those resources.
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session transition is invalid")]
    Transition,
    #[error("session ownership or route epoch mismatch")]
    Scope,
    #[error("session deadline expired")]
    Expired,
    #[error("duplicate or out-of-order output")]
    OutputOrder,
    #[error("recovery budget exceeded")]
    RecoveryLimit,
    #[error("durable epoch storage failed: {0}")]
    Storage(#[from] std::io::Error),
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EpochJournal {
    version: u16,
    last_epoch: u64,
}

pub struct EpochCounter;
impl EpochCounter {
    /// Commit a new epoch before returning it. A crash may skip a value but can
    /// never reuse an acknowledged value. The lock serializes independent
    /// processes sharing the same device identity; corruption fails closed.
    pub fn advance(path: &Path) -> Result<u64, SessionError> {
        let _lock = crate::identity::lock_private_state(path)?;
        let current = match crate::identity::read_private_state(path, 4096) {
            Ok(bytes) => {
                let record: EpochJournal =
                    serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
                if record.version != 1 {
                    return Err(std::io::Error::other("unsupported epoch journal version").into());
                }
                record.last_epoch
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error.into()),
        };
        let next = current
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("route epoch exhausted"))?;
        let bytes = serde_json::to_vec(&EpochJournal {
            version: 1,
            last_epoch: next,
        })
        .map_err(std::io::Error::other)?;
        crate::identity::write_private_state(path, &bytes)?;
        Ok(next)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Preparing,
    Running,
    Recovering,
    Completed,
    Cancelled,
    Failed,
    Expired,
}
impl SessionState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::Failed | Self::Expired
        )
    }
}

#[derive(Debug)]
pub struct SessionProgress {
    owner: PeerId,
    session_id: String,
    route_epoch: u64,
    state: SessionState,
    deadline: Instant,
    delivered: usize,
    max_output: usize,
    recoveries: u8,
    replayed: usize,
    replay_required: usize,
}
impl SessionProgress {
    pub fn new(
        owner: PeerId,
        session_id: String,
        route_epoch: u64,
        max_output: usize,
        deadline: Instant,
        now: Instant,
    ) -> Result<Self, SessionError> {
        if route_epoch == 0
            || session_id.is_empty()
            || session_id.len() > 64
            || !session_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"-_".contains(&byte))
            || max_output == 0
            || max_output > crate::protocol::MAX_SEQUENCE_LEN
            || deadline <= now
        {
            return Err(SessionError::Transition);
        }
        Ok(Self {
            owner,
            session_id,
            route_epoch,
            state: SessionState::Preparing,
            deadline,
            delivered: 0,
            max_output,
            recoveries: 0,
            replayed: 0,
            replay_required: 0,
        })
    }
    pub fn state(&self) -> SessionState {
        self.state
    }
    pub fn delivered(&self) -> usize {
        self.delivered
    }
    pub fn route_epoch(&self) -> u64 {
        self.route_epoch
    }
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    fn ensure_live(&mut self, now: Instant) -> Result<(), SessionError> {
        if !self.state.is_terminal() && now >= self.deadline {
            self.state = SessionState::Expired;
        }
        if self.state == SessionState::Expired {
            return Err(SessionError::Expired);
        }
        if self.state.is_terminal() {
            return Err(SessionError::Transition);
        }
        Ok(())
    }
    pub fn authorize(
        &mut self,
        owner: PeerId,
        epoch: u64,
        now: Instant,
    ) -> Result<(), SessionError> {
        self.ensure_live(now)?;
        if owner != self.owner || epoch != self.route_epoch {
            return Err(SessionError::Scope);
        }
        Ok(())
    }
    pub fn commit(&mut self, owner: PeerId, epoch: u64, now: Instant) -> Result<(), SessionError> {
        self.authorize(owner, epoch, now)?;
        if self.state != SessionState::Preparing {
            return Err(SessionError::Transition);
        }
        self.state = SessionState::Running;
        Ok(())
    }
    /// Call only AFTER the output channel has accepted the token. Delivery
    /// credit cannot advance on a full/closed stream or during replay.
    pub fn delivered_token(
        &mut self,
        epoch: u64,
        index: usize,
        now: Instant,
    ) -> Result<(), SessionError> {
        self.ensure_live(now)?;
        if epoch != self.route_epoch {
            return Err(SessionError::Scope);
        }
        if self.state != SessionState::Running {
            return Err(SessionError::Transition);
        }
        if index != self.delivered || self.delivered >= self.max_output {
            return Err(SessionError::OutputOrder);
        }
        self.delivered += 1;
        Ok(())
    }
    pub fn begin_recovery(
        &mut self,
        next_epoch: u64,
        replay_tokens: usize,
        now: Instant,
    ) -> Result<(), SessionError> {
        self.ensure_live(now)?;
        if self.state != SessionState::Running || next_epoch <= self.route_epoch {
            return Err(SessionError::Transition);
        }
        if self.recoveries >= 2
            || replay_tokens > crate::protocol::MAX_SEQUENCE_LEN
            || replay_tokens < self.delivered
        {
            self.state = SessionState::Failed;
            return Err(SessionError::RecoveryLimit);
        }
        self.recoveries += 1;
        self.route_epoch = next_epoch;
        self.replay_required = replay_tokens;
        self.replayed = 0;
        self.state = SessionState::Recovering;
        Ok(())
    }
    /// Replay rebuilds KV only; it never emits output or increments delivered.
    pub fn replay_progress(
        &mut self,
        epoch: u64,
        completed_tokens: usize,
        now: Instant,
    ) -> Result<(), SessionError> {
        self.ensure_live(now)?;
        if self.state != SessionState::Recovering || epoch != self.route_epoch {
            return Err(SessionError::Scope);
        }
        if completed_tokens < self.replayed || completed_tokens > self.replay_required {
            return Err(SessionError::OutputOrder);
        }
        self.replayed = completed_tokens;
        Ok(())
    }
    pub fn resume(&mut self, now: Instant) -> Result<(), SessionError> {
        self.ensure_live(now)?;
        if self.state != SessionState::Recovering || self.replayed != self.replay_required {
            return Err(SessionError::Transition);
        }
        self.state = SessionState::Running;
        Ok(())
    }
    pub fn complete(&mut self, now: Instant) -> Result<(), SessionError> {
        self.ensure_live(now)?;
        if self.state != SessionState::Running {
            return Err(SessionError::Transition);
        }
        self.state = SessionState::Completed;
        Ok(())
    }
    pub fn cancel(&mut self) {
        if !self.state.is_terminal() {
            self.state = SessionState::Cancelled;
        }
    }
    pub fn fail(&mut self) {
        if !self.state.is_terminal() {
            self.state = SessionState::Failed;
        }
    }
    pub fn expire(&mut self, now: Instant) -> bool {
        let _ = self.ensure_live(now);
        self.state == SessionState::Expired
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    fn progress(now: Instant) -> (PeerId, SessionProgress) {
        let owner = PeerId::random();
        (
            owner,
            SessionProgress::new(
                owner,
                "session-1".into(),
                1,
                10,
                now + Duration::from_secs(30),
                now,
            )
            .unwrap(),
        )
    }
    #[test]
    fn epoch_journal_never_reuses_an_acknowledged_epoch_and_rejects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("epochs.json");
        assert_eq!(EpochCounter::advance(&path).unwrap(), 1);
        assert_eq!(EpochCounter::advance(&path).unwrap(), 2);
        std::fs::write(&path, "broken").unwrap();
        assert!(EpochCounter::advance(&path).is_err());
        std::fs::write(&path, r#"{"version":1,"last_epoch":18446744073709551615}"#).unwrap();
        assert!(EpochCounter::advance(&path).is_err());
    }
    #[test]
    fn cross_session_owner_and_stale_epoch_cannot_access_progress() {
        let now = Instant::now();
        let (owner, mut session) = progress(now);
        assert!(session.commit(PeerId::random(), 1, now).is_err());
        session.commit(owner, 1, now).unwrap();
        session.delivered_token(1, 0, now).unwrap();
        assert!(session.delivered_token(1, 0, now).is_err());
        session.begin_recovery(2, 5, now).unwrap();
        assert!(session.delivered_token(1, 1, now).is_err());
        assert!(session.delivered_token(2, 1, now).is_err());
        session.replay_progress(2, 5, now).unwrap();
        session.resume(now).unwrap();
        assert_eq!(session.delivered(), 1);
        session.delivered_token(2, 1, now).unwrap();
        assert_eq!(session.delivered(), 2);
    }
    #[test]
    fn cancelled_and_expired_sessions_never_restart() {
        let now = Instant::now();
        let (owner, mut session) = progress(now);
        session.commit(owner, 1, now).unwrap();
        session.cancel();
        session.cancel();
        assert!(session.begin_recovery(2, 0, now).is_err());
        assert!(session.delivered_token(1, 0, now).is_err());
        let (_, mut expired) = progress(now);
        assert!(expired.expire(now + Duration::from_secs(30)));
        assert!(
            expired
                .commit(owner, 1, now + Duration::from_secs(30))
                .is_err()
        );
    }
    #[test]
    fn repeated_recovery_is_bounded_and_requires_complete_replay() {
        let now = Instant::now();
        let (owner, mut session) = progress(now);
        session.commit(owner, 1, now).unwrap();
        for epoch in [2, 3] {
            session.begin_recovery(epoch, 5, now).unwrap();
            assert!(session.resume(now).is_err());
            session.replay_progress(epoch, 5, now).unwrap();
            session.resume(now).unwrap();
        }
        assert!(matches!(
            session.begin_recovery(4, 5, now),
            Err(SessionError::RecoveryLimit)
        ));
        assert_eq!(session.state(), SessionState::Failed);
    }
}
