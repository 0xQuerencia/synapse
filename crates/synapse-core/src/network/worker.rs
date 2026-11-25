//! Local worker controls. Commands are acknowledged only after the node has
//! applied cleanup; GUI/CLI controls cannot silently succeed while detached.
use serde::Serialize;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

const CONTROL_CAPACITY: usize = 8;
const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, serde::Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerAction {
    Pause,
    Resume,
    Drain,
    Stop,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    Available,
    Paused,
    Draining,
    Stopped,
}
#[derive(Clone, Debug, Serialize)]
pub struct WorkerSnapshot {
    pub state: WorkerState,
    pub active_jobs: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ControlError {
    #[error("worker is stopped; start a new worker to resume")]
    Stopped,
    #[error("worker command queue is unavailable")]
    Disconnected,
    #[error("worker did not acknowledge the command before its deadline")]
    Timeout,
}
#[derive(Debug)]
pub struct WorkerControlRequest {
    pub action: WorkerAction,
    reply: oneshot::Sender<Result<WorkerSnapshot, ControlError>>,
}
impl WorkerControlRequest {
    pub fn respond(self, result: Result<WorkerSnapshot, ControlError>) {
        let _ = self.reply.send(result);
    }
}
#[derive(Clone, Debug)]
pub struct WorkerControlHandle {
    sender: mpsc::Sender<WorkerControlRequest>,
}
impl WorkerControlHandle {
    pub async fn request(&self, action: WorkerAction) -> Result<WorkerSnapshot, ControlError> {
        tokio::time::timeout(CONTROL_TIMEOUT, async {
            let (reply, receive) = oneshot::channel();
            self.sender
                .send(WorkerControlRequest { action, reply })
                .await
                .map_err(|_| ControlError::Disconnected)?;
            receive.await.map_err(|_| ControlError::Disconnected)?
        })
        .await
        .map_err(|_| ControlError::Timeout)?
    }
}
pub fn worker_control_channel() -> (WorkerControlHandle, mpsc::Receiver<WorkerControlRequest>) {
    let (sender, receiver) = mpsc::channel(CONTROL_CAPACITY);
    (WorkerControlHandle { sender }, receiver)
}
pub async fn recv_control(
    receiver: &mut Option<mpsc::Receiver<WorkerControlRequest>>,
) -> Option<WorkerControlRequest> {
    match receiver {
        Some(receiver) => receiver.recv().await,
        None => std::future::pending().await,
    }
}

#[derive(Clone, Debug)]
pub struct ControlEffect {
    /// Node must release the current lease and clear KV/tensors before ACK.
    pub release_active: bool,
    /// Node must exit its event loop after sending the applied ACK.
    pub exit: bool,
    pub snapshot: WorkerSnapshot,
}
#[derive(Debug)]
pub struct WorkerController {
    state: WorkerState,
    drain_deadline: Option<Instant>,
}
impl Default for WorkerController {
    /// Constructed only for an explicitly started worker. Installation itself
    /// never starts contribution; service enablement belongs to the launcher.
    fn default() -> Self {
        Self {
            state: WorkerState::Available,
            drain_deadline: None,
        }
    }
}
impl WorkerController {
    pub fn allows_assignment(&self) -> bool {
        self.state == WorkerState::Available
    }
    pub fn snapshot(&self, active_jobs: usize) -> WorkerSnapshot {
        WorkerSnapshot {
            state: self.state,
            active_jobs,
        }
    }
    pub fn apply(
        &mut self,
        action: WorkerAction,
        active_jobs: usize,
        now: Instant,
    ) -> Result<ControlEffect, ControlError> {
        if self.state == WorkerState::Stopped && !matches!(action, WorkerAction::Stop) {
            return Err(ControlError::Stopped);
        }
        let mut release_active = false;
        let mut exit = false;
        match action {
            WorkerAction::Pause => {
                self.state = WorkerState::Paused;
                self.drain_deadline = None;
                release_active = active_jobs > 0;
            }
            WorkerAction::Resume => {
                self.state = WorkerState::Available;
                self.drain_deadline = None;
            }
            WorkerAction::Drain => {
                self.state = if active_jobs == 0 {
                    WorkerState::Paused
                } else {
                    WorkerState::Draining
                };
                self.drain_deadline = (active_jobs > 0).then_some(now + DRAIN_TIMEOUT);
            }
            WorkerAction::Stop => {
                self.state = WorkerState::Stopped;
                self.drain_deadline = None;
                release_active = active_jobs > 0;
                exit = true;
            }
        }
        Ok(ControlEffect {
            release_active,
            exit,
            snapshot: self.snapshot(if release_active { 0 } else { active_jobs }),
        })
    }
    pub fn tick(&mut self, active_jobs: usize, now: Instant) -> Option<ControlEffect> {
        if self.state != WorkerState::Draining {
            return None;
        }
        if active_jobs == 0 || self.drain_deadline.is_some_and(|deadline| now >= deadline) {
            self.state = WorkerState::Paused;
            self.drain_deadline = None;
            return Some(ControlEffect {
                release_active: active_jobs > 0,
                exit: false,
                snapshot: self.snapshot(0),
            });
        }
        None
    }
}

/// Apply a strict address-space ceiling to this process only. Use exclusively
/// in a dedicated native worker before model loading. RLIMIT_AS is not a RAM
/// or GPU/VRAM meter: large mmap files count against this virtual-memory cap.
/// Never call this in the GUI process, and never raise an existing OS ceiling.
#[cfg(target_os = "linux")]
pub fn apply_address_space_limit_bytes(requested: u64) -> std::io::Result<u64> {
    if !(64 * 1024 * 1024..=1024 * 1024 * 1024 * 1024).contains(&requested) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "address-space limit must be 64 MiB through 1 TiB",
        ));
    }
    let mut current = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: current is a valid writable rlimit and RLIMIT_AS is a Linux resource.
    if unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut current) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let target = requested.min(current.rlim_cur).min(current.rlim_max);
    let next = libc::rlimit {
        rlim_cur: target,
        rlim_max: target,
    };
    // SAFETY: next is a valid initialized rlimit. This only lowers our process's limit.
    if unsafe { libc::setrlimit(libc::RLIMIT_AS, &next) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(target)
}
#[cfg(not(target_os = "linux"))]
pub fn apply_address_space_limit_bytes(_: u64) -> std::io::Result<u64> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "strict address-space limits are currently verified on Linux only",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pause_drain_and_stop_have_distinct_cleanup_semantics() {
        let now = Instant::now();
        let mut controller = WorkerController::default();
        let effect = controller.apply(WorkerAction::Drain, 1, now).unwrap();
        assert!(!effect.release_active && !effect.exit);
        assert!(!controller.allows_assignment());
        assert!(controller.tick(1, now + Duration::from_secs(9)).is_none());
        assert!(
            controller
                .tick(1, now + Duration::from_secs(10))
                .unwrap()
                .release_active
        );
        controller.apply(WorkerAction::Resume, 0, now).unwrap();
        assert!(controller.allows_assignment());
        assert!(
            controller
                .apply(WorkerAction::Pause, 1, now)
                .unwrap()
                .release_active
        );
        let effect = controller.apply(WorkerAction::Stop, 1, now).unwrap();
        assert!(effect.exit && effect.release_active);
        assert!(controller.apply(WorkerAction::Resume, 0, now).is_err());
    }
    #[tokio::test]
    async fn control_acknowledgement_requires_actual_worker_application() {
        let (handle, mut receiver) = worker_control_channel();
        let worker = tokio::spawn(async move {
            let request = receiver.recv().await.unwrap();
            let mut controller = WorkerController::default();
            let result = controller
                .apply(request.action, 0, Instant::now())
                .map(|effect| effect.snapshot);
            request.respond(result);
        });
        assert_eq!(
            handle.request(WorkerAction::Pause).await.unwrap().state,
            WorkerState::Paused
        );
        worker.await.unwrap();
        assert!(matches!(
            handle.request(WorkerAction::Resume).await,
            Err(ControlError::Disconnected)
        ));
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn address_space_limit_child() {
        if std::env::var_os("SYNAPSE_TEST_LIMIT_CHILD").is_none() {
            return;
        }
        let applied = apply_address_space_limit_bytes(256 * 1024 * 1024).unwrap();
        assert!(applied <= 256 * 1024 * 1024);
        let mut allocation = Vec::<u8>::new();
        assert!(allocation.try_reserve_exact(512 * 1024 * 1024).is_err());
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_kernel_enforces_the_dedicated_child_limit() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "network::worker::tests::address_space_limit_child",
                "--nocapture",
            ])
            .env("SYNAPSE_TEST_LIMIT_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child limit test failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
