//! One bounded local session, real runtime status, and acknowledged controls.
use anyhow::{Context, Result, anyhow, ensure};
use std::{
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use synapse_core::{
    ModelIdentity, generate_pairing_code,
    network::{
        ChatEvent, NodeStatusSnapshot, SynapseNode, SynapseRole, parse_listen_addr,
        worker::{WorkerAction, WorkerControlHandle, WorkerSnapshot},
    },
};
use tokio::sync::{Mutex, broadcast, mpsc, watch};

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UiChatEvent {
    WorkerConnected,
    TokenGenerated {
        token_id: u32,
        position: usize,
    },
    Token {
        text: String,
    },
    GenerationDone {
        tokens_generated: usize,
        reason: String,
    },
}
impl From<ChatEvent> for UiChatEvent {
    fn from(event: ChatEvent) -> Self {
        match event {
            ChatEvent::WorkerConnected => Self::WorkerConnected,
            ChatEvent::TokenGenerated { token_id, position } => {
                Self::TokenGenerated { token_id, position }
            }
            ChatEvent::Token(text) => Self::Token { text },
            ChatEvent::GenerationDone {
                tokens_generated,
                reason,
            } => Self::GenerationDone {
                tokens_generated,
                reason: format!("{reason:?}"),
            },
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionRole {
    Idle,
    Master,
    Worker,
    Relay,
}

#[derive(Clone, Debug)]
pub struct HostOptions {
    pub max_tokens: usize,
    pub listen: String,
    pub memory_mib: Option<u64>,
    pub raw_prompt: bool,
    pub coordinator_only: bool,
}
#[derive(Clone, Debug)]
pub struct WorkerOptions {
    pub listen: String,
    pub memory_mib: Option<u64>,
    /// Drain automatically after this many minutes; an active job has 10s to end.
    pub duration_minutes: Option<u64>,
}

type LastError = Arc<StdMutex<Option<String>>>;
struct SessionState {
    role: SessionRole,
    prompt_tx: Option<mpsc::Sender<String>>,
    event_bcast: Option<broadcast::Sender<UiChatEvent>>,
    run_handle: tokio::task::JoinHandle<()>,
    bridge_handle: Option<tokio::task::JoinHandle<()>>,
    schedule_handle: Option<tokio::task::JoinHandle<()>>,
    worker_control: Option<WorkerControlHandle>,
    pairing_code: Option<String>,
    listen_addr: String,
    peer_id: String,
    model_id: String,
    max_tokens: usize,
    is_ollama: bool,
    raw_prompt: bool,
    generating: Arc<AtomicBool>,
    last_error: LastError,
    drain_at_unix: Option<u64>,
    status_rx: Option<watch::Receiver<NodeStatusSnapshot>>,
}
impl SessionState {
    fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            role: self.role,
            running: !self.run_handle.is_finished(),
            peer_id: Some(self.peer_id.clone()),
            model_id: Some(self.model_id.clone()),
            pairing_code: self.pairing_code.clone(),
            max_tokens: Some(self.max_tokens),
            listen_addr: Some(self.listen_addr.clone()),
            generating: self.generating.load(Ordering::Acquire),
            last_error: self
                .last_error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            drain_at_unix: self.drain_at_unix,
            backend: if self.is_ollama {
                "ollama_local"
            } else {
                "candle"
            }
            .into(),
        }
    }
}
#[derive(Clone, Default)]
pub struct SessionStore {
    inner: Arc<Mutex<Option<SessionState>>>,
}

fn memory_budget(mib: Option<u64>) -> Result<Option<u64>> {
    mib.map(|mib| {
        mib.checked_mul(1024 * 1024)
            .filter(|v| *v > 0)
            .context("inference memory budget must be positive and must not overflow")
    })
    .transpose()
}
fn canonical_model(id: &str) -> Result<String> {
    ensure!(
        !id.is_empty() && id.len() <= 1024,
        "model identifier must be 1..1024 bytes"
    );
    if std::path::Path::new(id).is_dir() {
        Ok(
            synapse_core::inference::model_loader::ModelFiles::from_directory(
                std::path::Path::new(id),
            )?
            .manifest
            .model_id,
        )
    } else {
        Ok(id.to_owned())
    }
}
fn spawn_node(
    node: SynapseNode,
    error: LastError,
    generating: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(failure) = node.run().await {
            tracing::warn!(error = %failure, "session stopped with an error");
            *error.lock().unwrap_or_else(|e| e.into_inner()) = Some(failure.to_string());
        }
        generating.store(false, Ordering::Release);
    })
}
fn accept_event(event: ChatEvent, busy: &AtomicBool, sender: &broadcast::Sender<UiChatEvent>) {
    if matches!(event, ChatEvent::GenerationDone { .. }) {
        busy.store(false, Ordering::Release);
    }
    let _ = sender.send(event.into());
}
impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }
    pub async fn snapshot(&self) -> SessionSnapshot {
        self.inner
            .lock()
            .await
            .as_ref()
            .map(SessionState::snapshot)
            .unwrap_or_default()
    }
    pub async fn subscribe_events(&self) -> Option<broadcast::Receiver<UiChatEvent>> {
        self.inner
            .lock()
            .await
            .as_ref()
            .and_then(|s| s.event_bcast.as_ref().map(|tx| tx.subscribe()))
    }
    pub async fn node_status(&self) -> Option<NodeStatusSnapshot> {
        self.inner
            .lock()
            .await
            .as_ref()
            .and_then(|s| s.status_rx.as_ref())
            .map(|rx| rx.borrow().clone())
    }
    pub async fn start_master(
        &self,
        model_id: String,
        options: HostOptions,
    ) -> Result<MasterStartResult> {
        ensure!(
            (1..=4096).contains(&options.max_tokens),
            "output limit must be 1..4096 tokens"
        );
        let budget = memory_budget(options.memory_mib)?;
        let canonical = canonical_model(&model_id)?;
        let mut guard = self.inner.lock().await;
        ensure!(
            guard.is_none(),
            "a session already exists; stop it before starting another"
        );
        let generating = Arc::new(AtomicBool::new(false));
        let last_error: LastError = Arc::new(StdMutex::new(None));
        let (bcast_tx, _) = broadcast::channel::<UiChatEvent>(256);
        let (prompt_tx, run_handle, bridge_handle, peer_id, pairing_code, status_rx, is_ollama) =
            if let Some(ollama_model) = model_id.strip_prefix("ollama:") {
                ensure!(
                    budget.is_none() && !options.coordinator_only,
                    "local Ollama does not enforce Synapse shard memory options"
                );
                let client = crate::ollama::OllamaClient::new(
                    std::env::var("SYNAPSE_OLLAMA_URL")
                        .unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
                )?;
                let available = tokio::time::timeout(Duration::from_secs(10), client.list_models())
                    .await
                    .context("Ollama model discovery timed out")??;
                ensure!(
                    available.iter().any(|id| id == ollama_model),
                    "requested local Ollama model is not installed"
                );
                let (chat_tx, mut chat_rx) = mpsc::channel::<ChatEvent>(32);
                let bridge_sender = bcast_tx.clone();
                let bridge_busy = generating.clone();
                let bridge = tokio::spawn(async move {
                    while let Some(event) = chat_rx.recv().await {
                        accept_event(event, &bridge_busy, &bridge_sender);
                    }
                    bridge_busy.store(false, Ordering::Release);
                });
                let (prompt_tx, mut prompt_rx) = mpsc::channel::<String>(1);
                let name = ollama_model.to_owned();
                let busy = generating.clone();
                let errors = last_error.clone();
                let run = tokio::spawn(async move {
                    while let Some(prompt) = prompt_rx.recv().await {
                        let (cancel_tx, cancel_rx) = mpsc::channel::<()>(1);
                        let result = tokio::time::timeout(
                            Duration::from_secs(120),
                            client.stream_generate(
                                &name,
                                &prompt,
                                options.max_tokens as u32,
                                &chat_tx,
                                cancel_rx,
                            ),
                        )
                        .await;
                        drop(cancel_tx);
                        if !matches!(result, Ok(Ok(()))) {
                            *errors.lock().unwrap_or_else(|e| e.into_inner()) =
                                Some("local Ollama generation failed or timed out".into());
                            let _ = chat_tx
                                .send(ChatEvent::GenerationDone {
                                    tokens_generated: 0,
                                    reason: synapse_core::network::StopReason::Error,
                                })
                                .await;
                        }
                        busy.store(false, Ordering::Release);
                    }
                });
                (
                    prompt_tx,
                    run,
                    bridge,
                    format!("ollama:{ollama_model}"),
                    None,
                    None,
                    true,
                )
            } else {
                // Reject unsupported templates before starting, rather than silently wrapping every model in ChatML.
                synapse_core::inference::prompt_format::format_prompt_for_model(
                    &canonical,
                    "readiness",
                    options.raw_prompt,
                )?;
                let pairing = generate_pairing_code();
                let mut node = SynapseNode::new(
                    SynapseRole::Master {
                        expected_pairing_code: pairing.clone(),
                        max_tokens: options.max_tokens,
                    },
                    ModelIdentity::new(&model_id),
                    parse_listen_addr(&options.listen).map_err(|e| anyhow!(e))?,
                )
                .await?;
                if let Some(bytes) = budget {
                    node.set_memory_limit_bytes(bytes)?;
                }
                node.set_coordinator_only(options.coordinator_only)?;
                node.load_model().await?;
                let peer_id = node.local_peer_id().to_string();
                let status_rx = node.status_receiver();
                let (prompt_tx, mut event_rx) = node.chat_channels();
                let bridge_sender = bcast_tx.clone();
                let bridge_busy = generating.clone();
                let bridge = tokio::spawn(async move {
                    while let Some(event) = event_rx.recv().await {
                        accept_event(event, &bridge_busy, &bridge_sender);
                    }
                    bridge_busy.store(false, Ordering::Release);
                });
                (
                    prompt_tx,
                    spawn_node(node, last_error.clone(), generating.clone()),
                    bridge,
                    peer_id,
                    Some(pairing),
                    Some(status_rx),
                    false,
                )
            };
        let state = SessionState {
            role: SessionRole::Master,
            prompt_tx: Some(prompt_tx),
            event_bcast: Some(bcast_tx),
            run_handle,
            bridge_handle: Some(bridge_handle),
            schedule_handle: None,
            worker_control: None,
            pairing_code: pairing_code.clone(),
            listen_addr: options.listen,
            peer_id,
            model_id: canonical,
            max_tokens: options.max_tokens,
            is_ollama,
            raw_prompt: options.raw_prompt,
            generating,
            last_error,
            drain_at_unix: None,
            status_rx,
        };
        let session = state.snapshot();
        *guard = Some(state);
        Ok(MasterStartResult {
            pairing_code,
            session,
        })
    }

    pub async fn start_worker(
        &self,
        model_id: String,
        code: String,
        master_addr: Option<String>,
        options: WorkerOptions,
    ) -> Result<String> {
        let budget = memory_budget(options.memory_mib)?;
        ensure!(
            options
                .duration_minutes
                .is_none_or(|v| (1..=10080).contains(&v)),
            "duration must be 1 minute through 7 days"
        );
        let canonical = canonical_model(&model_id)?;
        synapse_core::validate_pairing_code(&code)?;
        let mut guard = self.inner.lock().await;
        ensure!(guard.is_none(), "a session already exists; stop it first");
        let manual_master = master_addr
            .as_deref()
            .map(parse_listen_addr)
            .transpose()
            .map_err(|e| anyhow!(e))?;
        let mut node = SynapseNode::new(
            SynapseRole::Worker {
                pairing_code: code,
                manual_master,
            },
            ModelIdentity::new(&model_id),
            parse_listen_addr(&options.listen).map_err(|e| anyhow!(e))?,
        )
        .await?;
        if let Some(bytes) = budget {
            node.set_memory_limit_bytes(bytes)?;
        }
        node.load_model().await?;
        let peer_id = node.local_peer_id().to_string();
        let status_rx = node.status_receiver();
        let control = node.worker_control()?;
        let generating = Arc::new(AtomicBool::new(false));
        let last_error: LastError = Arc::new(StdMutex::new(None));
        let run_handle = spawn_node(node, last_error.clone(), generating.clone());
        let drain_at_unix = options.duration_minutes.map(|minutes| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                + minutes * 60
        });
        let schedule_handle = options.duration_minutes.map(|minutes| {
            let control = control.clone();
            let error = last_error.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(minutes * 60)).await;
                if let Err(failure) = control.request(WorkerAction::Drain).await {
                    *error.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(format!("scheduled drain failed: {failure}"));
                }
            })
        });
        *guard = Some(SessionState {
            role: SessionRole::Worker,
            prompt_tx: None,
            event_bcast: None,
            run_handle,
            bridge_handle: None,
            schedule_handle,
            worker_control: Some(control),
            pairing_code: None,
            listen_addr: options.listen,
            peer_id: peer_id.clone(),
            model_id: canonical,
            max_tokens: 0,
            is_ollama: false,
            raw_prompt: true,
            generating,
            last_error,
            drain_at_unix,
            status_rx: Some(status_rx),
        });
        Ok(peer_id)
    }

    pub async fn start_relay(&self, listen: String) -> Result<String> {
        let mut guard = self.inner.lock().await;
        ensure!(guard.is_none(), "a session already exists; stop it first");
        let node = SynapseNode::new(
            SynapseRole::RelayServer,
            ModelIdentity::new("synapse-relay"),
            parse_listen_addr(&listen).map_err(|e| anyhow!(e))?,
        )
        .await?;
        let peer_id = node.local_peer_id().to_string();
        let status_rx = node.status_receiver();
        let generating = Arc::new(AtomicBool::new(false));
        let last_error = Arc::new(StdMutex::new(None));
        let run_handle = spawn_node(node, last_error.clone(), generating.clone());
        *guard = Some(SessionState {
            role: SessionRole::Relay,
            prompt_tx: None,
            event_bcast: None,
            run_handle,
            bridge_handle: None,
            schedule_handle: None,
            worker_control: None,
            pairing_code: None,
            listen_addr: listen,
            peer_id: peer_id.clone(),
            model_id: "synapse-relay".into(),
            max_tokens: 0,
            is_ollama: false,
            raw_prompt: true,
            generating,
            last_error,
            drain_at_unix: None,
            status_rx: Some(status_rx),
        });
        Ok(peer_id)
    }

    pub async fn worker_control(&self, action: WorkerAction) -> Result<WorkerSnapshot> {
        let control = self
            .inner
            .lock()
            .await
            .as_ref()
            .and_then(|s| s.worker_control.clone())
            .context("no controllable worker is running")?;
        Ok(control.request(action).await?)
    }
    pub async fn stop(&self) -> Result<()> {
        let mut guard = self.inner.lock().await;
        let Some(mut state) = guard.take() else {
            return Ok(());
        };
        if let Some(handle) = state.schedule_handle.take() {
            handle.abort();
            let _ = handle.await;
        }
        let control_error = if let Some(control) = state.worker_control.take()
            && !state.run_handle.is_finished()
        {
            control.request(WorkerAction::Stop).await.err()
        } else {
            None
        };
        drop(state.prompt_tx.take());
        if tokio::time::timeout(Duration::from_secs(3), &mut state.run_handle)
            .await
            .is_err()
        {
            state.run_handle.abort();
            let _ = state.run_handle.await;
        }
        if let Some(handle) = state.bridge_handle.take() {
            handle.abort();
            let _ = handle.await;
        }
        state.generating.store(false, Ordering::Release);
        if let Some(error) = control_error {
            return Err(anyhow!(
                "worker did not acknowledge stop; its local task was terminated: {error}"
            ));
        }
        Ok(())
    }
    pub async fn send_prompt(&self, text: String, max_tokens: Option<usize>) -> Result<()> {
        ensure!(
            !text.trim().is_empty() && text.len() <= 256 * 1024,
            "prompt must be nonempty and at most 256 KiB"
        );
        let guard = self.inner.lock().await;
        let state = guard.as_ref().context("no session is running")?;
        ensure!(
            state.role == SessionRole::Master && !state.run_handle.is_finished(),
            "no running host session"
        );
        ensure!(
            max_tokens.is_none_or(|limit| limit == state.max_tokens),
            "output limit is fixed at session start; restart to change it"
        );
        let payload = if state.is_ollama {
            text
        } else {
            synapse_core::inference::prompt_format::format_prompt_for_model(
                &state.model_id,
                &text,
                state.raw_prompt,
            )?
        };
        let prompt_tx = state
            .prompt_tx
            .as_ref()
            .context("host is not accepting prompts")?;
        ensure!(
            state
                .generating
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "host is busy; wait for the active generation"
        );
        if prompt_tx.try_send(payload).is_err() {
            state.generating.store(false, Ordering::Release);
            return Err(anyhow!("host prompt queue is unavailable"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSnapshot {
    pub role: SessionRole,
    pub running: bool,
    pub peer_id: Option<String>,
    pub model_id: Option<String>,
    pub pairing_code: Option<String>,
    pub max_tokens: Option<usize>,
    pub listen_addr: Option<String>,
    pub generating: bool,
    pub last_error: Option<String>,
    pub drain_at_unix: Option<u64>,
    pub backend: String,
}
impl Default for SessionSnapshot {
    fn default() -> Self {
        Self {
            role: SessionRole::Idle,
            running: false,
            peer_id: None,
            model_id: None,
            pairing_code: None,
            max_tokens: None,
            listen_addr: None,
            generating: false,
            last_error: None,
            drain_at_unix: None,
            backend: "none".into(),
        }
    }
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct MasterStartResult {
    pub pairing_code: Option<String>,
    pub session: SessionSnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn missing_sessions_and_unacknowledged_controls_fail_explicitly() {
        let store = SessionStore::new();
        assert!(!store.snapshot().await.running);
        assert!(store.worker_control(WorkerAction::Pause).await.is_err());
        assert!(store.send_prompt("Hello".into(), None).await.is_err());
        assert!(store.send_prompt(" ".into(), None).await.is_err());
        store.stop().await.unwrap();
    }
    #[test]
    fn memory_input_is_checked_and_completion_releases_admission() {
        assert!(memory_budget(Some(0)).is_err());
        assert!(memory_budget(Some(u64::MAX)).is_err());
        let (tx, mut rx) = broadcast::channel(2);
        let busy = AtomicBool::new(true);
        accept_event(
            ChatEvent::GenerationDone {
                tokens_generated: 2,
                reason: synapse_core::network::StopReason::MaxTokens,
            },
            &busy,
            &tx,
        );
        assert!(!busy.load(Ordering::Acquire));
        assert!(matches!(
            rx.try_recv().unwrap(),
            UiChatEvent::GenerationDone {
                tokens_generated: 2,
                ..
            }
        ));
    }
}
