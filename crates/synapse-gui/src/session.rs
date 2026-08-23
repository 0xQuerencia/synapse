//! Owns the running `SynapseNode` (if any) and the WebSocket fan-out.
//!
//! The GUI is a thin shell around the existing `synapse-core` API.
//! For a master session we call `chat_channels()` and keep the
//! `event_rx` half in the GUI; the node's run loop sends events
//! through the corresponding `event_tx` (now buried inside the
//! swarm runtime). We fan those out via a `tokio::broadcast` so
//! multiple browser tabs share the same master cleanly.
//!
//! Worker and relay sessions just spawn `node.run()` and report
//! "running" status; their event streams are not yet surfaced.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use synapse_core::{
    ModelIdentity, format_instruct_prompt, generate_pairing_code,
    network::{
        ChatEvent, NetworkError, NodeStatusSnapshot, SynapseNode, SynapseRole, default_listen_addr,
        parse_listen_addr,
    },
};
use tokio::sync::{Mutex, broadcast, mpsc, watch};

/// The set of `ChatEvent`s a UI cares about, serialised as JSON.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UiChatEvent {
    WorkerConnected,
    Token {
        text: String,
    },
    GenerationDone {
        tokens_generated: usize,
        reason: String,
    },
}

impl From<ChatEvent> for UiChatEvent {
    fn from(value: ChatEvent) -> Self {
        match value {
            ChatEvent::WorkerConnected => Self::WorkerConnected,
            ChatEvent::Token(t) => Self::Token { text: t },
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

/// State for the currently running node, if any.
pub struct SessionState {
    pub role: SessionRole,
    /// `Some` when role is `Master` — used to forward prompts.
    pub prompt_tx: Option<mpsc::Sender<String>>,
    /// `Some` when role is `Master` — used to subscribe to events.
    pub event_bcast: Option<broadcast::Sender<UiChatEvent>>,
    /// Handle to the background task running `node.run()`. Drop cancels it.
    pub run_handle: Option<tokio::task::JoinHandle<Result<(), NetworkError>>>,
    /// Pairing code for the master, if known.
    pub pairing_code: Option<String>,
    /// Listen address used to start the node, if known.
    pub listen_addr: Option<String>,
    /// Cached peer id string.
    pub peer_id: String,
    /// The model id used for this session.
    pub model_id: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionRole {
    Idle,
    Master,
    Worker,
    Relay,
}

impl SessionRole {
    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            SessionRole::Idle => "idle",
            SessionRole::Master => "master",
            SessionRole::Worker => "worker",
            SessionRole::Relay => "relay",
        }
    }
}

/// Container for the optional active session, shared across handlers.
#[derive(Clone, Default)]
pub struct SessionStore {
    inner: Arc<Mutex<Option<SessionState>>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn snapshot(&self) -> SessionSnapshot {
        let guard = self.inner.lock().await;
        match guard.as_ref() {
            None => SessionSnapshot {
                role: SessionRole::Idle,
                peer_id: None,
                model_id: None,
                pairing_code: None,
                max_tokens: None,
                listen_addr: None,
            },
            Some(state) => SessionSnapshot {
                role: state.role,
                peer_id: Some(state.peer_id.clone()),
                model_id: Some(state.model_id.clone()),
                pairing_code: state.pairing_code.clone(),
                max_tokens: Some(state.max_tokens),
                listen_addr: state.listen_addr.clone(),
            },
        }
    }

    #[allow(dead_code)]
    pub async fn is_busy(&self) -> bool {
        let guard = self.inner.lock().await;
        guard
            .as_ref()
            .map(|s| s.role != SessionRole::Idle)
            .unwrap_or(false)
    }

    /// Subscribe to the active master session's chat events.
    /// Returns None if no master session is running.
    pub async fn subscribe_events(&self) -> Option<broadcast::Receiver<UiChatEvent>> {
        let guard = self.inner.lock().await;
        guard
            .as_ref()
            .and_then(|s| s.event_bcast.as_ref().map(|tx| tx.subscribe()))
    }

    pub async fn start_master(
        &self,
        model_id: String,
        max_tokens: usize,
        listen: String,
    ) -> Result<MasterStartResult> {
        let mut guard = self.inner.lock().await;
        if guard
            .as_ref()
            .map(|s| s.role != SessionRole::Idle)
            .unwrap_or(false)
        {
            return Err(anyhow!("a session is already running; stop it first"));
        }

        let identity = ModelIdentity::new(&model_id);
        identity.validate().context("invalid model id")?;
        let listen_addr = parse_listen_addr(&listen).map_err(|e| anyhow!(e))?;
        let pairing_code = generate_pairing_code();

        let mut node = SynapseNode::new(
            SynapseRole::Master {
                expected_pairing_code: pairing_code.clone(),
                max_tokens,
            },
            identity,
            listen_addr,
        )
        .await
        .map_err(|e| anyhow!("failed to construct master node: {e}"))?;

        node.load_model()
            .await
            .map_err(|e| anyhow!("failed to load model: {e}"))?;

        let peer_id = node.local_peer_id().to_string();
        let (prompt_tx, mut event_rx) = node.chat_channels();
        let (bcast_tx, _) = broadcast::channel::<UiChatEvent>(1024);

        // Bridge: forward node's chat events into the broadcast.
        let bcast_for_bridge = bcast_tx.clone();
        tokio::spawn(async move {
            while let Some(ev) = event_rx.recv().await {
                let _ = bcast_for_bridge.send(ev.into());
            }
        });

        let node_handle = tokio::spawn(async move { node.run().await });

        // Build the snapshot from the state we just inserted BEFORE
        // dropping the mutex. Re-locking the mutex via `self.snapshot()`
        // would deadlock — tokio::Mutex is non-reentrant.
        let session = SessionSnapshot {
            role: SessionRole::Master,
            peer_id: Some(peer_id.clone()),
            model_id: Some(model_id.clone()),
            pairing_code: Some(pairing_code.clone()),
            max_tokens: Some(max_tokens),
            listen_addr: Some(listen.clone()),
        };

        *guard = Some(SessionState {
            role: SessionRole::Master,
            prompt_tx: Some(prompt_tx),
            event_bcast: Some(bcast_tx),
            run_handle: Some(node_handle),
            pairing_code: Some(pairing_code.clone()),
            listen_addr: Some(listen),
            peer_id,
            model_id,
            max_tokens,
        });

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
    ) -> Result<String> {
        let mut guard = self.inner.lock().await;
        if guard
            .as_ref()
            .map(|s| s.role != SessionRole::Idle)
            .unwrap_or(false)
        {
            return Err(anyhow!("a session is already running; stop it first"));
        }

        let identity = ModelIdentity::new(&model_id);
        identity.validate().context("invalid model id")?;
        synapse_core::validate_pairing_code(&code)
            .context("pairing code format validation failed")?;

        let manual_master = master_addr
            .as_deref()
            .map(parse_listen_addr)
            .transpose()
            .map_err(|e| anyhow!("invalid master address: {e}"))?;

        let mut node = SynapseNode::new(
            SynapseRole::Worker {
                pairing_code: code,
                manual_master,
            },
            identity,
            default_listen_addr(),
        )
        .await
        .map_err(|e| anyhow!("failed to construct worker node: {e}"))?;

        node.load_model()
            .await
            .map_err(|e| anyhow!("failed to load model: {e}"))?;

        let peer_id = node.local_peer_id().to_string();
        let node_handle = tokio::spawn(async move { node.run().await });

        *guard = Some(SessionState {
            role: SessionRole::Worker,
            prompt_tx: None,
            event_bcast: None,
            run_handle: Some(node_handle),
            pairing_code: None,
            listen_addr: None,
            peer_id: peer_id.clone(),
            model_id,
            max_tokens: 0,
        });

        Ok(peer_id)
    }

    pub async fn start_relay(&self, listen: String) -> Result<String> {
        let mut guard = self.inner.lock().await;
        if guard
            .as_ref()
            .map(|s| s.role != SessionRole::Idle)
            .unwrap_or(false)
        {
            return Err(anyhow!("a session is already running; stop it first"));
        }

        let listen_addr = parse_listen_addr(&listen).map_err(|e| anyhow!(e))?;
        let identity = ModelIdentity::new("synapse-relay");
        let node = SynapseNode::new(SynapseRole::RelayServer, identity, listen_addr)
            .await
            .map_err(|e| anyhow!("failed to construct relay node: {e}"))?;
        let peer_id = node.local_peer_id().to_string();
        let node_handle = tokio::spawn(async move { node.run().await });

        *guard = Some(SessionState {
            role: SessionRole::Relay,
            prompt_tx: None,
            event_bcast: None,
            run_handle: Some(node_handle),
            pairing_code: None,
            listen_addr: Some(listen),
            peer_id: peer_id.clone(),
            model_id: "synapse-relay".to_owned(),
            max_tokens: 0,
        });

        Ok(peer_id)
    }

    pub async fn stop(&self) -> Result<()> {
        let mut guard = self.inner.lock().await;
        let Some(state) = guard.as_mut() else {
            return Ok(());
        };
        if let Some(handle) = state.run_handle.take() {
            handle.abort();
        }
        // Drop prompt_tx so the node's prompt channel closes if it cares.
        drop(state.prompt_tx.take());
        *guard = None;
        Ok(())
    }

    pub async fn send_prompt(&self, text: String, max_tokens: Option<usize>) -> Result<()> {
        let guard = self.inner.lock().await;
        let Some(state) = guard.as_ref() else {
            return Err(anyhow!("no session is running"));
        };
        if state.role != SessionRole::Master {
            return Err(anyhow!("prompts can only be sent to a master session"));
        }
        let Some(prompt_tx) = state.prompt_tx.as_ref() else {
            return Err(anyhow!("master session is not accepting prompts yet"));
        };

        let _ = max_tokens; // The master's max_tokens is locked at start; the
        // chat_channels prompt stream doesn't accept a
        // per-message override. We accept the API anyway
        // for forward compatibility.
        let formatted = format_instruct_prompt(&text);
        prompt_tx
            .send(formatted)
            .await
            .map_err(|e| anyhow!("failed to send prompt: {e}"))?;
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSnapshot {
    pub role: SessionRole,
    pub peer_id: Option<String>,
    pub model_id: Option<String>,
    pub pairing_code: Option<String>,
    pub max_tokens: Option<usize>,
    pub listen_addr: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MasterStartResult {
    pub pairing_code: String,
    /// Full session snapshot — lets the React side update its
    /// `status` cache immediately so `isMaster` flips to true without
    /// waiting for the next 2s status poll. Without this, prompts
    /// typed right after Start are silently dropped because the
    /// WebSocket hasn't been opened yet.
    pub session: SessionSnapshot,
}
