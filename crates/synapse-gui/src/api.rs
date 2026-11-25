//! REST endpoints exposed to the browser.
//!
//! Each handler returns either a JSON body or an `ApiError` mapped to
//! an HTTP status. Server-side errors are logged via `tracing`; only
//! short, non-sensitive messages are returned to the client.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use serde_json::json;
use synapse_core::{NodeIdentity, list_cached_hf_models};

use crate::{
    config_io,
    session::{MasterStartResult, SessionStore},
};

/// Container of all handler-shared state.
#[derive(Clone)]
pub struct AppState {
    pub sessions: SessionStore,
    pub auth: Arc<crate::auth::LocalAuth>,
    pub inference: Option<crate::compatible_api::InferenceService>,
}

/// Build the REST sub-router. Returns a `Router<AppState>` that
/// the caller can mount alongside more routes (WS, static assets)
/// before calling `.with_state(state)` once at the top.
pub fn router() -> axum::Router<AppState> {
    use axum::routing::{get, post};

    axum::Router::new()
        .route("/api/identity", get(get_identity))
        .route("/api/status", get(get_status))
        .route("/api/peers", get(get_peers))
        .route("/api/cached-models", get(get_cached_models))
        .route("/api/benchmark", post(post_benchmark))
        .route("/api/download-model", post(post_download_model))
        .route("/api/worker-info", get(get_worker_info))
        .route("/api/config/:file", get(get_config).put(put_config))
        .route("/api/session/host", post(post_host))
        .route("/api/session/join", post(post_join))
        .route("/api/session/relay", post(post_relay))
        .route("/api/session/stop", post(post_stop))
        .route("/api/session/prompt", post(post_prompt))
}

// ─── Identity / status ──────────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct IdentityResponse {
    peer_id: String,
    short_id: String,
    public_key: String,
    key_path: String,
}

async fn get_identity() -> Result<Json<IdentityResponse>, ApiError> {
    let identity = NodeIdentity::load_or_create()
        .map_err(|e| ApiError::internal("identity", e.to_string()))?;
    let pub_bytes = identity.public_key_bytes();
    let pub_hex: String = pub_bytes.iter().map(|b| format!("{b:02x}")).collect();

    Ok(Json(IdentityResponse {
        peer_id: identity.peer_id().to_string(),
        short_id: identity.display_short(),
        public_key: truncate(&pub_hex, 16),
        key_path: identity.key_path().display().to_string(),
    }))
}

async fn get_status(State(state): State<AppState>) -> Result<Json<StatusResponse>, ApiError> {
    let identity = NodeIdentity::load_or_create()
        .map_err(|e| ApiError::internal("identity", e.to_string()))?;
    let cached =
        list_cached_hf_models().map_err(|e| ApiError::internal("cached_models", e.to_string()))?;
    let net = synapse_core::NetConfig::load_default();
    let session = state.sessions.snapshot().await;

    Ok(Json(StatusResponse {
        peer_id: identity.peer_id().to_string(),
        cached_models: cached,
        bootstrap_count: net.bootstrap.len(),
        relay_count: net.relays.len(),
        session,
    }))
}

#[derive(serde::Serialize)]
struct StatusResponse {
    peer_id: String,
    cached_models: Vec<String>,
    bootstrap_count: usize,
    relay_count: usize,
    session: crate::session::SessionSnapshot,
}

async fn get_cached_models() -> Result<Json<Vec<String>>, ApiError> {
    list_cached_hf_models()
        .map(Json)
        .map_err(|e| ApiError::internal("cached_models", e.to_string()))
}

// ─── Contribute: compute test + model download ───────────────────────────────

#[derive(serde::Serialize)]
struct BenchmarkResponse {
    /// Relative compute score the planner divides layers by. Higher = faster.
    score: u64,
    /// Which device the benchmark ran on ("cpu" / "cuda" / "metal").
    device: String,
    /// Logical CPU thread count, for display.
    cpu_threads: u16,
}

/// Run a real on-device micro-benchmark and return the measured score.
///
/// This is the "compute test" — it times repeated matmuls on the best
/// available device, exactly the number the master uses to divide layers
/// proportionally. Runs on a blocking thread since it's CPU/GPU-bound.
async fn post_benchmark() -> Result<Json<BenchmarkResponse>, ApiError> {
    let result = tokio::task::spawn_blocking(|| {
        let device =
            synapse_core::get_best_device().map_err(|e| format!("device selection failed: {e}"))?;
        let score = synapse_core::run_micro_benchmark(&device);
        let device_label = if device.is_cpu() {
            "cpu"
        } else if device.is_cuda() {
            "cuda"
        } else {
            "metal"
        };
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(u16::MAX as usize) as u16)
            .unwrap_or(1);
        Ok::<_, String>((score, device_label.to_owned(), cpu_threads))
    })
    .await
    .map_err(|e| ApiError::internal("benchmark_join", e.to_string()))?
    .map_err(|e| ApiError::internal("benchmark", e))?;

    Ok(Json(BenchmarkResponse {
        score: result.0,
        device: result.1,
        cpu_threads: result.2,
    }))
}

#[derive(Deserialize)]
struct DownloadRequest {
    model: String,
}

#[derive(serde::Serialize)]
struct DownloadResponse {
    model: String,
    cached: bool,
}

/// Download (or confirm the cache of) a model's weights for this machine.
///
/// This is the worker-side "Download Model" button. It fetches the
/// safetensors + tokenizer into the local HF cache so the worker can load
/// the layers it gets assigned. Idempotent — re-running just confirms cache.
async fn post_download_model(
    Json(req): Json<DownloadRequest>,
) -> Result<Json<DownloadResponse>, ApiError> {
    if req.model.starts_with("ollama:") {
        return Err(ApiError::bad_request(
            "ollama models are managed by ollama, not downloadable here",
        ));
    }
    synapse_core::download_model(&req.model)
        .await
        .map_err(|e| ApiError::internal("download_model", e.to_string()))?;
    Ok(Json(DownloadResponse {
        model: req.model,
        cached: true,
    }))
}

/// Everything a friend's worker needs to join this master, in one call.
///
/// The onboarding page polls this so the contributor never has to copy a
/// pairing code by hand. Crucially it reports `supports_workers`: an ollama
/// (Gemma) session has NO swarm, so workers can't join it — only a candle
/// master (e.g. SmolLM2) accepts workers.
async fn get_worker_info(State(state): State<AppState>) -> Json<WorkerInfoResponse> {
    let snap = state.sessions.snapshot().await;
    let model_id = snap.model_id.clone().unwrap_or_default();
    let is_master = matches!(snap.role, crate::session::SessionRole::Master);
    let is_ollama = model_id.starts_with("ollama:");
    let supports_workers = is_master && !is_ollama;

    let reason = if !is_master {
        "No master session is running. Start a model on the host first."
    } else if is_ollama {
        "The host is running an ollama model, which has no swarm. Switch the \
         host to a candle model (e.g. SmolLM2) so workers can contribute."
    } else {
        "Ready for workers."
    };

    Json(WorkerInfoResponse {
        supports_workers,
        model_id,
        pairing_code: snap.pairing_code.clone(),
        // Same-LAN workers auto-discover the master via mDNS, so they don't
        // need an address. Cross-network needs a relay (see /contribute).
        discovery: "mdns".to_owned(),
        host_os: std::env::consts::OS.to_owned(),
        reason: reason.to_owned(),
    })
}

#[derive(serde::Serialize)]
struct WorkerInfoResponse {
    /// True only when a candle master is running (ollama has no swarm).
    supports_workers: bool,
    model_id: String,
    pairing_code: Option<String>,
    /// "mdns" — same-LAN auto-discovery (no address needed).
    discovery: String,
    host_os: String,
    reason: String,
}

/// Live peer / swarm snapshot for the running node.
///
/// Returns the real networking state published by the node's run loop:
/// connected + authenticated workers, the active generation route, and (for
/// worker sessions) reconnection status. When no swarm exists — no session,
/// the ollama backend, or a relay — `online` is 0 and `peers` is empty, which
/// the UI renders as "local / no peers" rather than fabricated numbers.
async fn get_peers(State(state): State<AppState>) -> Json<PeersResponse> {
    match state.sessions.node_status().await {
        Some(snapshot) => {
            let online = snapshot
                .peers
                .iter()
                .filter(|p| p.state == "authenticated" || p.state == "connected")
                .count();
            Json(PeersResponse {
                available: true,
                online,
                authenticated: snapshot.authenticated_count,
                role: snapshot.role,
                peers: snapshot.peers,
                route_plan: snapshot.route_plan,
                redial: snapshot.redial,
            })
        }
        // No swarm: ollama backend, relay, or idle. Report honestly.
        None => Json(PeersResponse {
            available: false,
            online: 0,
            authenticated: 0,
            role: "local".to_owned(),
            peers: Vec::new(),
            route_plan: None,
            redial: None,
        }),
    }
}

#[derive(serde::Serialize)]
struct PeersResponse {
    /// Whether a swarm exists at all (false for ollama / relay / idle).
    available: bool,
    /// Workers currently connected or authenticated.
    online: usize,
    /// Workers that completed challenge-response authentication.
    authenticated: usize,
    role: String,
    peers: Vec<synapse_core::network::PeerSummary>,
    route_plan: Option<synapse_core::network::RoutePlanView>,
    redial: Option<synapse_core::network::RedialStatus>,
}

// ─── Config editor ──────────────────────────────────────────────────────────

async fn get_config(Path(file): Path<String>) -> Result<Json<ConfigResponse>, ApiError> {
    let Some(file) = config_io::ConfigFile::from_label(&file) else {
        return Err(ApiError::bad_request("unknown config file"));
    };
    let contents =
        config_io::read(file).map_err(|e| ApiError::internal("config_read", e.to_string()))?;
    Ok(Json(ConfigResponse {
        file: file.label().to_owned(),
        contents,
    }))
}

async fn put_config(
    Path(file): Path<String>,
    Json(body): Json<ConfigWriteRequest>,
) -> Result<Json<ConfigResponse>, ApiError> {
    let Some(file) = config_io::ConfigFile::from_label(&file) else {
        return Err(ApiError::bad_request("unknown config file"));
    };
    config_io::write(file, &body.contents).map_err(|e| ApiError::bad_request(&e.to_string()))?;
    Ok(Json(ConfigResponse {
        file: file.label().to_owned(),
        contents: body.contents,
    }))
}

#[derive(serde::Serialize)]
struct ConfigResponse {
    file: String,
    contents: String,
}

#[derive(Deserialize)]
struct ConfigWriteRequest {
    contents: String,
}

// ─── Session lifecycle ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct HostRequest {
    model: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_listen")]
    listen: String,
}

fn default_max_tokens() -> usize {
    200
}
fn default_listen() -> String {
    "0.0.0.0:0".to_owned()
}

async fn post_host(
    State(state): State<AppState>,
    Json(req): Json<HostRequest>,
) -> Result<Json<MasterStartResult>, ApiError> {
    state
        .sessions
        .start_master(req.model, req.max_tokens, req.listen)
        .await
        .map(Json)
        .map_err(|e| ApiError::bad_request(&e.to_string()))
}

#[derive(Deserialize)]
struct JoinRequest {
    model: String,
    code: String,
    master_addr: Option<String>,
    /// Optional measured benchmark score from `/api/benchmark`. When
    /// present, the worker advertises it so the master divides layers by
    /// this machine's real throughput instead of a thread-count guess.
    #[serde(default)]
    compute_score: Option<u64>,
}

async fn post_join(
    State(state): State<AppState>,
    Json(req): Json<JoinRequest>,
) -> Result<Json<JoinResponse>, ApiError> {
    let peer_id = state
        .sessions
        .start_worker(req.model, req.code, req.master_addr, req.compute_score)
        .await
        .map_err(|e| ApiError::bad_request(&e.to_string()))?;
    let session = state.sessions.snapshot().await;
    Ok(Json(JoinResponse { peer_id, session }))
}

#[derive(serde::Serialize)]
struct JoinResponse {
    peer_id: String,
    session: crate::session::SessionSnapshot,
}

#[derive(Deserialize)]
struct RelayRequest {
    #[serde(default = "default_relay_listen")]
    listen: String,
}

fn default_relay_listen() -> String {
    "0.0.0.0:4001".to_owned()
}

async fn post_relay(
    State(state): State<AppState>,
    Json(req): Json<RelayRequest>,
) -> Result<Json<RelayResponse>, ApiError> {
    let peer_id = state
        .sessions
        .start_relay(req.listen)
        .await
        .map_err(|e| ApiError::bad_request(&e.to_string()))?;
    let session = state.sessions.snapshot().await;
    Ok(Json(RelayResponse { peer_id, session }))
}

#[derive(serde::Serialize)]
struct RelayResponse {
    peer_id: String,
    session: crate::session::SessionSnapshot,
}

async fn post_stop(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    state
        .sessions
        .stop()
        .await
        .map_err(|e| ApiError::internal("stop", e.to_string()))?;
    Ok(Json(json!({ "ok": true })))
}

#[derive(Deserialize)]
struct PromptRequest {
    text: String,
    max_tokens: Option<usize>,
}

async fn post_prompt(
    State(state): State<AppState>,
    Json(req): Json<PromptRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state
        .sessions
        .send_prompt(req.text, req.max_tokens)
        .await
        .map_err(|e| ApiError::bad_request(&e.to_string()))?;
    Ok(Json(json!({ "ok": true })))
}

// ─── Errors ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    pub fn bad_request(message: &str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.to_owned(),
        }
    }

    pub fn internal(scope: &'static str, detail: String) -> Self {
        tracing::error!(scope, error = %detail, "api internal error");
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "internal error".to_owned(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(json!({ "error": self.message }));
        (self.status, body).into_response()
    }
}

fn truncate(s: &str, prefix_bytes: usize) -> String {
    let chars = prefix_bytes * 2;
    if s.len() <= chars + 6 {
        s.to_owned()
    } else {
        format!("{}...{}", &s[..chars], &s[s.len() - 6..])
    }
}

// `Arc` is used by the `Clone` impl on AppState below.
const _: fn() = || {
    let _: Arc<()> = Arc::new(());
};
