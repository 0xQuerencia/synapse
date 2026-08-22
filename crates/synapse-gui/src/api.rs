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
use tower_http::cors::CorsLayer;

use crate::{
    config_io,
    session::{MasterStartResult, SessionStore},
};

/// Container of all handler-shared state.
#[derive(Clone)]
pub struct AppState {
    pub sessions: SessionStore,
    /// Optional shared access token. When `Some`, the `auth_guard`
    /// middleware requires it on every request. `None` = no auth
    /// (localhost-only mode).
    pub token: Option<Arc<String>>,
}

/// Request gate: when a token is configured, every request must present it
/// via `?token=`, `x-synapse-token`, `Authorization: Bearer`, or the
/// `synapse_token` cookie. A valid `?token=` also sets the cookie so the
/// browser (and the WebSocket upgrade, which can't send custom headers)
/// stays authenticated for the rest of the session.
pub async fn auth_guard(
    State(state): State<AppState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::header;

    let Some(expected) = state.token.as_deref() else {
        // No token configured — open (only safe on 127.0.0.1).
        return next.run(req).await;
    };

    let from_query = req.uri().query().and_then(|q| query_param(q, "token"));
    let headers = req.headers();
    let from_header = headers
        .get("x-synapse-token")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .or_else(|| {
            headers
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
                .map(str::to_owned)
        });
    let from_cookie = headers
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| cookie_value(c, "synapse_token"));

    let presented = from_query.clone().or(from_header).or(from_cookie);

    match presented {
        Some(tok) if ct_eq(tok.as_bytes(), expected.as_bytes()) => {
            let set_cookie = from_query.is_some();
            let mut resp = next.run(req).await;
            if set_cookie
                && let Ok(val) = header::HeaderValue::from_str(&format!(
                    "synapse_token={expected}; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400"
                ))
            {
                resp.headers_mut().append(header::SET_COOKIE, val);
            }
            resp
        }
        _ => (
            StatusCode::UNAUTHORIZED,
            "unauthorized — open this server with ?token=YOUR_TOKEN appended to the URL",
        )
            .into_response(),
    }
}

/// Constant-time byte comparison to avoid leaking the token via timing.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Pull a single value out of a `key=value&key2=value2` query string.
fn query_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        let mut it = pair.splitn(2, '=');
        if it.next() == Some(key) {
            return it.next().map(str::to_owned);
        }
    }
    None
}

/// Pull a single cookie value out of a `Cookie:` header string.
fn cookie_value(cookies: &str, name: &str) -> Option<String> {
    let prefix = format!("{name}=");
    for c in cookies.split(';') {
        if let Some(v) = c.trim().strip_prefix(&prefix) {
            return Some(v.to_owned());
        }
    }
    None
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
        .layer(CorsLayer::permissive())
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
}

async fn post_join(
    State(state): State<AppState>,
    Json(req): Json<JoinRequest>,
) -> Result<Json<JoinResponse>, ApiError> {
    let peer_id = state
        .sessions
        .start_worker(req.model, req.code, req.master_addr)
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
