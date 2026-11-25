//! Local authentication with a private token file and same-origin browser access.
use std::{collections::BTreeSet, net::SocketAddr, path::Path, sync::Arc};

use anyhow::{Context, Result, bail};
use axum::{
    Json,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{Html, IntoResponse, Redirect, Response},
};
use rand::RngCore;
use serde::Deserialize;

use crate::api::AppState;

#[derive(Clone)]
pub struct LocalAuth {
    token: Arc<String>,
    hosts: BTreeSet<String>,
}

impl LocalAuth {
    pub fn new(token: String, address: SocketAddr) -> Result<Self> {
        if !address.ip().is_loopback() {
            bail!(
                "the admin/API server must bind a loopback address; use an authenticated encrypted tunnel for private remote access"
            );
        }
        if !valid_token_format(&token) {
            bail!(
                "access token must contain 32..256 ASCII letters, digits, underscores or hyphens"
            );
        }
        let mut hosts = BTreeSet::new();
        hosts.insert(address.to_string());
        hosts.insert(format!("localhost:{}", address.port()));
        Ok(Self {
            token: Arc::new(token),
            hosts,
        })
    }

    fn accepts(&self, presented: &str) -> bool {
        if presented.len() != self.token.len() {
            return false;
        }
        presented
            .bytes()
            .zip(self.token.bytes())
            .fold(0u8, |difference, (a, b)| difference | (a ^ b))
            == 0
    }

    fn valid_origin(&self, request: &Request) -> bool {
        let Some(host) = request
            .headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
        else {
            return false;
        };
        if !self.hosts.contains(host) {
            return false;
        }
        if let Some(origin) = request.headers().get(header::ORIGIN) {
            let Ok(origin) = origin.to_str() else {
                return false;
            };
            if origin != format!("http://{host}") {
                return false;
            }
        }
        request
            .headers()
            .get("sec-fetch-site")
            .is_none_or(|value| value != "cross-site")
    }
}

fn valid_token_format(token: &str) -> bool {
    (32..=256).contains(&token.len())
        && token
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
}

fn read_token(path: &Path) -> Result<String> {
    let bytes = synapse_core::identity::read_private_state(path, 257)?;
    let token = String::from_utf8(bytes).context("token file must be UTF-8")?;
    let token = token.trim();
    if !valid_token_format(token) {
        bail!("invalid access-token file format");
    }
    Ok(token.to_owned())
}

/// Read or atomically create an owner-only random secret. Concurrent processes
/// never observe a partially written token. Existing unsafe files fail closed.
/// The token is never logged or placed in a URL.
pub fn token_file(path: &Path) -> Result<String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut directory = std::fs::DirBuilder::new();
    directory.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        directory.mode(0o700);
    }
    directory.create(parent)?;
    synapse_core::identity::validate_owned_directory(parent)?;
    match read_token(path) {
        Ok(token) => return Ok(token),
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound) => {}
        Err(error) => return Err(error),
    }
    let mut random = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut random);
    let token: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
    match synapse_core::identity::publish_private_state_new(path, token.as_bytes()) {
        Ok(()) => Ok(token),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => read_token(path),
        Err(error) => Err(error.into()),
    }
}

pub async fn guard(State(state): State<AppState>, request: Request, next: Next) -> Response {
    if !state.auth.valid_origin(&request) {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error":{"code":"origin_rejected","message":"request origin or host is not allowed"}}))).into_response();
    }
    let path = request.uri().path();
    if path == "/login" || path == "/api/auth" {
        return next.run(request).await;
    }
    // Credentials supplied in URLs are never accepted, even alongside a header.
    if request
        .uri()
        .query()
        .is_some_and(|q| q.split('&').any(|p| p.starts_with("token=")))
    {
        return (
            StatusCode::BAD_REQUEST,
            "credentials must not be supplied in URLs",
        )
            .into_response();
    }
    let bearer = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let explicit = bearer.or_else(|| {
        request
            .headers()
            .get("x-synapse-token")
            .and_then(|v| v.to_str().ok())
    });
    let cookie = request
        .headers()
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|cookies| {
            cookies
                .split(';')
                .find_map(|part| part.trim().strip_prefix("synapse_session="))
        });
    // A browser cookie alone is insufficient for mutations/WS without an Origin.
    let browser_action = !matches!(
        *request.method(),
        axum::http::Method::GET | axum::http::Method::HEAD
    ) || path == "/ws";
    if explicit.is_none()
        && cookie.is_some()
        && browser_action
        && !request.headers().contains_key(header::ORIGIN)
    {
        return (
            StatusCode::FORBIDDEN,
            "browser actions require a same-origin request",
        )
            .into_response();
    }
    if explicit
        .or(cookie)
        .is_some_and(|value| state.auth.accepts(value))
    {
        return next.run(request).await;
    }
    if path == "/" || path == "/contribute" {
        return Redirect::temporary("/login").into_response();
    }
    (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error":{"code":"unauthorized","message":"provide an Authorization: Bearer token or sign in locally"}}))).into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Login {
    token: String,
}

pub async fn login(State(state): State<AppState>, Json(login): Json<Login>) -> Response {
    if !state.auth.accepts(&login.token) {
        return (StatusCode::UNAUTHORIZED, "invalid token").into_response();
    }
    let mut response = Json(serde_json::json!({"ok":true})).into_response();
    response.headers_mut().insert(
        header::SET_COOKIE,
        format!(
            "synapse_session={}; Path=/; HttpOnly; SameSite=Strict; Max-Age=3600",
            state.auth.token
        )
        .parse()
        .expect("validated ASCII token"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
}

pub async fn logout() -> Response {
    let mut response = Json(serde_json::json!({"ok":true})).into_response();
    response.headers_mut().insert(
        header::SET_COOKIE,
        "synapse_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
            .parse()
            .unwrap(),
    );
    response
}

pub async fn page() -> Html<&'static str> {
    Html(
        r#"<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sign in · Synapse</title><style>body{color:#edf0f7;background:#10121a;font:17px system-ui;max-width:32rem;margin:12vh auto;padding:24px}label,input,button{display:block}input{box-sizing:border-box;width:100%;padding:14px;margin:12px 0 20px;background:#1a1e2b;color:white;border:1px solid #66718c;border-radius:8px}button{padding:12px 20px;background:#b9a2ff;border:0;border-radius:8px;font:inherit}a{color:#c4b4ff}:focus-visible{outline:3px solid #81d4fa;outline-offset:4px}p{line-height:1.65}</style><main><h1>Your local Synapse</h1><p>Enter the access token from your private token file. The startup message shows its location.</p><form id="login"><label for="token">Access token</label><input id="token" type="password" autocomplete="off" required minlength="32" maxlength="256"><button>Sign in</button><p id="status" role="status" aria-live="polite"></p></form></main><script>document.getElementById('login').addEventListener('submit',async e=>{e.preventDefault();const input=document.getElementById('token');const status=document.getElementById('status');try{const r=await fetch('/api/auth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:input.value})});input.value='';if(r.ok){location.replace('/')}else{status.textContent='Sign-in failed. Check your token file and try again.'}}catch{status.textContent='Could not reach the local server.'}})</script></html>"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    struct TokenDirectory(std::path::PathBuf);
    impl TokenDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir()
                .join(format!("synapse-auth-test-{:032x}", rand::random::<u128>()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TokenDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn concurrent_token_creators_observe_one_complete_secret() {
        let directory = TokenDirectory::new();
        let path = directory.0.join("access.token");
        let barrier = std::sync::Barrier::new(8);
        let tokens = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        token_file(&path).unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(
            tokens
                .iter()
                .all(|token| token.len() == 64 && token == &tokens[0])
        );
        assert_eq!(token_file(&path).unwrap(), tokens[0]);
    }

    #[test]
    fn invalid_or_oversized_token_file_is_never_replaced() {
        let directory = TokenDirectory::new();
        let path = directory.0.join("access.token");
        for invalid in [
            "short".to_owned(),
            "x".repeat(258),
            format!("{}\nheader", "x".repeat(32)),
        ] {
            synapse_core::identity::write_private_state(&path, invalid.as_bytes()).unwrap();
            assert!(token_file(&path).is_err());
            assert_eq!(std::fs::read_to_string(&path).unwrap(), invalid);
        }
    }

    #[cfg(unix)]
    #[test]
    fn token_files_reject_symlinks_and_shared_permissions() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let directory = TokenDirectory::new();
        let target = directory.0.join("real.token");
        let original = token_file(&target).unwrap();
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let link = directory.0.join("link.token");
        symlink(&target, &link).unwrap();
        assert!(token_file(&link).is_err());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), original);
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(token_file(&target).is_err());
    }

    #[test]
    fn local_auth_rejects_remote_bind_and_header_injection() {
        assert!(LocalAuth::new("a".repeat(32), "0.0.0.0:8080".parse().unwrap()).is_err());
        assert!(
            LocalAuth::new(
                "x\r\nSet-Cookie: bad".repeat(3),
                "127.0.0.1:8080".parse().unwrap()
            )
            .is_err()
        );
    }
    #[test]
    fn same_origin_and_dns_rebinding_checks() {
        let auth = LocalAuth::new("a".repeat(32), "127.0.0.1:8080".parse().unwrap()).unwrap();
        for (host, origin, expected) in [
            ("127.0.0.1:8080", "http://127.0.0.1:8080", true),
            ("evil.example:8080", "http://evil.example:8080", false),
            ("localhost:8080", "null", false),
            ("localhost:8080", "https://evil.example", false),
        ] {
            let req = Request::builder()
                .header(header::HOST, host)
                .header(header::ORIGIN, origin)
                .body(axum::body::Body::empty())
                .unwrap();
            assert_eq!(auth.valid_origin(&req), expected);
        }
    }
}
