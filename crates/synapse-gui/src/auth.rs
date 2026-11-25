//! Local authentication with a private token file and same-origin browser access.
use std::{
    collections::BTreeSet,
    io::{Read, Write},
    net::SocketAddr,
    path::Path,
    sync::Arc,
};

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
        if token.len() < 32
            || token.len() > 256
            || !token
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
        {
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

/// Read or create an owner-only random secret. Never print the secret or put it
/// in an URL. Existing files are opened without following a final symlink.
pub fn token_file(path: &Path) -> Result<String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut random = [0u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut random);
            let token: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
            let mut create = std::fs::OpenOptions::new();
            create.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                create.mode(0o600).custom_flags(libc::O_NOFOLLOW);
            }
            match create.open(path) {
                Ok(mut file) => {
                    file.write_all(token.as_bytes())?;
                    file.sync_all()?;
                    return Ok(token);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    options.open(path)?
                }
                Err(error) => return Err(error.into()),
            }
        }
        Err(error) => return Err(error.into()),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > 257 {
        bail!("invalid token file");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            bail!("token file must have owner-only permissions (chmod 600)");
        }
    }
    let mut token = String::new();
    file.read_to_string(&mut token).context("read token file")?;
    Ok(token.trim().to_owned())
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
