//! Embedded static-asset serving for the self-contained HTML GUI.
//!
//! The current frontend uses local HTML, CSS and JavaScript with no
//! CDN or build step. `rust-embed` packs it into the
//! binary at compile time; we serve it at `/` and as the SPA
//! fallback for any other path. The HTML talks to the Rust
//! backend via `/api/*` REST and `/ws` WebSocket on the same
//! origin, so no CORS plumbing is needed at the embed layer.

use axum::{
    body::Body,
    extract::Request,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "web/"]
#[prefix = ""]
pub struct Assets;

pub async fn serve_root() -> Response {
    serve_index()
}

/// Serve the worker onboarding page at a clean `/contribute` URL.
pub async fn serve_contribute() -> Response {
    serve_index()
}

/// Final fallback for any request that didn't match the API or WS
/// router. We pull the path off the original request URI, try to
/// serve a static asset, otherwise serve `index.html` (SPA
/// fallback so any client-side route works on refresh).
pub async fn spa_fallback(req: Request) -> Response {
    let path = req.uri().path().trim_start_matches('/');
    if !path.is_empty()
        && let Some(file) = Assets::get(path)
    {
        return file_response(path, file);
    }
    (StatusCode::NOT_FOUND, "not found").into_response()
}

fn serve_index() -> Response {
    match Assets::get("index.html") {
        Some(file) => file_response("index.html", file),
        None => (
            StatusCode::NOT_FOUND,
            "index.html missing from embed — did the build script copy it to web/?",
        )
            .into_response(),
    }
}

fn file_response(path: &str, file: rust_embed::EmbeddedFile) -> Response {
    let mime = mime_guess::from_path(path)
        .first_or_octet_stream()
        .to_string();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime)
        .body(Body::from(file.data.into_owned()))
        .expect("static response builder should not fail")
}
