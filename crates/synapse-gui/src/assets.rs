//! Embedded static-asset serving for the React/Vite production build.
//!
//! `rust-embed` packs `web/dist/` into the binary at compile time.
//! The router serves `index.html` at `/` and falls back to
//! `index.html` for any unknown path so React Router client routes
//! work on refresh.

use axum::{
    body::Body,
    extract::Request,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "web/dist/"]
pub struct Assets;

pub async fn serve_root() -> Response {
    serve_index()
}

/// Final fallback for any request that didn't match the API or WS
/// router. We pull the path off the original request URI, try to
/// serve a static asset, otherwise serve `index.html`.
pub async fn spa_fallback(req: Request) -> Response {
    let path = req.uri().path().trim_start_matches('/');
    if !path.is_empty() {
        if let Some(file) = Assets::get(path) {
            return file_response(path, file);
        }
    }
    serve_index()
}

fn serve_index() -> Response {
    match Assets::get("index.html") {
        Some(file) => file_response("index.html", file),
        None => (StatusCode::NOT_FOUND, "index.html missing from embed").into_response(),
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
