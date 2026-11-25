//! Bounded live session stream. Lag is an explicit error, never a skipped token.
use crate::api::AppState;
use axum::{
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;

pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.max_message_size(300 * 1024)
        .max_frame_size(300 * 1024)
        .on_upgrade(move |socket| handle_socket(socket, state))
}
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut tx, mut rx) = socket.split();
    let Some(mut events) = state.sessions.subscribe_events().await else {
        let _ = tx
            .send(Message::Text(
                json!({"type":"error","message":"start a host session before opening its stream"})
                    .to_string(),
            ))
            .await;
        return;
    };
    if tx
        .send(Message::Text(
            json!({"type":"hello","role":"synapse-gui"}).to_string(),
        ))
        .await
        .is_err()
    {
        return;
    }
    loop {
        let (message, close) = tokio::select! {
            event = events.recv() => match event {
                Ok(event) => (json!({"type":"event","id":null,"event":event}), false),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => (json!({"type":"error","message":"stream consumer fell behind; output is incomplete"}), true),
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
            incoming = rx.next() => {
                let Some(Ok(incoming)) = incoming else { break };
                match incoming {
                    Message::Close(_) => break,
                    Message::Text(text) => match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::Prompt {id, text}) if id.len() <= 128 => {
                            match state.sessions.send_prompt(text, None).await {
                                Ok(()) => (json!({"type":"accepted","id":id}), false),
                                Err(error) => (json!({"type":"error","id":id,"message":error.to_string()}), false),
                            }
                        }
                        _ => (json!({"type":"error","message":"invalid prompt message"}), false),
                    },
                    _ => continue,
                }
            }
        };
        if !matches!(
            tokio::time::timeout(
                std::time::Duration::from_secs(1),
                tx.send(Message::Text(message.to_string()))
            )
            .await,
            Ok(Ok(()))
        ) || close
        {
            break;
        }
    }
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ClientMessage {
    Prompt { id: String, text: String },
}
