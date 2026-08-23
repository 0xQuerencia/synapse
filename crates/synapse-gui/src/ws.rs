//! WebSocket bridge: forwards browser prompts into the master session
//! and streams `ChatEvent`s back.
//!
//! Wire format (all messages are JSON text frames):
//!
//! Client → Server:
//! ```json
//! { "type": "prompt", "id": "uuid", "text": "Hello" }
//! ```
//!
//! Server → Client:
//! ```json
//! { "type": "event", "id": "uuid-or-null", "event": { ...UiChatEvent... } }
//! ```

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
use tokio::sync::broadcast;

use crate::{api::AppState, session::UiChatEvent};

pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut tx, mut rx) = socket.split();

    let mut bcast_rx: broadcast::Receiver<UiChatEvent> =
        state.sessions.subscribe_events().await.unwrap_or_else(|| {
            // No master session running yet — give the client a closed
            // channel. They'll need to reconnect (or reload) after a
            // session starts. The Dashboard polling will reflect that.
            let (_tx, rx) = broadcast::channel::<UiChatEvent>(1);
            rx
        });

    let _ = tx
        .send(Message::Text(
            json!({ "type": "hello", "role": "synapse-gui" }).to_string(),
        ))
        .await;

    let send_task = async move {
        loop {
            match bcast_rx.recv().await {
                Ok(ev) => {
                    let msg = json!({ "type": "event", "id": null, "event": ev });
                    if tx.send(Message::Text(msg.to_string())).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    let recv_task = async move {
        while let Some(Ok(msg)) = rx.next().await {
            let Message::Text(text) = msg else { continue };
            let Ok(ClientMessage::Prompt {
                id,
                text: prompt_text,
            }) = serde_json::from_str::<ClientMessage>(&text)
            else {
                continue;
            };
            if let Err(e) = state.sessions.send_prompt(prompt_text, None).await {
                tracing::warn!(error = %e, "failed to send ws prompt");
            }
            let _ = id;
        }
    };

    tokio::select! {
        _ = send_task => {}
        _ = recv_task => {}
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    Prompt { id: String, text: String },
}
