//! A deliberately bounded, authenticated subset of the chat completions API.
use crate::api::AppState;
use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
};
use futures::Stream;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use synapse_core::backend::{Cancellation, CandleBackend, ChatMessage, FinishReason};
use tokio::sync::{Semaphore, mpsc};

#[derive(Clone)]
pub struct InferenceService {
    model_id: String,
    backend: Arc<Mutex<CandleBackend>>,
    slot: Arc<Semaphore>,
}

impl InferenceService {
    pub fn new(backend: CandleBackend) -> Self {
        Self {
            model_id: backend.model_id().to_owned(),
            backend: Arc::new(Mutex::new(backend)),
            slot: Arc::new(Semaphore::new(1)),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f64>,
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

fn error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"error":{"message":message,"type":"invalid_request_error","code":code}})),
    )
        .into_response()
}

pub fn router() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/v1/models", axum::routing::get(models))
        .route("/v1/chat/completions", axum::routing::post(chat))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = state
        .inference
        .as_ref()
        .map(|service| {
            vec![json!({"id":service.model_id,"object":"model","created":0,"owned_by":"local"})]
        })
        .unwrap_or_default();
    Json(json!({"object":"list","data":data}))
}

struct CancelOnDrop(Cancellation);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

struct TokenStream {
    receiver: mpsc::Receiver<Event>,
    cancellation: Cancellation,
}
impl Stream for TokenStream {
    type Item = Result<Event, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(context).map(|item| item.map(Ok))
    }
}
impl Drop for TokenStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// Bounded output queue with a deadline. Slow clients cannot pin a worker forever.
fn send_event(
    sender: &mpsc::Sender<Event>,
    mut event: Event,
    cancellation: &Cancellation,
    deadline: Instant,
) -> bool {
    loop {
        if cancellation.is_cancelled() || Instant::now() >= deadline {
            return false;
        }
        match sender.try_send(event) {
            Ok(()) => return true,
            Err(mpsc::error::TrySendError::Closed(_)) => return false,
            Err(mpsc::error::TrySendError::Full(returned)) => {
                event = returned;
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

async fn chat(
    State(state): State<AppState>,
    payload: Result<Json<ChatRequest>, JsonRejection>,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(_) => {
            return error(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                "invalid JSON, unsupported field, or unsupported message shape",
            );
        }
    };
    let Some(service) = state.inference else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_unavailable",
            "start the server with an approved --model-dir to enable inference",
        );
    };
    if request.model != service.model_id {
        return error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "requested model is not hosted here",
        );
    }
    if request
        .temperature
        .is_some_and(|value| !value.is_finite() || value != 0.0)
    {
        return error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "only deterministic temperature=0 is supported",
        );
    }
    if request.stream_options.is_some() && !request.stream {
        return error(
            StatusCode::BAD_REQUEST,
            "invalid_stream_options",
            "stream_options requires stream=true",
        );
    }
    let permit = match service.slot.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "host_busy",
                "host is serving another request; retry after it completes",
            );
        }
    };
    let prepared = service
        .backend
        .lock()
        .map_err(|_| "backend unavailable".to_owned())
        .and_then(|backend| {
            let prompt = backend
                .render_chat(&request.messages)
                .map_err(|e| e.to_string())?;
            let max_tokens = request.max_tokens.unwrap_or(
                256.min(backend.limits().max_output_tokens)
                    .min(backend.limits().max_context_tokens / 2),
            );
            backend
                .validate_prompt(&prompt, max_tokens)
                .map_err(|e| e.to_string())?;
            Ok((prompt, max_tokens, backend.limits().max_duration))
        });
    let (prompt, max_tokens, max_duration) = match prepared {
        Ok(value) => value,
        Err(message) => return error(StatusCode::BAD_REQUEST, "invalid_request", &message),
    };
    let id = format!("chatcmpl-{:032x}", rand::random::<u128>());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let cancellation = Cancellation::default();
    let worker_cancel = cancellation.clone();
    let model = service.model_id.clone();
    if request.stream {
        let (sender, receiver) = mpsc::channel(8);
        let include_usage = request
            .stream_options
            .is_some_and(|options| options.include_usage);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let deadline = Instant::now() + max_duration;
            let chunk = |delta: Value, finish: Value| {
                json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]}).to_string()
            };
            if !send_event(
                &sender,
                Event::default().data(chunk(json!({"role":"assistant"}), Value::Null)),
                &worker_cancel,
                deadline,
            ) {
                return;
            }
            let result = service
                .backend
                .lock()
                .map_err(|_| anyhow::anyhow!("backend lock poisoned"))
                .and_then(|mut backend| {
                    backend.generate(&prompt, max_tokens, &worker_cancel, |token| {
                        Ok(send_event(
                            &sender,
                            Event::default()
                                .data(chunk(json!({"content":token.text}), Value::Null)),
                            &worker_cancel,
                            deadline,
                        ))
                    })
                });
            match result {
                Ok(usage)
                    if !matches!(
                        usage.finish_reason,
                        FinishReason::Deadline | FinishReason::Cancelled
                    ) =>
                {
                    let reason = if usage.finish_reason == FinishReason::Eos {
                        "stop"
                    } else {
                        "length"
                    };
                    let _ = send_event(
                        &sender,
                        Event::default().data(chunk(json!({}), json!(reason))),
                        &worker_cancel,
                        deadline,
                    );
                    if include_usage {
                        let value = json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[],"usage":{"prompt_tokens":usage.input_tokens,"completion_tokens":usage.output_tokens,"total_tokens":usage.input_tokens + usage.output_tokens}});
                        let _ = send_event(
                            &sender,
                            Event::default().data(value.to_string()),
                            &worker_cancel,
                            deadline,
                        );
                    }
                }
                Ok(_) => {
                    let _ = sender.try_send(Event::default().event("error").data(json!({"error":{"code":"generation_interrupted","message":"generation exceeded its deadline or the stream stopped accepting output"}}).to_string()));
                }
                Err(_) => {
                    let _ = sender.try_send(Event::default().event("error").data(json!({"error":{"code":"generation_failed","message":"inference failed"}}).to_string()));
                }
            }
            let _ = send_event(
                &sender,
                Event::default().data("[DONE]"),
                &worker_cancel,
                deadline + Duration::from_secs(1),
            );
        });
        Sse::new(TokenStream {
            receiver,
            cancellation,
        })
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
        .into_response()
    } else {
        let _cancel_on_drop = CancelOnDrop(cancellation);
        let task = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut backend = service
                .backend
                .lock()
                .map_err(|_| anyhow::anyhow!("backend unavailable"))?;
            let mut text = String::new();
            let usage = backend.generate(&prompt, max_tokens, &worker_cancel, |token| {
                text.push_str(&token.text);
                Ok(true)
            })?;
            Ok::<_, anyhow::Error>((text, usage))
        });
        match tokio::time::timeout(max_duration + Duration::from_secs(1), task).await {
            Ok(Ok(Ok((text, usage))))
                if !matches!(
                    usage.finish_reason,
                    FinishReason::Deadline | FinishReason::Cancelled
                ) =>
            {
                let reason = if usage.finish_reason == FinishReason::Eos {
                    "stop"
                } else {
                    "length"
                };
                Json(json!({"id":id,"object":"chat.completion","created":created,"model":model,"choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":reason}],"usage":{"prompt_tokens":usage.input_tokens,"completion_tokens":usage.output_tokens,"total_tokens":usage.input_tokens + usage.output_tokens}})).into_response()
            }
            Err(_) | Ok(Ok(Ok(_))) => error(
                StatusCode::GATEWAY_TIMEOUT,
                "generation_timeout",
                "inference exceeded its deadline",
            ),
            _ => error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_failed",
                "inference failed",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use tower::ServiceExt;

    fn application() -> (axum::Router, InferenceService) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../synapse-core/tests/fixtures/tiny-llama");
        let files =
            synapse_core::inference::model_loader::ModelFiles::from_directory(&path).unwrap();
        let backend = CandleBackend::load(
            "tiny".into(),
            &files,
            synapse_core::backend::BackendLimits::default(),
            synapse_core::backend::PromptFormat::Raw,
        )
        .unwrap();
        let service = InferenceService::new(backend);
        let state = AppState {
            sessions: crate::session::SessionStore::new(),
            auth: Arc::new(
                crate::auth::LocalAuth::new("a".repeat(32), "127.0.0.1:8080".parse().unwrap())
                    .unwrap(),
            ),
            inference: Some(service.clone()),
        };
        let app = router()
            .route("/api/auth", axum::routing::post(crate::auth::login))
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                crate::auth::guard,
            ))
            .with_state(state);
        (app, service)
    }

    fn request(body: Value) -> Request<Body> {
        Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("host", "127.0.0.1:8080")
            .header("authorization", format!("Bearer {}", "a".repeat(32)))
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn body(stream: bool) -> Value {
        json!({"model":"tiny","messages":[{"role":"user","content":"<s> </s> Hello"}],"max_tokens":8,"stream":stream,"temperature":0})
    }

    #[tokio::test]
    async fn actual_backend_stream_and_json_have_equal_text_and_usage() {
        let (app, _) = application();
        let response = app.clone().oneshot(request(body(false))).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let result: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(
            result["usage"],
            json!({"prompt_tokens":3,"completion_tokens":8,"total_tokens":11})
        );
        assert_eq!(result["choices"][0]["finish_reason"], "length");
        let mut input = body(true);
        input["stream_options"] = json!({"include_usage":true});
        let response = app.oneshot(request(input)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response.headers()["content-type"]
                .to_str()
                .unwrap()
                .starts_with("text/event-stream")
        );
        let bytes = to_bytes(response.into_body(), 65536).await.unwrap();
        let events = std::str::from_utf8(&bytes).unwrap();
        assert!(events.ends_with("data: [DONE]\n\n"));
        let mut text = String::new();
        let mut usage = Value::Null;
        for line in events
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
        {
            let value: Value = serde_json::from_str(line).unwrap();
            if let Some(part) = value["choices"][0]["delta"]["content"].as_str() {
                text.push_str(part);
            }
            if value.get("usage").is_some() {
                usage = value["usage"].clone();
            }
        }
        assert_eq!(text, result["choices"][0]["message"]["content"]);
        assert_eq!(usage, result["usage"]);
    }

    #[tokio::test]
    async fn authentication_origin_schema_and_admission_are_enforced() {
        let (app, service) = application();
        let mut missing = request(body(false));
        missing.headers_mut().remove("authorization");
        assert_eq!(
            app.clone().oneshot(missing).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
        let mut foreign = request(body(false));
        foreign
            .headers_mut()
            .insert("origin", "https://evil.example".parse().unwrap());
        assert_eq!(
            app.clone().oneshot(foreign).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        let mut rebound = request(body(false));
        rebound
            .headers_mut()
            .insert("host", "evil.example:8080".parse().unwrap());
        assert_eq!(
            app.clone().oneshot(rebound).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        let mut tools = body(false);
        tools["tools"] = json!([]);
        assert_eq!(
            app.clone().oneshot(request(tools)).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
        let mut wrong = body(false);
        wrong["model"] = json!("other");
        assert_eq!(
            app.clone().oneshot(request(wrong)).await.unwrap().status(),
            StatusCode::NOT_FOUND
        );
        let _permit = service.slot.clone().acquire_owned().await.unwrap();
        assert_eq!(
            app.oneshot(request(body(false))).await.unwrap().status(),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[tokio::test]
    async fn token_query_is_rejected_and_login_sets_strict_cookie() {
        let (app, _) = application();
        let mut query = request(body(false));
        *query.uri_mut() = "/v1/chat/completions?token=secret".parse().unwrap();
        assert_eq!(
            app.clone().oneshot(query).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
        let login = Request::builder()
            .uri("/api/auth")
            .method("POST")
            .header("host", "127.0.0.1:8080")
            .header("origin", "http://127.0.0.1:8080")
            .header("content-type", "application/json")
            .body(Body::from(json!({"token":"a".repeat(32)}).to_string()))
            .unwrap();
        let response = app.oneshot(login).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let cookie = response.headers()["set-cookie"].to_str().unwrap();
        assert!(cookie.contains("HttpOnly"));
        assert!(cookie.contains("SameSite=Strict"));
    }
    #[test]
    fn unsupported_fields_and_non_text_messages_are_rejected() {
        for body in [
            json!({"model":"tiny","messages":[],"tools":[]}),
            json!({"model":"tiny","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}),
            json!({"model":"tiny","messages":[],"stream_options":{"unknown":true}}),
        ] {
            assert!(serde_json::from_value::<ChatRequest>(body).is_err());
        }
    }
    #[test]
    fn closed_stream_cancels_work() {
        let cancellation = Cancellation::default();
        let (_, receiver) = mpsc::channel(1);
        drop(TokenStream {
            receiver,
            cancellation: cancellation.clone(),
        });
        assert!(cancellation.is_cancelled());
    }
    #[test]
    fn stalled_client_has_a_bounded_wait() {
        let (sender, _receiver) = mpsc::channel(1);
        sender.try_send(Event::default().data("first")).unwrap();
        let started = Instant::now();
        assert!(!send_event(
            &sender,
            Event::default().data("second"),
            &Cancellation::default(),
            started + Duration::from_millis(10)
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
