# Local HTTP interface

The existing GUI routes under `/api/*` control the local node and are separate from the compatible inference interface. `/api/session/host`, `/api/session/join`, `/api/session/stop` and `/api/session/prompt` must not be treated as public community endpoints. Consult registered routes and CLI help for the integrated revision.

PR-21 defines an intentionally narrow local compatible subset: `GET /v1/models` and `POST /v1/chat/completions`. Accepted fields are `model`, `messages`, `max_tokens`, `temperature`, `stream`, and `stream_options.include_usage`. Initial deterministic sampling accepts `temperature: 0`; unsupported sampling values, tools, functions, images and unknown fields return an error. The exact implemented limits are authoritative and must be tested before publishing a broader compatibility claim.

The raw template supports a single user message. A ChatML template may be used only for a model whose template is explicitly configured and validated. Do not silently wrap arbitrary model families with a SmolLM-specific template. Model identity and availability reflect actual verified local state; no fabricated catalog entries.

Authentication uses a local credential and protected file/bootstrap flow; do not put tokens in URLs. Keep the listener on loopback unless separately configured and reviewed. Validate Host/Origin and limit body size before dispatch. Administrative credentials do not silently authorize unknown external compute.

An SSE response carries completion identity, role/content deltas and a terminal finish reason, with an optional usage event when requested, then `data: [DONE]`. Errors have stable `code` and `message` fields. A disconnect/cancel must stop generation and release its resources; a partially written stream cannot be converted into a successful completion. The exact delivery/cancellation boundary must be covered by integration tests.

This document specifies the PR-21 subset; its presence does not establish that an endpoint is live at every earlier revision. Community quote, reserve, balance and receipt APIs remain separately gated. Private/local inference does not make billing calls.
