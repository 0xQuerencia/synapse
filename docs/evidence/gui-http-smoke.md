# Live GUI, API and worker smoke evidence

`scripts/gui_smoke.py` starts two actual GUI server processes with separate
temporary homes, device identities, token files and configuration directories.
It accepts an already-built binary, binds ephemeral loopback ports, uses the
offline tiny-Llama fixture and cleans up both process groups. It never builds a
binary or downloads a model. Credential values stay in memory/private temporary
files; saved logs are checked and redacted before publication.

```sh
python scripts/gui_smoke.py \
  --binary /absolute/path/to/synapse-gui \
  --out artifacts/gui-smoke \
  --source-revision SOURCE_REVISION \
  --require-websocket
```

The output directory must be new so a failing rerun cannot overwrite earlier
evidence. Python's standard library handles HTTP and process management. The
optional `websockets` sync client enables token verification; `--require-websocket`
makes its absence a failure instead of a labeled skip.

## Observed run

The run against source revision supplied as `401386c` passed. The binary's actual
SHA-256 was
`b6975af01233fab0487295a4a5740f76bfbe849a64a4ffd38a7bc680998516ff`.
The fixture manifest SHA-256 was
`1d9796ab2f4fdb6549a96b3c75835347f69e351c0b961169785592b6d57893b0`.
The complete machine-readable result and redacted logs were saved to
`artifacts/gui-smoke-first/` by the integration runner.

| Live check | Result |
| --- | --- |
| Unauthenticated `/v1/models` | HTTP 401 |
| Foreign Origin and Host | HTTP 403 for each |
| Credential query parameter | HTTP 400 |
| Explicit identity path across HTTP handler threads | Honored |
| Invalid bootstrap edit after a valid edit | HTTP 400; valid prior contents preserved |
| Worker control before a worker exists | HTTP 400 |
| Nonstream fixture completion | 3 prompt tokens, 8 completion tokens, `length` termination |
| SSE fixture completion | Same text/usage, 8 content events, one `[DONE]` after terminal information |
| Two GUI node identities | Distinct |
| Real worker join over loopback QUIC | Mutual authentication completed |
| Pause, resume, idle drain, resume | Acknowledgements matched live `worker_state` |
| Worker stop | Acknowledged `stopped`, zero active jobs, task exited |
| Session cleanup after acknowledged stop | Returned successfully and reported idle |
| Distributed WebSocket generation | All 8 reference token IDs and positions matched |
| Actual worker execution | Routed prefill and 7 decode-completion records |
| Process cleanup | Both GUI processes exited |
| Credential values in saved logs | None detected |

The distributed IDs were `45, 20, 61, 20, 46, 8, 60, 34`, with positions `0..7`.
The host was explicitly coordinator-only, so its successful output required
the authenticated worker to execute the transformer range. The harness checks
actual prefill routing and worker decode-completion logs as additional evidence.

## Scope limits

This proves same-host process, HTTP, WebSocket and QUIC behavior for the offline
fixture. It does not prove separate physical devices, WAN/CGNAT connectivity,
heterogeneous hardware, browser accessibility, or production performance. The
live controls used an idle worker; active-job drain deadline and cleanup have
separate runtime tests. The GUI still serializes session startup while model
loading, so stopping a loading session remains a documented follow-up gate.
