# Local device workspace

Build `synapse-gui` and start it on its default loopback address. A private token
file is created automatically; the console reports its path, never its value.
Read that file locally and enter the token on `/login`. The same-origin login
cookie is HttpOnly, SameSite Strict, and expires after one hour. The page embeds
all required HTML/CSS/JavaScript; cached model workflows have no CDN dependency.

The workspace can host a verified model or contribute a worker. It accepts an
explicit listen address and inference memory budget, supports the appropriate
model template or explicit raw mode, and shows actual authenticated peer counts,
resident tensor bytes, model digest, assigned ranges, and live runtime state.
Unsupported per-prompt output-limit overrides are rejected instead of ignored.
Only one generation is admitted in a host session; browser lag is an explicit
incomplete-stream error. No placeholder utilization or reward balance appears.

Worker pause/resume/drain operations use the runtime's acknowledged control
channel. Pause releases the active assignment and its cache. Drain refuses new
assignments and allows the current job up to ten seconds, then pauses. A worker
can schedule a drain between one minute and seven days after startup. Stop waits
for acknowledgement and task cleanup; a failure to acknowledge is reported even
if the local task has to be terminated. These in-process controls do not impose
a process RSS cap. The CLI's separate Linux virtual address-space cap is an
explicit option for a dedicated worker and is not applied to the GUI.

The optional `/v1` API requires an explicitly configured local manifest directory
at server startup. Its local full-model execution is labelled separately from a
distributed host session. Streaming requests have a cancel button and retain
real token usage. An existing local Ollama adapter is labelled local-only; it
uses bounded streaming and the server's authoritative output-token count. It
does not advertise distributed shard memory enforcement.

## Limits still requiring acceptance

Model startup currently holds the exclusive session operation while artifacts
are verified or downloaded. Stop becomes available after startup completes;
cancellable model-download startup needs a separate lifecycle change. Native
screen-reader testing, an independent first-time two-device onboarding test,
and physical/WAN runs remain open. Signed one-time invitation and revocation
primitives exist in core, but the current UI enrolls through explicit pinned
addresses and mutual pairing. It does not claim a completed QR/invitation flow.
