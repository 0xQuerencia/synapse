# Local admin persistence and streaming review

The local admin review found three reproducible implementation gaps and added
regression coverage for the corrected paths.

1. **Token publication:** creating the final token file before writing let a
   second process observe an empty or partial secret. The GUI now uses the
   identity module's exclusive private temporary file and no-overwrite atomic
   publication. Eight concurrent creators converge on the same complete token.
   Reads are bounded to 257 bytes, validate the token format, and reject unsafe
   ownership/permissions, nonregular files and final-component symlinks. Existing
   malformed tokens are preserved and rejected rather than silently replaced.
2. **Configuration writes:** the editor previously used `fs::write` despite
   describing the operation as atomic, accepted arbitrary TOML shapes and used
   the wrong bootstrap key in its default. Writes now validate before touching
   disk, cap content at 64 KiB, use a fixed filename allowlist, reject symlink
   parents/destinations and use an owned directory with atomic private replacement
   and filesystem synchronization. Relay/bootstrap endpoints use the same strict
   address validator as the runtime. Bootstrap uses `peers`; discovery fields and
   pinned identities are validated. Model shortcut tables reject unknown fields
   and path traversal. Checked-in readable configuration remains readable; new
   saved files have mode 0600 on Unix.
3. **SSE terminal status:** a full token queue could drop an error event while a
   later `[DONE]` reached the consumer. Terminal status now receives a bounded
   one-second delivery grace period. `[DONE]` is queued only after terminal status
   and any requested usage were accepted. A slow-consumer regression fills the
   queue, resumes reading and requires the failure to precede `[DONE]`.

Validation on the integrated fixture baseline:

```sh
cargo test --locked -p synapse-gui
cargo clippy --locked -p synapse-gui --all-targets -- -D warnings
cargo fmt --all -- --check
```

The GUI test run passed **17 tests, 0 failures**, including actual Candle fixture
stream/JSON parity, host/origin/authentication checks, busy-host admission,
backpressure, token-file concurrency and configuration persistence. GUI clippy
passed with warnings denied. The root integration records the subsequent whole
workspace run and live browser/server results.

This is an internal implementation review. It does not establish external
penetration-test coverage, hosted multi-user authorization or a WAN release
approval. Unix ownership/mode checks are platform-specific; Windows/macOS
deployment and installer gates remain separate acceptance work.
