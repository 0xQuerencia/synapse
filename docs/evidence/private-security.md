# Private network security review

This is implementation and local regression evidence for PR-19, not a completed
WAN release approval. The scope is a private pool whose operator selects the
peers, model and shared pairing secret. A worker receives plaintext activations
inside its process and must be trusted with those data. Transport encryption
does not make an untrusted inference worker confidential or correct.

## Trust boundaries and controls

| Boundary or threat | Implemented control | Regression evidence |
| --- | --- | --- |
| A different key claims a known device identity | Ed25519 application proof is checked against the authenticated libp2p PeerId | `identity_proof_is_bound_to_transport_peer` |
| Key creation race or local symlink substitution | Exclusive private temporary file, atomic no-overwrite publication, owner/mode and regular-file checks, bounded no-follow reads | `concurrent_creation_converges_on_one_complete_key`, `creation_never_overwrites_an_identity`, `private_keys_reject_symlinks_and_insecure_permissions` |
| Revocation disappears after restart or a concurrent writer | OS lock, reload-and-union, durable atomic replacement before acknowledgement; corrupt or unsafe state fails closed | `independent_revokers_preserve_each_others_durable_entries`, `revoked_old_key_stays_revoked_after_restart_and_rotation`, `failed_revocation_write_does_not_change_memory`, `unsafe_state_lock_cannot_guard_a_security_write` |
| A captured pairing response authorizes another peer/session | Random nonce, 60-second age, peer-bound HMAC plus worker signature, distinct coordinator proof; pending challenge consumed on every attempt | `challenge_proof_rejects_expiry_wrong_secret_wrong_peer_and_replay` |
| Stolen, expired, reused or cross-pool invitation | Recipient/issuer/pool binding, signed expiry, persistent one-use ledger under the same private-state rules | `stolen_expired_wrong_pool_and_tampered_invites_fail`, `separate_redeemers_cannot_spend_the_same_invite`, `revoked_recipient_and_failed_persistence_cannot_enroll` |
| Pairing identity churn or exponential-backoff overflow | 1,024 tracked peers, 15-minute expiry, 128 failures/minute globally, 30–300-second capped per-peer lockout | `pairing_limiter_bounds_churn_and_eventually_expires_partial_attempts`, `pairing_backoff_cannot_overflow_or_be_reset_by_peer_rotation` |
| A discovery result becomes authorization | Manual/pinned candidate policy; LAN and DHT discovery default off; cached address must contain its declared PeerId; application authentication remains separate | `candidate_discovery_does_not_authorize_unknown_peers`, `cached_addresses_are_bound_to_the_named_peer` |
| Oversized, malformed or slowly delivered transport data | 64-KiB controls, bounded data frames and aggregate decoding, 15-second frame deadlines, finite streams/connections, semantic validation before dispatch | codec malformed corpus, `oversized_control_is_rejected_after_tag`, `decoding_budget_is_released_even_on_failure`, `stalled_peer_hits_frame_deadline` |
| Compressed data expands beyond declared tensor shape | Check metadata before decoding; enforce exact output size, LZ4 size prefix and Zstandard window/output bounds | `bounded_decoder_rejects_embedded_lz4_size_before_allocating`, `bounded_zstd_decoder_stops_at_declared_output_limit` |
| Traffic quotas reset by rotating identities | Shared per-peer and global byte/message windows, bounded accounting state; runtime applies the budget to both incoming requests and responses | `per_peer_and_global_limits_cannot_be_bypassed_by_identity_rotation`; runtime `requests_and_responses_share_framed_receive_budget` |
| Late or unrelated replies mutate the current session | Request ID, peer, response kind, session, epoch and trace correlation; one completion; bounded queue and deadlines | `stale_unrelated_and_duplicate_responses_cannot_advance_a_request`, `cancellation_and_timeout_clean_scoped_requests_without_cross_session_loss` |
| Old routes access new state or duplicate displayed output | Durable increasing route epochs; lease-bound KV ownership; output progress advances only after accepted delivery; bounded recovery state | session tests and runtime admission/assignment tests |
| Pause/stop reports success before work is released | Bounded acknowledged command channel; actual worker event-loop cleanup before acknowledgement; drain deadline | `control_acknowledgement_requires_actual_worker_application`, `pause_drain_and_stop_have_distinct_cleanup_semantics` |
| A dedicated Linux worker exceeds its process address-space ceiling | Explicit, irreversible lowering of RLIMIT_AS; kernel-enforced allocation failure | `linux_kernel_enforces_the_dedicated_child_limit` |

The malformed corpus deterministically mutates valid frames and generates 1,024
bounded hostile frames, decoded as both request and response. This is a regression
corpus, not a claim of a long-running coverage-guided fuzzing campaign.

## Runtime review findings

Review with the runtime owner found and corrected these concrete integration
problems: new enum variants initially changed stable bincode ordinals; failed-send
diagnostics printed entire response payloads; receive accounting omitted
responses; authentication state needed cleanup on failure and expiry; a logical
peer disconnect needed to wait until its final transport connection closed.
The node integration commits carry their own tests. The core golden frame remains
`ff0400000007000000` for `HealthPing` under `/synapse/0.4.0`.

The connection tracker now begins in an unknown/unhealthy state until a real
health response arrives. Queuing a dial is never reported as measured RTT.

## Reproduction

Use the repository's pinned Rust toolchain. Run:

```sh
cargo test --locked -p synapse-core --test private_transport
cargo test --locked -p synapse-core --lib --no-default-features
cargo fmt --all -- --check
cargo clippy --locked -p synapse-core --all-targets -- -D warnings
```

The local networking branch's intermediate run explicitly excluded the three
legacy model-loader/pipeline/tokenizer test modules, which downloaded shared
online fixtures and conflicted over cache locks. It passed 150 tests, including
the real Linux child resource-limit check. This intermediate result does not
stand in for the final integrated offline-fixture suite. The root integration
work supplies that result; no failing test has been converted to an ignored test.
`cargo fmt --all -- --check` and core `clippy --all-targets -- -D warnings` also
passed for this branch. The explicit intermediate filter command was:

```sh
cargo test --locked -p synapse-core --lib --no-default-features -- \
  --skip inference::model_loader::tests \
  --skip inference::pipeline::tests \
  --skip inference::tokenizer::tests
```

Local direct and relayed encrypted links have a separate evidence record in
[`private-transport.md`](private-transport.md).

## Remaining release gates and limits

* Signed invites and one-time redemption are core APIs. End-to-end enrollment
  through those invitations still needs an application flow; current node joins
  use mutual pairing. An operator must exchange the generated secret privately
  and select trusted peers. This is not hosted account/pool authorization.
* The generated pairing code has twelve characters from a 32-character alphabet.
  This protocol is not a PAKE. Do not describe a user-chosen low-entropy code as
  resistant to offline guessing by a malicious participant.
* Same-host direct TCP/Noise and Circuit Relay v2/Noise tests establish local
  transport behavior. Physical separate-network WAN, CGNAT, blocked UDP,
  heterogeneous OSes and relay exhaustion remain external acceptance gates.
  Relays need correctly configured reachable advertised addresses.
* RLIMIT_AS constrains virtual address space, including memory-mapped files. It
  is an explicit dedicated-CLI option, not a substitute for resident-memory
  accounting and not an application sandbox. No equivalent Windows/macOS kernel
  limit, battery/thermal enforcement, disk quota or network namespace is claimed.
* The private runtime may fail a generation explicitly after worker loss. The
  recovery state model alone does not demonstrate resumable multi-worker replay.
* GUI HTTP/WebSocket origin/authentication review, dependency advisory review,
  supply-chain evidence and the operational WAN run belong to the integrated
  PR-19/PR-45 release gate. This document does not assert that gate passed or that
  an independent external security audit occurred.

The transport role exposes model/shard operations rather than arbitrary command
execution. This narrow interface still requires artifact-path validation,
manifest integrity and resource admission; those controls are verified with
the model/runtime packages during integration.
