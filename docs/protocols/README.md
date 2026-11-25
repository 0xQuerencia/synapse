# Versioned implementation contracts

The initial v2 roadmap's `/synapse/control/1.0` and separate inference streams are proposed architecture, not the current encoding. The implementation keeps the existing request/response transport and introduces a breaking **`/synapse/0.4.0`** version for lifecycle/authentication changes. A 0.3 peer must fail protocol negotiation; there is no implicit downgrade or guessed compatibility.

## Artifact and wire contracts

| Contract | Source | Published shape |
| --- | --- | --- |
| Verified artifact manifest | `inference/model_loader.rs`: `ModelManifest`, `ModelArtifact` | [manifest schema](../../schemas/model-manifest-v1.schema.json) |
| Half-open layer slice | `protocol.rs`: `ShardPlan` | [shard schema](../../schemas/shard-plan-v0.4.schema.json) |
| Worker admission lease | `inference/assignment.rs`: `ShardAssignment`, `LeaseKey` | [assignment schema](../../schemas/shard-assignment-v1.schema.json) |
| Worker readiness | `inference/assignment.rs`: `ShardReady` | [readiness schema](../../schemas/shard-ready-v1.schema.json) |
| Signed resident range | `inference/registry.rs`: `SignedAvailability` | [availability schema](../../schemas/signed-availability-v1.schema.json) |
| Tensor representation | `protocol.rs`: `TensorPayload` | [tensor schema](../../schemas/tensor-v0.4.schema.json) |
| Runtime framing | `network/codec.rs`, `network/messages.rs` | [wire vectors](../../schemas/fixtures/wire-v0.4.json) |
| Evidence record | Package/test/release reporting | [evidence schema](../../schemas/evidence-v1.schema.json) |

JSON schemas describe serde object shapes for fixtures and tooling. **Network traffic is bincode, not JSON.** Runtime validation remains authoritative for byte lengths, cross-field invariants, authorized peers, epochs and actual tensor decompression. Schema validation cannot establish a trustworthy publisher, a correct model or a permitted job.

The manifest includes all input artifacts, including tokenizer/configuration. Version 1 fixes struct field order to `schema_version`, `model_id`, `revision`, `weights`, `artifacts`; artifacts are sorted lexically by filename and each uses `filename`, `size_bytes`, `sha256`. Encode compact UTF-8 JSON without ASCII escaping before SHA-256. [The canonical fixture](../../schemas/fixtures/model-manifest-digest-v1.json) pins the exact bytes and digest. Artifact array order must not affect identity. Revision or artifact content changes must affect it. This digest is an integrity receipt, not a publisher signature or redistribution license.

## Frame format and limits

A request/response frame is marker `0xff`, a four-byte unsigned little-endian body length, then fixed-integer little-endian bincode 1 serialization. Unsupported markers, trailing bytes, oversized frames and malicious declared collection lengths must be rejected by the decoder. Enum order is part of this version: adding/reordering variants or changing field encoding requires an explicit compatibility decision and golden-vector update.

The bounds are 64 MiB tensor bytes, at most 8 dimensions, and 4,096 sequence positions. Data request variant 1 and data response variants 1/5 permit a payload up to 64 MiB plus 4,096 bytes. All other control payloads are limited to 64 KiB. Frames have a 15-second read/write deadline; shared encoded-plus-decoded reservations are capped at 128 MiB plus 8,192 bytes. These are upper bounds, not a per-connection memory entitlement. Admission still constrains worker reservations. LZ4's embedded uncompressed size must match tensor metadata; Zstd decompression is bounded to the expected size plus one detection byte and window log 26. The golden `HealthPing` vector has enum index 7 and full frame `ff0400000007000000`; the Rust codec test independently pins the same bytes.

## Authentication and lifecycle

Transport authenticates peer identity; pool membership and assignment authority are additional checks. The 0.4 pairing proof uses HMAC-SHA256 under the normalized pairing secret, bound to domain, protocol, coordinator PeerId, worker PeerId, challenge nonce and timestamp. Peer signatures bind the same transcript and the HMAC. Challenges have a 60-second TTL and maximum five-second future skew. The implementation's exact transcript builder is authoritative; it uses length-prefixed text and fixed-width timestamps, not ambiguous string concatenation.

Signed invitations use a distinct `synapse:private-invite:v1\0` signing domain, version as big-endian u16, length-prefixed pool/issuer/recipient/nonce and big-endian u64 issue/expiry times. They are recipient-bound, one-use and valid for at most 24 hours. A valid signature does not override revocation or pool policy. Helpers and tests must not be described as CLI/node integration until the actual handlers use them.

Prepare/ready/release messages bind each assignment to a 16-hex session ID, nonzero route epoch and 64-hex model manifest digest. Ranges are half-open, total layers are capped at 1,024, context at 4,096 tokens and lease TTL at 1–300 seconds. The coordinator owns embeddings, output head and sampling; workers execute assigned transformer layers. The all-worker barrier requires each expected authenticated peer to echo its exact assignment. Duplicate acknowledgements do not advance it. A resident weight count must be positive and within its memory budget.

`ShardReady.availability` is nullable in the serde data shape for construction and unit tests, but the real coordinator handler requires a matching signed record. Availability signatures cover the domain `synapse-ready-range-v1\0` followed by the fixed-order JSON struct with an empty signature array. They bind peer, lease key, resident range/bytes, backend, sequence and issue/expiry times. Lifetime is at most 300 seconds with five seconds of allowed future skew. The registry retains replay sequence watermarks across disconnects and is bounded to 64 peers. Neither a JSON-schema-valid record nor an advertisement alone authorizes compute or reserves memory. Peer ID parsing, signature verification, current-time checks, membership, replay protection and admission remain mandatory runtime checks.

## Checks

```sh
python3 -m pip install -r schemas/requirements.txt
python3 schemas/check.py
python3 schemas/check.py --source-root . --wire-root .
```

The second command checks JSON Schema, canonical hashes, frame golden vectors and invalid cross-field fixtures. The last additionally checks the integrated Rust struct fields and 0.4 protocol version. It intentionally fails on the unmodified 0.3 baseline. Rust codec/serialization tests remain necessary; these Python checks cannot prove binary round trips, independent implementation interoperability or WAN behavior.
