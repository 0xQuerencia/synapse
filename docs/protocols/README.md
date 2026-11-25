# Versioned implementation contracts

The initial v2 roadmap's `/synapse/control/1.0` and separate inference streams are proposed architecture, not the current encoding. The implementation keeps the existing request/response transport and introduces a breaking **`/synapse/0.4.0`** version for lifecycle/authentication changes. A 0.3 peer must fail protocol negotiation; there is no implicit downgrade or guessed compatibility.

## Artifact and wire contracts

| Contract | Source | Published shape |
| --- | --- | --- |
| Verified artifact manifest | `inference/model_loader.rs`: `ModelManifest`, `ModelArtifact` | [manifest schema](../../schemas/model-manifest-v1.schema.json) |
| Half-open layer slice | `protocol.rs`: `ShardPlan` | [shard schema](../../schemas/shard-plan-v0.4.schema.json) |
| Tensor representation | `protocol.rs`: `TensorPayload` | [tensor schema](../../schemas/tensor-v0.4.schema.json) |
| Runtime framing | `network/codec.rs`, `network/messages.rs` | [wire vectors](../../schemas/fixtures/wire-v0.4.json) |
| Evidence record | Package/test/release reporting | [evidence schema](../../schemas/evidence-v1.schema.json) |

JSON schemas describe serde object shapes for fixtures and tooling. **Network traffic is bincode, not JSON.** Runtime validation remains authoritative for byte lengths, cross-field invariants, authorized peers, epochs and actual tensor decompression. Schema validation cannot establish a trustworthy publisher, a correct model or a permitted job.

The manifest includes all input artifacts, including tokenizer/configuration. Version 1 fixes struct field order to `schema_version`, `model_id`, `revision`, `weights`, `artifacts`; artifacts are sorted lexically by filename and each uses `filename`, `size_bytes`, `sha256`. Encode compact UTF-8 JSON without ASCII escaping before SHA-256. [The canonical fixture](../../schemas/fixtures/model-manifest-digest-v1.json) pins the exact bytes and digest. Artifact array order must not affect identity. Revision or artifact content changes must affect it. This digest is an integrity receipt, not a publisher signature or redistribution license.

## Frame format and limits

A request/response frame is marker `0xff`, a four-byte unsigned little-endian body length, then fixed-integer little-endian bincode 1 serialization. Unsupported markers, trailing bytes, oversized frames and malicious declared collection lengths must be rejected by the decoder. Enum order is part of this version: adding/reordering variants or changing field encoding requires an explicit compatibility decision and golden-vector update.

The existing bounds retained by this contract are 64 MiB tensor bytes, at most 8 dimensions, and 4,096 sequence positions; the frame bound is 64 MiB plus 4,096 bytes. These are upper bounds, not a per-connection memory entitlement. Aggregate in-flight limits, concurrency and admission must independently constrain allocation. The first golden `HealthPing` vector has enum index 7 and full frame `ff0400000007000000`.

## Authentication and lifecycle

Transport authenticates peer identity; pool membership and assignment authority are additional checks. The 0.4 pairing proof uses HMAC-SHA256 under the normalized pairing secret, bound to domain, protocol, coordinator PeerId, worker PeerId, challenge nonce and timestamp. Peer signatures bind the same transcript and the HMAC. Challenges have a 60-second TTL and maximum five-second future skew. The implementation's exact transcript builder is authoritative; it uses length-prefixed text and fixed-width timestamps, not ambiguous string concatenation.

Signed invitations use a distinct `synapse:private-invite:v1\0` signing domain, version as big-endian u16, length-prefixed pool/issuer/recipient/nonce and big-endian u64 issue/expiry times. They are recipient-bound, one-use and valid for at most 24 hours. A valid signature does not override revocation or pool policy. Helpers and tests must not be described as CLI/node integration until the actual handlers use them.

Prepare/ready/release messages must bind the assignment to a session, model digest, range and route epoch. All stages must be ready before generation. Budget admission is atomic; cancellation/expiry frees resources. No signed capability advertisement reserves memory. Full lifecycle type schemas are added when their integrated Rust field layout is stable; the roadmap's illustrative field names must not silently replace the implemented protocol.

## Checks

```sh
python3 -m pip install -r schemas/requirements.txt
python3 schemas/check.py
python3 schemas/check.py --source-root . --wire-root .
```

The second command checks JSON Schema, canonical hashes, frame golden vectors and invalid cross-field fixtures. The last additionally checks the integrated Rust struct fields and 0.4 protocol version. It intentionally fails on the unmodified 0.3 baseline. Rust codec/serialization tests remain necessary; these Python checks cannot prove binary round trips, independent implementation interoperability or WAN behavior.
