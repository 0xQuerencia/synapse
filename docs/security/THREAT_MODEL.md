# Threat model and review record

This is a design/control register, not a completed external security audit. Passing repository hygiene or an SBOM job does not prove absence of vulnerabilities. Runtime evidence belongs to the exact tested revision.

| Boundary | Threat | Required enforcement | Acceptance evidence |
| --- | --- | --- | --- |
| Untrusted peer to decoder | Oversized frame, hostile length, malformed tensor | Bounded decoding, marker/version validation, aggregate in-flight caps | Codec negative tests, fuzzing and live resource tests |
| Peer identity to cluster membership | Replayed challenge/invite, revoked peer | Transcript-bound proof, recipient-bound invitation, TTL, replay guard and ACL | Authentication integration and revoke tests |
| Assignment to worker | Wrong model/range, overcommit, stale epoch | Atomic local admission and prepare/ready/commit checks | Concurrent admission and wrong-route rejection |
| Worker to host OS | Model code execution, path traversal, runaway compute | Approved artifacts, narrow model operations, unprivileged limits, stop | Real target-OS enforcement and cleanup tests |
| Session to session | KV state or output contamination | Request/model/epoch binding and terminal cleanup | Multiple-session, cancel and recovery tests |
| Browser to local API | Cross-origin admin mutation, DNS rebinding, credential leak | Loopback default, local auth, Host/Origin checks, no URL tokens | API request matrix and browser flow tests |
| Upstream artifacts to cache | Corrupt/replaced weights, unsafe filenames | Pinned manifest, size/hash checks, atomic writes, origin policy | Malformed index, corruption and cache-race tests |
| Hosted control to ledger | Duplicate spend/reward, unauthorized grants | Durable journal, escrow, idempotency and reconciliation | Real database concurrency/crash tests before enabling |
| Release to installation | Modified package, unsafe archive, destructive update | Trusted signature/checksum, safe extraction, atomic pointer, rollback | Tamper/traversal/update/uninstall tests and independent install |

## Privacy boundary

Remote computing peers may see prompts or infer information from activations. Do not describe transport encryption as confidentiality from a worker. Prompt/output/KV persistence is off by default. Diagnostics and financial records must omit prompt content, activations, tokens, invitation codes and private keys. Test with synthetic data. Account deletion, retention and incident reporting need an approved operational policy before hosted use.

## Gating and residual risk

Physical WAN, cross-platform resource limits, independent review, production keys, actual model rights and community economics remain separate requirements. No security checklist activates those modes. Disable affected functionality if authorization, model integrity or accounting is uncertain. Preserve private use when hosted dependencies fail.

The root license discrepancy must be resolved by an authorized maintainer before distributing a signed release. The dependency inventory reports existing package metadata; it does not certify licenses, detect vulnerabilities or resolve this discrepancy.
