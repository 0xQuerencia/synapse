# Stage two: remaining roadmap work and acceptance boundaries

Base: previous handoff head `56a288c1b949b6e11c912f1b6366a23bcf7e71e9` (61 commits; 34 local roadmap work packages). The earlier bundle's checksums were verified. It is incremental and requires the original repository baseline. This environment cannot resolve GitHub, has no Rust toolchain and no PostgreSQL server/driver. The new source is therefore **an additive overlay**, tested independently, not an integrated native build or a GitHub PR. The previous binary/test evidence remains historical and revision-specific. None of the 22 remaining roadmap packages is production-certified by this tranche.

## Package map

| PR | Work added | Remaining acceptance / implementation |
|---|---|---|
| 22 | Operator-provisioned pool/member consent, model/device approval, suspension/revoke, signed presence | Hosted enrollment, owner identity, real workloads |
| 23 | SQLite integer double-entry ledger, capped treasury, immutable rows, projections, reconciliation | Hosted database roles, real PostgreSQL qualification |
| 24 | Expiring quote, atomic reservation, signed scoped assignments, settle/refund, retry and expiry | Actual inference metering and native cancellation, partial delivered-prefix policy |
| 27 | Signed receipts, independent verifier role, server-side synthetic known-answer challenge, invalid claims refunded | Real model validation/audits and anti-collusion |
| 28 | Actual fake-credit balance, activity and First Verified Job badge, dashboard | Opt-in leaderboard, disputes UX, verified real work |
| 29 | Loopback server, independent worker/verifier processes, stop/refund, local backup tooling | Staging infrastructure, TLS, budget, operator, real relays |
| 30 | Authenticated chunk serving, resume, hash/rights checks, bounded loopback peer fallback | Native catalog integration, canonical-origin fallback, actual licensed model delivery, full cache eviction/pinning |
| 31 | Exact 4-layer synthetic route, all stages validate same delivered units, deterministic split | Actual sharded LLM pipeline receipts, accepted token intervals, replay, WAN, trust review |
| 32 | Threat/operating policy and explicit closed gates | Independent review, production controls and owner sign-off |
| 34 | Source review bundle, reproducible local testnet quickstart and release stop | Signed public release, native integration, physical install/rollback and review |
| 35 | Unbilled native API probe, candidate/reference hash and usage comparison | Tensor parallelism, mixed GPU/CPU, fast-LAN tests and real benchmarks |
| 36 | No WebGPU worker implemented; web dashboard is only a client | Browser/mobile compute backend, authenticated transport, real devices and lifecycle |
| 37 | Explicit native contract probe and documented support boundary | Actual Windows/macOS/GPU/ARM code and platform tests |
| 38 | Explicit open-swarm stop conditions and risk register | Independent privacy, integrity, abuse and economic research; no permissionless implementation |
| 46 | SQLite transactional concurrency and crash recovery; PostgreSQL SQL and adapter candidate | Real PG migrations, concurrency, DB privileges, outbox delivery and recovery drills |
| 47 | Seeded economy simulation, seven fictional scenarios, exact conservation and tariff rules | Real measured capacity, funded operating budget and approved tariffs |
| 48 | Scoped random bearer credentials, proof of device key, consent, tenant/role denial, revocation | Hosted OIDC integration, refresh/rotation, production auth and independent review |
| 49 | Signed scoped receipts, synthetic recomputation, independent dispute review, funded compensation | Live verification audit policy, collusion analysis, retention, logical intervals across retry epochs |
| 50 | Verified SQLite snapshot/restore, integrity checks, crash regression, outbox records, suspend/refund | Hosted telemetry/budget, PG WAL and disaster recovery, external outbox processing |
| 54 | Data register, draft trust/privacy/terms governance and operator checklist | Actual owner/legal review and account export/deletion, production implementation |
| 55 | Separate loopback server, two workers and verifier; full and sharded synthetic settlement, fake-credit redemption | Consenting real participants, native AI, true capacity/cost measurements and pilot review |
| 56 | Machine-readable open gates, release assertion fails closed, review package and evidence | Full independent acceptance and published artifacts |

## Decisions and constraints

- **New Python control plane, isolated from Rust inference**: chosen for executable local work without a Rust toolchain. Not approved as the production framework. Python `cryptography` handles Ed25519; custom password crypto is not used.
- **SQLite reference, PostgreSQL candidate**: actual local transaction, concurrency, crash and backup tests exercise SQLite. PostgreSQL migration and adapter are present but not executed. No hosted deployment, money or production secrets.
- **Synthetic fixture only**: the API refuses arbitrary model identity. No prompt accepted; real model bridge is unbilled and tested only against a local contract double. No false claim of distributed LLM settlement or complete model support.
- **Local bearer auth**: tokens are hashed in SQLite and stored in operator-owned local role files. No hosted accounts/OIDC yet; no public listener or TLS deployment.
- **Conservative failure policy**: all failed/cancelled/expired synthetic jobs are refunded in full; verified stages are paid only when all complete. No cross-epoch replay. Paid partial prefix and provider compensation are production policy decisions.
- **Browser QA blocker**: actual headless Chromium navigation to the loopback UI was blocked by an administrator policy. Static HTTP, headers, same-origin API and Node syntax were tested; browser screenshots, interaction and physical mobile/assistive-technology claims remain unverified.
- **No GitHub mutation**: prior explicit publication blocker remains. No new remote branches, PRs, merge, CI, hosted service or release has been created.

## Verification

Run `community/verify.sh`; optional browser QA with local Playwright + Chromium, if allowed, `PYTHONPATH=community python3 community/tools/browser_qa.py`. The release assertion `python3 community/tools/release_gate.py --assert-release` must exit nonzero. Raw outputs and scope are retained in the handoff evidence. Reapply and rerun the prior native tranche checks on the exact checkout before claiming integrated behavior. Do not promote testnet evidence into production acceptance.
