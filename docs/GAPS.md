# Remaining gaps and evidence boundaries

The 34-package local tranche implements substantial runtime, network, product and packaging paths. Its [implementation status](IMPLEMENTATION_STATUS.md), [final execution record](evidence/final-tranche.md), and [machine-readable gates](evidence/gates.json) distinguish code, actual local tests and acceptance. No gate is accepted by this register. The [integrated draft audit](evidence/stage3-cloud.md) records later GitHub publication and what the draft stack established. Planned PR labels identify work packages.

## Required external evidence

| Gap | What the local work establishes | Next evidence needed to close it |
| --- | --- | --- |
| Physical memory benefit | Independent same-host processes use exact partial ranges and model-shaped reservations; process memory can be observed | Two consenting physical machines, actual resident/peak memory, no hidden full copy, correct output beyond each configured individual budget and an explanation of physical capacity |
| Separate-network connectivity | Authenticated local direct/relay transport, QUIC inference, bounded delayed-transport and worker-loss experiments | Actual devices behind separate routers, direct and relayed inference, UDP/CGNAT/firewall constraints, failure/cleanup tests and consented coarse topology |
| Useful model/hardware compatibility | Original Llama reference, pinned SmolLM2 CPU numerical/two- and three-process parity, supported architecture checks, and synthetic GGUF F32/Q8_0/Q4_0 full/sliced parity | Exact approved trained-model revision/template/backend, numerical and language-quality criteria, actual hardware and measured memory/latency. Any pinned-model local result must retain its real source/evidence. |
| License ownership and model rights | Root MIT preserved; explicit signed model grants, unknown/revoked/unauthorized policy rejection | Authorized maintainer resolution of Apache-2.0 metadata discrepancy; reviewed third-party model rights and approver; no inferred redistribution permission |
| Production trust and releases | Local signing/checksum tooling, atomic cache/package activation, key-bound verification, rollback and corruption tests | Identified key owners, independently authenticated trust-key distribution, rotation/revocation drills, reproducible artifact/SBOM and release approval |
| Independent installation/usability | Developer-run actual process/API controls, Linux installer tests, browser keyboard/responsive checks | A fresh independent tester installs, connects physical devices, runs, cancels, upgrades, rolls back and removes each supported package; assistive-technology and physical-mobile checks |
| Community operation and economics | Local/private inference and synthetic fake-credit settlement, refunds and reconciliation | Hosted membership and tenant authorization, transactional reservation/settlement, approved hosts/models, explicit consent, cash budget, durable recovery and accountable pilot operator |
| Community integrity/privacy | Signed identity and bounded encrypted transport; workers execute model-only operations | Independent adversarial review, approved host verification, explicit data threat model, privacy limits and operational drills. Signatures do not prove correct computation or hide activations from workers. |
| Published review and remote CI | Main plus drafts #46–51 and exact-head GitHub checks recorded in the integrated draft audit | Draft review and acceptance of the integrated stack and any later release candidate |

## Remaining implementation and integration boundaries

| Boundary | Implemented behavior | Remaining work or limit |
| --- | --- | --- |
| Output delivery and cancellation | Correlated request/session/epoch handling, bounded queues, duplicate protection, terminal failure before stream completion and acknowledged worker cleanup | Hosted exactly-once financial settlement is not implemented. Preserve actual accepted-output semantics when adding billing. |
| Recovery without prompt persistence | Durable route epochs, request-scoped KV/admission, bounded recovery state and explicit failure after unrecoverable worker loss | General seamless replay across workers/restarts requires approved token-history retention and reference parity; do not persist prompts by default just to claim recovery. |
| Signed invites | Recipient/pool-bound signatures, expiry, single-use ledger and revocation primitives | Current application joins use mutual pairing. Complete signed-invitation enrollment, QR and recovery flows need an actual caller and QA before being described as available. |
| Catalog approval | Explicit publisher keys/rights, expiry/revocation checks, safe regular-file import, immutable content-addressed views, current-policy verification and pin-aware pruning | Production publisher enrollment/distribution and third-party approvals remain owner work. A raw local model-directory development path is distinct from a catalog-enforced production flow. |
| Resource enforcement | Exact model/KV/scratch reservations, bounded protocol allocation, quotas, controls and a dedicated Linux `RLIMIT_AS` option | Reservation estimates are not OS isolation. No portable sandbox, resident-memory ceiling, GPU partition, disk-wide quota, battery or thermal enforcement is claimed. Cache quota counts logical unique artifacts, with bounded metadata overhead outside that count. |
| Session startup control | Actual workspace status and acknowledged idle-worker controls | The GUI serializes startup during model loading; interrupting that load needs additional lifecycle work. Active drain/recovery behavior must keep its separate runtime evidence. |
| Compatibility | `/synapse/0.4.0`, exact schemas, canonical signed encodings and golden bytes | Independent implementations and physical network interoperability remain external evidence. JSON schema validity does not prove signature validity or authorize a job. |
| Security review | Deterministic hostile-input corpus, regression tests, private state handling, repository hygiene and locked SBOM generation | No independent penetration test, long-running coverage-guided fuzz campaign or comprehensive dependency advisory audit is claimed. |

## Product and test scope

The landing has eight deterministic simulator tests, offline semantics/link/contrast checks and browser QA. Its per-device memory math and route comparisons are fictional examples, explicitly labeled; it opens no connections and creates no credits. The application has separate actual server/process tests and separate browser layout/keyboard checks. A labeled static workspace layout fixture rejects mutation; it does not impersonate a connected live session.

Browser iframe viewport checks exercise mobile/tablet CSS. They do not establish physical-phone, touch, native screen-reader or independent human usability. Reduced-motion CSS checks are recorded separately from OS preference emulation. Performance metrics must come from executed, identified workloads; timings on original tiny fixtures cannot support useful-model speed claims.

## Closing a gap

Attach the requirement, source/model revision, environment, exact command/run, actual result, artifact and limitations; include reviewer/owner approval where required. Use `loopback`, `physical_lan`, `physical_wan` or `simulated_wan` explicitly. Keep implemented, locally tested, accepted and released states separate. Preserve failed and blocked runs. Unknown measurements stay missing. Never invent users, geography, speed, earnings, rights or independent approval.
