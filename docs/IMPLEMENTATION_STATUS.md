# Implementation and evidence status

The current tranche covers **34 of the master plan's 56 work-package IDs**: 01–21, 25–26, 33, 39–45 and 51–53. These are implementation boundaries, with local tests and explicit remaining acceptance work. They are not 34 independently approved or merged GitHub pull requests. The master plan's physical, owner and production dependencies still apply even when a later package has useful code prepared early.

The [final tranche record](evidence/final-tranche.md) is the authority for the integrated revision, final test counts, commands, binary/artifact identifiers and unresolved failures. Intermediate records retain their original revisions and limits. Do not attach the original imported SHA to a modified checkout or reuse a filtered branch test count as the final workspace result.

See the [integrated draft audit](evidence/stage3-cloud.md) for later published drafts and their revision-specific results. The counts below describe the original local tranche.

## Implemented work and evidence boundaries

| Packages | Concrete implementation in this tranche | Local evidence and limits |
| --- | --- | --- |
| 01–02 | Reconciled baseline, pinned toolchain, fail-fast verification, approved plan, source/provenance and honest status | [Foundation record](evidence/foundation.md). Root license/metadata rights decision remains open. |
| 03–04, 40 | Pinned manifest, safe filenames/index values, complete hash verification, original offline Llama fixture and independent scalar reference | Deterministic parity and malformed-input tests. Useful third-party checkpoints require separate approval and compatibility results. |
| 05–07 | Exact contiguous routes, prepare/ready/release barrier, session/model/epoch binding and actual partial tensor residency | Planner and runtime tests plus independent-process execution. Configured budgets on one host do not establish physical RAM savings. |
| 08–09, 18 | Real process harnesses, retained JSONL, reference tokens, observed memory/timing, delayed transport and explicit worker-loss outcomes | Final report identifies actual loopback and simulated-WAN runs. Real LAN/WAN and inter-city tests remain outstanding. |
| 10, 42–43 | Model-shaped weight/KV/scratch admission, exact reservations, lease-scoped state, acknowledged controls and a Linux process address-space option | Local runtime/kernel tests. No cross-platform sandbox, GPU-memory isolation, thermal or battery enforcement is claimed. |
| 11–12 | Signed, expiring resident ranges and bounded route selection using measured topology/throughput inputs | Signature/replay/expiry and feasible-route tests. Discovery hints and unmeasured RTT are never evidence of readiness. |
| 13–17, 19, 41, 44 | Private pairing, device identity/revocation, signed invitation primitives, conservative discovery, bounded direct/relay transport, request correlation, recovery state and abuse controls | [Private security](evidence/private-security.md) and [transport evidence](evidence/private-transport.md). Pairing is integrated; a complete signed-invitation enrollment UI and physical WAN acceptance remain open. Worker loss may terminate explicitly; general seamless recovery is not promised. |
| 20–21, 26 | Bounded full-model backend, local compatible chat API, synthetic GGUF F32/Q8_0/Q4_0 full/sliced parity and F32 GGUF/safetensors parity | [Live API/GUI evidence](evidence/gui-http-smoke.md) and final format matrix. Synthetic quantization checks do not establish trained-model quality. Exact errors and any real pinned-model outcome belong to the final report; no generic GGUF, accelerator or speed claim follows. |
| 25 | Offline device onboarding, verified local model paths, doctor reservation output, actual peer/route status and worker pause/resume/drain/stop | Real server/process smoke plus [workspace browser checks](evidence/device-layout-qa.md). A fresh independent user's two-device setup remains a gate. |
| 33, 53 | Responsive research landing, true per-device simulation math, semantic/link/contrast checks, keyboard/mobile layout work and live app QA | [Landing QA](evidence/landing-qa.md), [workspace QA](evidence/device-layout-qa.md). Browser iframe widths are layout evidence, not physical-phone or screen-reader certification. |
| 39, 45 | Versioned schemas, signed canonical encodings and golden vectors, threat model, bounded repository checks and locked dependency SBOM tooling | Contract and hygiene suites. No broad dependency advisory audit or independent penetration test is claimed. |
| 51–52 | Explicit publisher trust/rights/revocation, atomic verified cache with pins/quotas, local package signatures/checksums, update/rollback/uninstall | [Catalog evidence](evidence/catalog-cache-local.json) and [installer contract](release-installation.md). Test keys and synthetic artifacts do not establish production trust, third-party rights or a supported release. |

## Acceptance gates

| Gate | Required acceptance | Current gate state |
| --- | --- | --- |
| G0 local foundation | Integrated checks, deterministic parity, reconciled rights and review | Local implementation/evidence exists; final aggregate verification and owner/release review remain distinct. Not accepted. |
| G1 real sharding | Two physical devices, exact resident ranges, no hidden full copy and measured individual/combined memory benefit | Awaiting physical evidence. Same-host processes and configured reservations are insufficient. |
| G2 private WAN | Separate routers, consenting peers, actual authenticated direct/relay inference, failure tests and review | Awaiting physical WAN evidence. Local relay and simulated impairment results are labeled separately. |
| G3 community full model | Approved pools/models, consent, durable accounting, verification, budget and pilot review | Disabled. Private/local API and catalog tools do not enable a community economy. |
| G4 community sharding | Complete multi-stage correctness, bounded recovery and exact stage settlement | Disabled pending community and separate security/privacy acceptance. |
| G5 closed beta | Trusted release signing, independent install/update/uninstall, support matrix and go/no-go | No released artifact attested. Installer code and local tests are preparation. |
| G6 open swarm | Independent adversarial, privacy and economic review | Research only. |

The machine-readable [gate record](evidence/gates.json) keeps every `accepted` flag false. Its local evidence references are explicitly separate from acceptance evidence. The [gap register](GAPS.md) names the next concrete proof for each unresolved boundary.

## Publication status

At completion of this local implementation tranche: **zero GitHub PRs published, zero release artifacts published, and zero production deployments**. Commit labels map to planned package IDs. Publication and external review require their own authorized step. Remote CI for an unpublished integrated branch is not claimed; the root execution report records local checks and prepared CI workflows separately.

## Status terminology

- **Proposed:** planned scope, no implementation claim.
- **Implemented:** concrete code/document changes with an actual integration path.
- **Locally tested:** named tests ran with recorded source/environment/results.
- **Awaiting evidence:** physical, network, independent, owner or production proof is missing.
- **Accepted:** every applicable criterion has evidence and review.
- **Released:** an identified published artifact passed its release gate.

Before changing a claim, record the requirement, exact command/run, environment, model/source revision, actual result, limitations and reviewer where required. Use `loopback`, `physical_lan`, `physical_wan` or `simulated_wan` explicitly. Unknown measurements remain missing, not zero. Preserve failure evidence and publish only redacted synthetic-test artifacts.
