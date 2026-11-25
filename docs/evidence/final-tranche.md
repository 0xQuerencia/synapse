# Synapse implementation tranche: final local verification

## Outcome and scope

Code and review material now cover **34 distinct planned work packages**: PR-01–21, PR-25–26, PR-33, PR-39–45 and PR-51–53. The [34 draft descriptions](../pr-drafts/README.md) explain each package's behavior, files, reproduction steps and outstanding acceptance. A planned PR label is not a published GitHub PR number. These changes are a substantial local implementation tranche, not an acceptance claim for all 56 roadmap packages.

**Published GitHub PRs: 0. Published release artifacts: 0. Production deployments: 0.** Automatic approval review rejected the first GitHub `create_tree` source-upload attempt for `0xQuerencia/synapse` because that destination required explicit publishing authorization. No GitHub write succeeded. The local changes, draft bodies, exact review refs, patches and evidence are prepared so publication can be reviewed as a concrete next step. No alternate remote-write path was used.

All seven release/acceptance gates remain unaccepted in [gates.json](gates.json). Physical-device, separate-network, independent-review, license-owner and production-release evidence cannot be supplied by these local checks. The remaining implementation boundaries are recorded in [GAPS.md](../GAPS.md).

## Revisions, environment and evidence locations

- Imported repository baseline: `1785da5b5f1f87ac6ec008a022ebd68ee22be7cc`.
- **Final executable source revision:** `aef2c5cd204d1740533a41b95eaac70fba6bef20`; working tree clean during build and all live process checks.
- Rust: `rustc 1.94.0 (4a4ef493e 2026-03-02)`. Environment: `Linux-6.18.57-x86_64-with-glibc2.41`. CPU backend, Linux x86_64, nine reported CPU threads; no accelerator was available.
- Build settings: `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`, `CARGO_BUILD_JOBS=6`; locked dependency graph. These are development-profile validation binaries, not published release builds.
- CLI SHA-256: `94bddbb7349111214c292203e68f4e2b210b2f76578bf0e5f077e09219e36b2b`.
- GUI SHA-256: `d4f323cbd87ad06c78568f9235e8ad5e70252da11a680b9f1120fddebe4a7154`.
- Verification date: 2026-11-25 UTC. Later handoff commits update documentation/evidence only; the executable source identifier above remains authoritative.

The handoff includes `artifacts/final-checks/` and `artifacts/final-processes/`, plus a chronological stack manifest and incremental Git bundle. The review bundle's metadata records the final documentation head separately from the tested executable source. Logs retain source/model revisions and actual event counts. Private keys, access tokens, pairing secrets, downloaded third-party weights and compiled application binaries are excluded from the handoff.

## What was implemented

| Planned packages | Integrated behavior |
| --- | --- |
| 01–04, 39–40 | Reproducible locked Rust checks; approved plan/status/provenance; immutable complete model manifests; strict paths, artifacts and architecture semantics; independent original fixture; explicit prompt/tokenizer/EOS/context behavior; executable contracts and pinned trained-model comparison. |
| 05–09, 43 | One exact contiguous layer plan; authenticated versioned shard preparation and readiness barrier; true assigned-range/IO loading; atomic capacity leases and request-scoped KV state; two/three-process harness, doctor, actual token timings and redacted failure evidence. |
| 10–12 | Model-shaped weights/KV/scratch/load reservations; measured CPU capabilities; signed expiring resident-range advertisements; bounded route selection using feasible capacity and measured topology/throughput evidence. |
| 13–19, 41–42, 44–45 | Identity-bound private pairing, signed invite/revocation primitives, private atomic key storage, opt-in discovery, encrypted direct/relay transport, quotas, correlated bounded streams, durable epochs, explicit worker-loss failure, acknowledged controls, Linux address-space option, hostile-input tests and supply-chain tooling. |
| 20–21 | Bounded Candle full-model backend; authenticated local models/chat endpoints; documented compatible subset; real SSE and usage; cancellation/deadline/error handling; Host/Origin protections and private token bootstrap. |
| 25–26, 33, 53 | Offline device workspace tied to actual node status and controls; research landing with labelled resource simulator; browser layout/keyboard/error checks; strict GGUF validation and a measured F32/Q8_0/Q4_0 test matrix; explicit CPU F32 execution for half-stored checkpoints. |
| 51–52 | Signed publisher/rights/revocation policy; atomic verified cache with quotas and live pins; native `catalog inspect/import/verify/prune/infer`; snapshot-consistent signed package construction; verified install/update/rollback/uninstall with protected paths. |

These groups describe implemented paths. They do not silently complete missing invitation/QR onboarding, cancellable GUI model loading, hosted accounts/credits, portable sandboxing, general transparent recovery or production model/release governance.

## Final automated checks

| Check | Actual result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Passed |
| `cargo test --workspace --locked` | **290 passed, 0 failed, 1 explicitly ignored live-Hub test** |
| `cargo build --workspace --locked` | Passed |
| `cargo build --locked -p synapse-core --examples` | Passed |
| Model/protocol schema, source shape and wire checks | Passed; 37 invalid fixtures rejected |
| Catalogue contract/signature check | Passed; exact canonical bytes, independent OpenSSL verification and 8 invalid fixtures |
| Repository security tool regressions | 7 passed |
| Installer/package/update/rollback tests | 26 passed |
| Benchmark reporter regressions | 5 passed |
| Landing simulator | 8 passed; see the static product check record |
| Live process driver | **20 checks passed**, including deliberately denied hosting rights and the expected worker-loss case |

The 290 Rust tests comprise 239 core unit tests, 32 core integration tests and 19 GUI tests. The optional Hub download test stays outside the default offline suite. A real pinned checkpoint was separately materialized and exercised below. Re-running the five quantization tests with output enabled recorded actual numerical/memory values; those tests are already included in the 290 total and are not counted twice.

Authoritative records: [Cargo commands/results](../../artifacts/final-checks/cargo-final-3.json), [Cargo test log](../../artifacts/final-checks/cargo-final-3-test.log), [auxiliary checks](../../artifacts/final-checks/tools-final.json), [live driver results](../../artifacts/final-processes/checks.json). The driver retains exact argv, exit code, duration and per-case output. Runtime durations are observations from this environment, not latency guarantees.

## Real inference and constrained placement

All runs used distinct temporary identities and actual encrypted process-to-process links. Each harness run first executed the full local backend, then compared every emitted distributed token ID and position. It also checked live route coverage, worker residency and cleanup.

| Run | Coordinator then worker layer ranges | Accepted tokens | Result |
| --- | --- | --- | --- |
| `tiny-2-process-quic` | [0,1) + [1,2) | 8 | Exact full-model token parity |
| `tiny-3-process-quic` | [0,0) + [0,1) + [1,2) | 8 | Exact full-model token parity |
| `tiny-2-process-tcp` | [0,1) + [1,2) | 8 | Exact full-model token parity |
| `tiny-3-process-tcp` | [0,0) + [0,1) + [1,2) | 8 | Exact full-model token parity |
| `trained-2-process-quic` | [0,15) + [15,30) | 8 | Exact full-model token parity |
| `trained-3-process-quic` | [0,0) + [0,15) + [15,30) | 8 | Exact full-model token parity |

The original two-layer fixture uses three input tokens and emits `[45, 20, 61, 20, 46, 8, 60, 34]`. Its weights are original random test data and have no language-quality meaning. Two-process fixture budgets were 52,000 bytes on the host and 45,000 on the worker; three-process budgets were 16,000 on the coordinator and 45,000 on each worker. The full-model reservation at context 11 is 60,608 bytes. Each one-layer worker reported 11,392 resident tensor bytes and a 40,256-byte reservation.

### Trained checkpoint

The concrete trained checkpoint was `HuggingFaceTB/SmolLM2-135M-Instruct`, immutable revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, complete manifest digest `4e44657e25c6f41421f24b9e92756acaaf08834dad8d187494385b65e21543cb`. The checkpoint stores BF16 tensors; this CPU backend executes them as F32. Three verified artifacts were materialized locally. Their weights are not included in the source or handoff.

The committed ChatML prompt file asks for the capital of France in one short sentence. It encodes to 22 input tokens. Both the separate `candle-transformers 0.8.4` Llama implementation and Synapse produced eight greedy tokens, including EOS: `[504, 3575, 282, 4649, 314, 7042, 30, 2]`. The decoded answer was **“The capital of France is Paris.”** Maximum absolute logit difference on the same prefixes was **5.19752502441e-05**, within the declared F32 tolerance of 0.005. This is one exact numerical/sample comparison, not a general quality evaluation.

The same checkpoint passed actual two-process and coordinator-plus-two-worker QUIC inference. At context 32, its full reservation is **765,603,328 bytes**, including **652,354,816 resident tensor bytes**. Two-process limits were 560,000,000 host bytes and 235,000,000 worker bytes, forcing a 15/15 layer split. The selected host reservation was 553,197,568 bytes and worker reservation 227,614,720 bytes. In the three-process run, a 350,000,000-byte coordinator held zero transformer layers; each 235,000,000-byte worker held exactly 15.

**These are inference reservations, not process RSS caps.** Sampled Linux VmHWM includes other runtime allocations and is recorded as an observed lower bound because short processes can exit between samples. It must not be substituted for a final process peak. All processes ran on one physical machine, sharing its OS and page cache. The experiments demonstrate actual bounded placement and numerical parity; they do not establish a physical two-device RAM benefit or internet performance.

See [trained independent-reference output](../../artifacts/final-processes/trained-independent-reference.stdout.log), [trained two-process evidence](../../artifacts/final-processes/trained-2-process-quic/run.json), [trained three-process evidence](../../artifacts/final-processes/trained-3-process-quic/run.json), and [raw benchmark JSON](../../artifacts/final-processes/benchmarks.json)/[CSV](../../artifacts/final-processes/benchmarks.csv). Mixed-model aggregate percentiles are descriptive only; inspect the per-run records before comparison.

## Quantized-format evidence

The generated quantization fixture is separate from the two-layer process fixture and the trained checkpoint. It compares the same original synthetic weights through the supported formats.

| Format | Resident tensor bytes | Maximum absolute logit error versus original F32 |
| --- | ---: | ---: |
| GGUF F32 | 333,056 | Reference |
| GGUF Q8_0 | 104,448 | 0.01690802 |
| GGUF Q4_0 | 65,536 | 0.18867838 |

All five tests passed: exact full/sliced parity within each format across prefill and eight decoding steps; F32 GGUF/safetensors equivalence with nondefault F32 metadata; tied-head alias behavior; bounded quantization error; malformed metadata, extents, shapes, ranges, sidecars and truncation rejection. The header preflight runs before the upstream loader can allocate from untrusted sizes. [Captured numerical output](../../artifacts/final-checks/quantized-metrics.log).

No trained Q4/Q8 language-quality, accelerator, llama.cpp comparison or distributed physical-memory claim follows from this synthetic matrix.

## Network failure and relay evidence

The final local impairment matrix passed four cases: TCP baseline, 20 ms delay with 5 ms jitter, a 65,536-byte/s proxy with 10 ms delay, and termination of a worker after accepted output began. The injected loss case is intentionally an unsuccessful inference: it ended with an explicit error, contiguous partial token positions and a bounded exit. It was not rewritten as a successful completion.

The proxy operates on loopback TCP chunks. It does not simulate real IP packet loss, NAT, CGNAT, geography or Wi-Fi. Local libp2p integration tests independently cover encrypted direct and relayed links. A one-reservation relay rejects the second distinct peer with an actual resource-limit error while preserving the first connection/listener. See [matrix](../../artifacts/final-processes/transport-matrix/matrix.json), [retained worker-loss record](../../artifacts/final-processes/transport-matrix/worker_disconnect/run.json) and [private transport evidence](private-transport.md).

## Application, API and catalogue checks

The actual GUI smoke launched two independent GUI processes and verified unauthenticated denial, wrong Host/Origin denial, query-token rejection, explicit identity paths, preservation of old configuration after invalid edits and rejection of inactive worker controls. Nonstream and SSE output/usage matched exactly. The WebSocket distributed session matched all fixture token IDs and positions **and its assembled text matched the local API**. Actual worker execution reported prefill plus seven decode completions. Pause, resume, drain and stop were checked against acknowledged idle-worker state; active drain semantics have separate unit coverage. Both processes exited and no credential appeared in their logs. [Live GUI result](../../artifacts/final-processes/gui-smoke/summary.json).

The real API contention run attempted 20 requests at concurrency 4: **1 completed, 19 received HTTP 429, 0 other failures**, with 8 actual completion tokens. This verifies the single-admission behavior and honest accounting of rejected work. One successful latency sample is not a useful throughput/SLO benchmark. [API request record](../../artifacts/final-processes/api-load.json).

The native catalogue CLI created a private original-fixture test publisher, inspected its signature/rights, imported and verified **31,596 unique artifact bytes**, rejected remote-hosting rights, and generated the exact eight independent fixture IDs through `catalog infer`. The pin remained alive through loading, generation and unloading. Afterward, pruning reached zero bytes. Test identities and policy directories were removed; no production publisher was enrolled. Existing catalogue tests additionally prove independent active pins block pruning, competing writers fail closed, corrupt content is rejected and revocations are reevaluated. [Driver catalogue outcome](../../artifacts/final-processes/checks.json).

Browser QA is separately scoped in [device workspace QA](device-layout-qa.md) and [landing QA](landing-qa.md). It covers rendered mobile/tablet widths, keyboard controls, error presentation and labelled simulation state. The workspace layout fixture rejected mutations; it was not an authenticated backend. Actual backend behavior is established by the process checks above. Native screen readers, physical touch devices and an independent tester remain unverified. Preview processes were stopped after QA.

## Gaps discovered and corrected during integration

1. CPU BF16 matrix multiplication was unsupported for the trained checkpoint. Execution now explicitly converts half-stored CPU checkpoints to F32, and memory admission uses the execution dtype plus the larger source conversion allocation where needed.
2. GGUF F32 numerical metadata was being treated as missing. Strict typed parsing now preserves epsilon/RoPE values, validates sidecar agreement, and rejects malformed or unsupported architecture semantics before tensor loading.
3. Speculative verification could under-account logits/context memory. Its verification width and remaining context are bounded, with matching reservations and regressions.
4. Distributed text decoded tokens independently, breaking whitespace and multi-token UTF-8. Ordinary and accepted speculative tokens now share bounded upstream stream state. Five regressions cover multibyte boundaries, turn isolation, special/unfinished tokens and real decoder errors; fabricated token placeholders were removed.
5. A full parallel test run exposed a catalogue lock-lifetime failure. Duplicate or fork-inherited descriptors could retain a flock after a bare File was dropped. Owner-process RAII guards now unlock at the intended scope; actual fork and duplicate-descriptor regressions preserve independent live pins and genuine contention refusal. [Detailed diagnosis](catalog-lock-lifetime.md).
6. Installer ancestors and managed children needed stronger protection from replacement. Ownership, real-directory and write-permission checks now cover each relevant path component, including safe sticky-parent handling.
7. Package construction hashed files and later reopened them, allowing concurrent builds to create inconsistent signed archives. It now hashes and archives the same private snapshots; the signed-install regression mutates original inputs after snapshotting and verifies the original bytes.
8. Admin credential/config creation, SSE terminal failures, disabled-discovery behavior and relay exhaustion received targeted fixes and tests during integration.

The failed `cargo-final-2` record remains in the handoff: 230 core tests passed and the catalogue lock test failed. It was diagnosed and fixed, not ignored or hidden. The final `cargo-final-3` run passed all 290 tests. Earlier process attempts and initial evidence remain separate from the authoritative final runs.

## Remaining work and publication boundary

The 22 roadmap packages outside this tranche are PR-22–24, 27–32, 34–38, 46–50 and 54–56. They principally cover hosted membership/accounts, transactional credits/settlement and audits, community distribution/compute, operating budgets and pilot evidence, broader platform/research work, policies and actual releases.

Within the prepared tranche, remaining acceptance/integration includes physical multi-device and WAN experiments; signed-invitation/QR application onboarding; cancellable GUI model startup; general transparent recovery with an approved replay-history policy; target GPU/platform qualification; independent security/usability review; third-party rights approval; resolution of root MIT versus Cargo Apache-2.0 metadata by the authorized owner; and production signing-key/release ownership. Community economics and public untrusted pools remain disabled.

The handoff's 34 local review refs divide chronological history into reviewable slices. Draft bodies describe each plan package's integrated behavior and identify cross-cutting later changes. **Each intermediate boundary has not been independently built or accepted.** `stack.json` lists exact bases, heads, included commits and all later package-tagged follow-ups. Publication should create draft review objects and let their actual CI/review results speak for those boundaries. The source publication requires explicit user authorization because automatic approval review rejected the earlier upload attempt.
