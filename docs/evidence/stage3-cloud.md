# Integrated Stage 3 and cloud qualification

`main` resolved to `a127c8907113862379e597a84ec4cc16ee98bd5a` after GitHub PRs #2–45. The cloud checkout used the exact #51 draft head, `1a5f44b2c5100cc84730e93eb54a1164d875c56f`, from `stage3/operational-readiness`. The stack contains six commits above `main`. GitHub PR numbers and the master plan's work-package IDs are separate.

## Exact draft evidence

GitHub reported every check in the rollup as `SUCCESS` on the following draft heads. The named assertions come from the test files and workflow commands in that checkout. PostgreSQL and the process evidence are recorded as remote runs when cited here.

| Draft head | Checks relevant to roadmap scope |
| --- | --- |
| [#46](https://github.com/0xQuerencia/synapse/pull/46) `d382116bd2e135477ed36307903bb037b270941c` | `community/tests/test_postgres.py` exercises isolated PostgreSQL, concurrent reservation and settlement, capacity, refund/reconciliation, rollback and pool isolation. The `postgres-testnet.yml` workflow supplies a disposable PostgreSQL 16 service and restricted schema. |
| [#47](https://github.com/0xQuerencia/synapse/pull/47) `689de6f5d6069b5d44c08535bb108d57953dface` | `test_native_lab.py` covers settle-once, mismatch refund, verifier separation and unapproved models. The process workflow invokes `scripts/native_accounting_lab.py` against the actual GUI binary and original fixture with fake credits. |
| [#48](https://github.com/0xQuerencia/synapse/pull/48) `6dcb032ce9243f8914d170129e2e9975386aac97` | `test_artifacts.py` and `test_fixture_transfer.py` cover pins and shared chunks, deterministic eviction, bounds, rights/revocation, atomic export and the original fixture transfer over authenticated loopback. |
| [#49](https://github.com/0xQuerencia/synapse/pull/49) `82e46cc26b8119451982e7de96893809182978de` | `test_enrollment.py` covers recipient-bound redemption, consent, concurrent single use, expiry/revocation, authorization and atomic credential rotation. This qualifies testnet membership enrollment. Native peer invitation and QR workflows remain open. |
| [#50](https://github.com/0xQuerencia/synapse/pull/50) `359f1fbe6e8c4c89c0e7bffa6b9b5522f581a740` | `test_sharded_lab.py` checks route order, distinct peers, model binding and fake-credit conservation. The integrated process workflow runs `scripts/sharded_accounting_lab.py` with the actual CLI over TCP and QUIC and interrupts a TCP worker to check the refund. |
| [#51](https://github.com/0xQuerencia/synapse/pull/51) `1a5f44b2c5100cc84730e93eb54a1164d875c56f` | `test_outbox.py` covers duplicate-safe same-sink export, concurrency and scope. `test_contributor.py` covers opt-in ranking and private idempotent disputes. The integrated PostgreSQL suite adds outbox and contributor privacy coverage. GitHub reported 11 successful checks at this head, including both the PostgreSQL and process workflows. |

## Remaining roadmap

| Work package or gate | Evidence and closure still needed |
| --- | --- |
| G0 and package 26 | The integrated tests and original fixture establish deterministic local behavior. Approved useful models need exact revisions, terms, parity and hardware results. The owner still needs to settle the MIT/Apache metadata discrepancy. |
| G1 and packages 05–10, 42–43 | Run a consenting two-physical-device trial. Record each assigned/resident range, actual per-device peak memory and usable capacity, successful reference output with the model beyond either configured individual budget, and cleanup. One-host processes establish mechanics. |
| G2 and packages 13–19, 44 | Run authenticated direct and relay inference across separate routers with consenting peers. Exercise reachable and constrained UDP/CGNAT/firewall paths, loss, cancellation and cleanup. Qualify native invitation and recovery UX separately. |
| G3 and packages 22–24, 27–29, 32, 46–50, 54–55 | The disposable service exercises enrollment, synthetic accounting, local known-answer verification, reconciliation and contributor views. An approved pilot still needs hosted identity/operations, signed worker releases, independent receipt and abuse review, approved pools and models, privacy terms and consent, restore/incident drills, a budget and accountable operators. |
| G4 and packages 17, 31, 49 | Native loopback work establishes exact synthetic sharded metering and interrupted-work refunds. Real multi-host recovery and separate accounting, privacy and independent security acceptance remain open. |
| G5 and packages 25, 30, 33–34, 51–53, 56 | Package preflight covers host-target integrity and exact build identity in this checkout. Release still requires reviewer-approved provenance/SBOM, publisher key ownership and independently authenticated distribution, rotation/revocation, approved model redistribution rights, independent clean installation/update/rollback/uninstall, supported-target and accessibility QA, and release sign-off. |
| G6 and packages 35–38 | Public network acceptance requires independent privacy, integrity, abuse and economic review, validated resource controls and explicit launch approval. Broader hardware, browser workers and open swarm remain research. |

## Cloud verification

The existing installer activation tests were attempted before the release change. This environment reports UID 65534 for `/`, so the installer refused that ancestor even when `TMPDIR` was moved to an agent-owned workspace. A network-disabled Debian 13 container in the same cloud workspace supplied a normal root mapping and passed all 29 installer tests, including activation, update, rollback and uninstall with a synthetic native payload. The added artifact preflight exercises the package signature, target, snapshot digest and manifest-bound payload through `install.py verify-artifact` without activation. Two focused regressions passed for intact and tampered archives, an invalid signature and absence of an install prefix.

The existing community verification completed on the exact draft source with 94 tests discovered, 8 PostgreSQL tests skipped without an isolated database, and the synthetic demos and release gate completed. The static landing built, all eight simulator tests passed, and the semantic and acceptance-gate checks passed. The seven acceptance gates remain open.

Run the installer suite under a normally mapped Linux root, or reproduce the isolated cloud run with the exact image used here.

```sh
docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,exec \
  --mount type=bind,src="$PWD",dst=/repo,readonly \
  -e PYTHONDONTWRITEBYTECODE=1 -w /repo \
  python@sha256:9dd21fafb95870b99f9faa703a947d08feeb673048ad556e92ce503d0c086a95 \
  python3 -m unittest discover -s scripts/release -p 'test_*.py' -v
```

## Build memory and cancellation

Codex Cloud ran Debian 13 x86_64, Linux 6.18.57 and Rust 1.94.0 with a 16 GiB cgroup memory ceiling. The table reports KiB. The cold run includes dependency fetching and compilation. The warm run used the same build target and cached dependencies. All runs reported no surviving owned process-group members. The cgroup reported zero OOM kills.

| Run | Elapsed, seconds | Sampled process-group RSS peak | `wait4` maximum RSS | Result |
| --- | ---: | ---: | ---: | --- |
| Cold scoped CLI, GUI and catalog build | 1120.400 | 1,288,352 | 1,039,044 | Passed |
| Same build, warm and offline | 0.727 | 122,020 | 122,156 | Passed |
| Rebuild of the CLI and GUI after the ownership fix | 169.130 | 1,064,800 | 948,696 | Passed |
| A live core-only test build interrupted during compilation by SIGTERM | 32.138 | 525,308 | 405,512 | Cargo status −15 and no surviving group members |

```sh
python3 scripts/low_memory.py build -p synapse-cli -p synapse-gui -p synapse-core \
  --example catalog --bin synapse --bin synapse-gui
python3 scripts/low_memory.py build --offline -p synapse-cli -p synapse-gui -p synapse-core \
  --example catalog --bin synapse --bin synapse-gui
python3 -m unittest discover -s scripts -p 'test_low_memory.py' -v
```

The core-only build changed dependency feature resolution and began recompiling previously built dependencies. Selecting the workspace with `--lib` kept the tested target limited to the core library while reusing the build's feature resolution. RSS measurements describe the one-job configuration on this host. The aggregate samples can double-count shared pages and miss brief peaks.

## Runtime ownership and resident memory

The new core regression uses an isolated Tokio runtime and disposable loopback listeners. The exact #51 runtime code with only the regression added retained 2, 4, 6 and 8 tasks after four aborted node sessions. Each aborted node had left its two signal-listener tasks waiting on process signals. The fix keeps both subscriptions inside the node's run future, so termination drops them along with the rest of the session. All 30 focused network-node tests passed, including the regression, which returned to zero retained tasks after each abort.

```sh
python3 scripts/low_memory.py test --offline --workspace --lib \
  network::node::lifecycle::tests::repeated_aborted_sessions_release_runtime_tasks -- --exact
python3 scripts/runtime_memory.py --binary target/low-memory/debug/synapse-gui --cycles 32
python3 scripts/release/smoke_cli.py --binary target/low-memory/debug/synapse
```

The GUI workload uses the original `fixture-v1` manifest with SHA-256 `1d9796ab2f4fdb6549a96b3c75835347f69e351c0b961169785592b6d57893b0`. Each cycle starts a loopback host, checks bounded JSON and SSE inference, stops the host, and waits for idle before reading `smaps_rollup`. This exercises the real GUI server and a tiny local CPU workload.

| GUI runtime | `VmHWM`, KiB | Idle RSS at readiness / cycle 1 / cycle 32, KiB | Idle anonymous memory at cycle 1 / cycle 32, KiB |
| --- | ---: | ---: | ---: |
| Exact #51 runtime | 54,480 | 17,540 / 27,768 / 46,716 | 6,428 / 25,368 |
| With the signal ownership fix | 52,344 | 16,832 / 27,328 / 46,892 | 6,544 / 26,104 |

Both runs exited cleanly with zero surviving process-group members. Every post-stop sample had 10 open descriptors and 11 threads. In cycles 9–32 the baseline idle RSS ranged from 46,396 to 46,716 KiB and the fixed idle RSS ranged from 41,556 to 48,920 KiB. Most of the RSS increase occurred during early cycles. Later variation and the similar final RSS are consistent with runtime warm-up and allocator retention, but RSS cannot attribute individual allocations. The Tokio task count demonstrated the ownership leak independently. The RSS series does not establish an unbounded heap trend or a measurable RSS saving from the fix.

Both GUI binaries also passed the existing two-process authenticated loopback smoke with the WebSocket requirement. The host routed prefill, the worker recorded seven decode completions, and the output matched all eight reference token IDs. Both GUI processes exited. The fixed GUI binary SHA-256 was `d8afe55e83953935cba2c0a5dd6d7b044cf8f1f79b9cf77c2fe301b1a11e0441`.

The native CLI package smoke verified the signed archive and matched the exact packaged binary digest to the build. That binary produced all eight fixture reference tokens. The local archive SHA-256 was `3e90b8c6812e638f207568a7694d5b40f05f009cb141e6651c103694d909f9c2`. It was generated with a disposable key.

The network-disabled container also installed, updated and rolled back two test versions of that same native payload. The installed executable matched all eight reference tokens at each stage. Uninstall removed the disposable prefix and preserved a separate configuration file.

The full core library suite initially failed 19 catalog tests because the cloud maps `/tmp` to an untrusted UID. All 240 tests passed on rerun with an owned temporary directory. The installer and catalog path guards continued to enforce their ownership checks.

```sh
mkdir -p target/low-memory/tmp
TMPDIR="$PWD/target/low-memory/tmp" python3 scripts/low_memory.py test --offline --workspace --lib
python3 scripts/gui_smoke.py --binary target/low-memory/debug/synapse-gui \
  --require-websocket --out artifacts/stage3-gui-smoke
```
