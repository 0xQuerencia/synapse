# Contributing

Start with the [master plan](docs/MASTER_PLAN.md) and [evidence status](docs/IMPLEMENTATION_STATUS.md). Keep changes reviewable and map each to its work-package ID, actual dependencies and acceptance criteria. Avoid moving crates solely for organization.

Every change should explain the concrete problem, behavior, relevant trust/resource impact, provenance, executed commands and known limits. A failing regression test is useful for a real bug; do not add tests that only restate implementation details. Preserve unrelated work and do not remove checks to claim a passing build.

Use `cargo fmt --all -- --check`, locked build/test and applicable Clippy checks for Rust changes. Network model integration belongs in a separate lane from offline fixtures. Website and release tooling changes need their focused checks. Report exact output and environment; do not mark unexecuted tests as passed.

For memory-constrained Linux development, use the [opt-in Cargo runner and runtime profiler](docs/development-memory.md).

Do not import code or redistribute model artifacts until their exact provenance and license are recorded. The root MIT/workspace Apache metadata discrepancy remains an owner decision. Never commit secrets, real prompts, private peer information or access tokens.

A commit named `[PR-XX] ...` identifies a plan work package, not an actual pull request number. Include migrations/rollback where relevant. Independent hardware and security acceptance cannot be replaced by a passing unit test.
