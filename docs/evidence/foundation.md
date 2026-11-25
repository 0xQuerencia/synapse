# PR-01 foundation evidence

## Reviewed source

Repository `0xQuerencia/synapse`, main at
`1785da5b5f1f87ac6ec008a022ebd68ee22be7cc`. No open pull requests were returned
at intake. The recorded GitHub Actions run `32949751902` reports failed Build,
Clippy and Tests jobs and successful Rustfmt. Its job logs return HTTP 410,
so exact historical diagnostic output is unavailable.

## Local reproduction and repair

Linux x86_64, Rust 1.94.0 (`4a4ef493e`), Cargo 1.94.0 (`85eff7c80`).
`cargo build --workspace --locked` completed in 3m57s but emitted unused-import
and dead-code warnings; CI treats these as errors. The unused registry method
was not connected to heartbeat processing. Removed the unused production
helper and redundant calibration peer field; test-only calibration inspectors
remain compiled and exercised by unit tests.

`cargo clippy --workspace --all-targets --locked -- -D warnings` reproduced
four additional errors: a malformed doc-list continuation, manual repeat/take,
an approximate PI test constant, and a zero-multiplied test index. Corrected
those expressions without weakening any warning or test gate. The same Clippy
command then passed. `cargo fmt --all -- --check` passed before and after repairs.

The toolchain is pinned, Cargo operations use the lockfile, workflows have
bounded run times and read-only contents permission, and obsolete runs cancel
per ref. `scripts/verify_all.sh` now exits on failure instead of printing FAIL
and returning success. CI also runs for stacked PR bases.

## Evidence limits

This package establishes local build/lint repairs. Test-suite results and
GitHub check URLs are recorded in the final implementation evidence after
integration. Model-network tests are migrated to deterministic original
fixtures in PR-40; that is separate from pretending a network download passed.
No physical-device, WAN, performance, production or release acceptance is
implied by this foundation repair.
