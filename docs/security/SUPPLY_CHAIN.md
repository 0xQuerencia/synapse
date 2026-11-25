# Supply chain and release integrity

## Automated checks

`security.yml` runs with read-only repository permission and no persisted checkout credentials. It does not use `pull_request_target`, deployment secrets or privileged runtime access. Actions are pinned to commit IDs; updates require review. Contract validation pins its direct validator version. Runtime builds use the checked-in lockfile and toolchain established by the foundation work.

```sh
python3 scripts/security/check_repository.py
python3 -m unittest discover -s scripts/security -p 'test_*.py' -v
python3 scripts/release/generate_sbom.py --output /tmp/synapse-sbom.cdx.json
```

The narrow hygiene check flags selected credential patterns and dangerous workflow settings in tracked/new files without printing the matched secret. It skips binary/large contents and does not scan history. Treat a passed check as a useful guard, not a claim that the repository has no secrets.

The SBOM generator consumes the full locked Cargo resolution graph, fails on missing components, preserves license expressions and marks unknown licenses for review. It never substitutes `--no-deps` for a complete inventory. The generated CycloneDX inventory is a build artifact; review it alongside a current advisory scan and dependency provenance. No advisory service result is invented by this workflow.

## Reviewer duties

Review lockfile and action-pin changes, known advisories, build scripts, model/parser dependencies, transitive licenses and unexpected network access. Never provide secrets to untrusted PR code. Signing keys belong to release infrastructure outside this repository and require an identified owner, rotation/revocation procedure and separate authorization. Test fixtures must contain synthetic credentials only.

## Publishing and installing

A checksum detects changes only when obtained from a trusted source. A detached signature is useful only with an independently trusted public key. Require both package digest verification and signature verification for release installation; local development packages need an explicit untrusted-development opt-in. Never make `curl | sh` the default installation path.

The release tooling must stage and inspect archives before use, reject traversal/symlinks/special files, install only in a user-owned prefix, preserve previous versions for rollback, and avoid background worker startup. Updates require the worker to be stopped or drained. Uninstall removes only files owned by the installer; model/config/key deletion is a separate explicit choice.

A successful local packaging test does not authorize a production release, create a trusted signing key, settle the license discrepancy or establish platform support. Publish an artifact only after the relevant go/no-go checks and independent installation evidence.
