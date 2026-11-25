# Source, model and license provenance

## Existing license discrepancy

At reviewed baseline `1785da5b5f1f87ac6ec008a022ebd68ee22be7cc`, root `LICENSE` contains MIT terms, while `[workspace.package].license` is `Apache-2.0`. The root license is preserved. This work does not relicense prior contributions, resolve ownership or provide a legal conclusion. A maintainer with authority over those contributions must reconcile the discrepancy before release.

Existing dependency licenses and model licenses are independent of that choice. Public download access does not establish redistribution or hosted-use permission. Do not publish an approved catalog until the exact model revisions and rights are recorded.

## Current implementation provenance

The new product documentation, static site, specification checks and release tooling in this workstream are original project changes. No code or brand assets were copied from EXO, Petals, Prima.cpp, Distributed Llama, llama.cpp or another Synapse project. Their architectural ideas are research references in the master plan. No external model weights are bundled by these documents.

## Required register for future imports

Record one entry per imported component: upstream URL, exact commit/revision, imported files, license and notices, modifications, test evidence and reviewing maintainer. Preserve third-party notices. Use the resolved locked dependency graph for an SBOM rather than a hand-written list of selected packages.

For models additionally record artifact hashes, tokenizer/template, supported backend/format, hosting permission, redistribution permission, source access restrictions and approver. `unknown` means excluded from an approved distribution catalog.

## Naming

Synapse is a provisional project name. The existence of other projects with that name is recorded in the plan. This repository does not claim a relationship with those projects or ownership of a domain or trademark.
