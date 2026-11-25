# Local draft PR descriptions

This index contains **34 substantive local draft descriptions** for the prepared Synapse implementation tranche. They use the required template in [the master plan](../MASTER_PLAN.md) and refer to planned local review refs `review/pr-XX`. **No GitHub PR is opened, merged or accepted by these files.**

## How to review the stack

The integration owner creates chronological local review boundaries and records their actual commits in the stack manifest. Numerical package order in this index is for navigation; it is not a claim about dependency or merge order. Cross-cutting follow-ups can appear in later stack entries. Each draft describes the integrated behavior attributable to its plan package and tells reviewers to inspect the exact boundary diff.

The [final tranche execution record](../evidence/final-tranche.md) owns actual source/binary identifiers, commands, outputs, environment and test totals. The drafts do **not** claim independent testing at every intermediate review boundary. Targeted commands in the reproduction sections are entry points to rerun against an identified checkout; a command appearing in a description is not itself execution evidence.

The [implementation status](../IMPLEMENTATION_STATUS.md), [gap register](../GAPS.md) and [machine-readable gates](../evidence/gates.json) preserve the distinction between local implementation, local tests and acceptance. Physical two-device benefit, actual separate-network WAN, production trust/rights, independent review and released-platform qualification remain open. Third-party model weights and secret values are excluded from these drafts.

Paths shortened to `inference/…` or `network/…` refer to `crates/synapse-core/src`; GUI module names refer to `crates/synapse-gui/src`. Other paths are relative to the repository root. Run commands from that root against an identified integrated checkout.

## Descriptions

| Plan package | Planned local review ref | Scope |
| --- | --- | --- |
| [PR-01](PR-01.md) | `review/pr-01` | Repair the foundation and make verification reproducible |
| [PR-02](PR-02.md) | `review/pr-02` | Publish the approved plan and truthful product status |
| [PR-03](PR-03.md) | `review/pr-03` | Pin model identity and verify every input artifact |
| [PR-04](PR-04.md) | `review/pr-04` | Define supported Llama numerics and prompt behavior |
| [PR-05](PR-05.md) | `review/pr-05` | Require complete feasible layer routes |
| [PR-06](PR-06.md) | `review/pr-06` | Prepare authenticated shard leases before generation |
| [PR-07](PR-07.md) | `review/pr-07` | Load only the assigned transformer and IO tensors |
| [PR-08](PR-08.md) | `review/pr-08` | Exercise actual coordinator and worker processes |
| [PR-09](PR-09.md) | `review/pr-09` | Measure completed inference and retain benchmark evidence |
| [PR-10](PR-10.md) | `review/pr-10` | Admit model-shaped memory and report measured capability |
| [PR-11](PR-11.md) | `review/pr-11` | Advertise signed expiring resident model ranges |
| [PR-12](PR-12.md) | `review/pr-12` | Choose feasible routes from bounded measured candidates |
| [PR-13](PR-13.md) | `review/pr-13` | Bind private pairing and invitation proofs to identities |
| [PR-14](PR-14.md) | `review/pr-14` | Make discovery conservative, pinned and bounded |
| [PR-15](PR-15.md) | `review/pr-15` | Verify encrypted direct and relayed private links |
| [PR-16](PR-16.md) | `review/pr-16` | Correlate bounded transport with the active inference session |
| [PR-17](PR-17.md) | `review/pr-17` | Persist epochs and bound recovery progress |
| [PR-18](PR-18.md) | `review/pr-18` | Retain honest delayed-transport and worker-loss evidence |
| [PR-19](PR-19.md) | `review/pr-19` | Harden private security state and local administration |
| [PR-20](PR-20.md) | `review/pr-20` | Provide a bounded full-model execution backend |
| [PR-21](PR-21.md) | `review/pr-21` | Serve the authenticated compatible chat subset |
| [PR-25](PR-25.md) | `review/pr-25` | Connect device onboarding to real local state |
| [PR-26](PR-26.md) | `review/pr-26` | Validate quantized storage and explicit CPU execution |
| [PR-33](PR-33.md) | `review/pr-33` | Build the research landing and honest resource simulator |
| [PR-39](PR-39.md) | `review/pr-39` | Publish executable versioned runtime contracts |
| [PR-40](PR-40.md) | `review/pr-40` | Use original offline references and retain real-model evidence |
| [PR-41](PR-41.md) | `review/pr-41` | Protect persistent device identity and revocation state |
| [PR-42](PR-42.md) | `review/pr-42` | Apply acknowledged worker controls and native process caps |
| [PR-43](PR-43.md) | `review/pr-43` | Reserve capacity atomically and isolate admitted state |
| [PR-44](PR-44.md) | `review/pr-44` | Bound wire decoding, quotas and relay exhaustion |
| [PR-45](PR-45.md) | `review/pr-45` | Prepare supply-chain provenance and repository safeguards |
| [PR-51](PR-51.md) | `review/pr-51` | Enforce signed model rights and verified cache lifecycle |
| [PR-52](PR-52.md) | `review/pr-52` | Verify local packages and make version switching recoverable |
| [PR-53](PR-53.md) | `review/pr-53` | Verify responsive accessibility and real application flows |

## Publication and reviewer status

These descriptions are prepared for user review. At local handoff, the project reports zero published GitHub PRs, zero release artifacts and zero production deployments for this tranche. A local branch name is not an external publication. All reviewer checkboxes remain unchecked; evidence links and draft text do not supply missing owner or independent approval.

Original source changes and fixture provenance are documented in [the provenance register](../upstream-provenance.md). The existing MIT root license remains unchanged, and the Apache-2.0 package-metadata discrepancy remains an authorized maintainer decision.
