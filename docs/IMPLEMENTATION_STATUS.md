# Implementation and evidence status

The master plan describes the target product. Source files describe implementation; test logs and benchmark artifacts describe evidence. None substitutes for another.

## Acceptance gates

| Gate | Required evidence | Initial imported status |
| --- | --- | --- |
| G0 local foundation | Reconciled baseline, build/CI, offline fixtures and exact model parity | Verification in progress; see revision-specific execution report |
| G1 real sharding | Two physical machines, exact ranges, no hidden full copy and measured memory savings | Physical evidence not supplied |
| G2 private WAN | Separate routers, authenticated direct/relay runs, limits and review | Physical WAN evidence not supplied |
| G3 community full model | Approved pools/models, consent, durable accounting, verification, budget and pilot review | Disabled pending gate |
| G4 community sharding | Route correctness, bounded recovery and exact stage settlement | Disabled pending gate |
| G5 closed beta | Signed packages, independent installation, support matrix and go/no-go | No release attested by this record |
| G6 open swarm | Independent adversarial, privacy and economic review | Research only |

The reviewed source baseline is `1785da5b5f1f87ac6ec008a022ebd68ee22be7cc`. New execution results must identify their actual revision; never reuse that baseline SHA for a modified checkout.

## Work-package terminology

- **Proposed:** planned scope, no implementation claim.
- **Implemented:** a concrete code/document change exists and has a real integration path.
- **Locally tested:** named tests ran successfully at a stated revision and environment.
- **Awaiting evidence:** physical-device, network, independent review or owner evidence remains.
- **Accepted:** every applicable package criterion has evidence and review.
- **Released:** an identified published artifact satisfies the relevant release gate.

A commit tagged `[PR-XX]` identifies planned scope. It is not a GitHub PR number, a merged review, a release or proof of every criterion. Keep failed and blocked checks visible. The research landing page uses conservative roadmap labels until release evidence is published.

## Before changing a claim

Record the requirement, command/run identifier, environment, model and source revision, actual output, limitations and reviewer. Classify topology as `loopback`, `physical_lan`, `physical_wan` or `simulated_wan`. Unknown performance measurements remain missing, not zero. Publish only redacted synthetic-test artifacts.
