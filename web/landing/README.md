# Synapse research landing page

A buildless HTML/CSS/ES-module site, with no runtime dependencies, remote fonts, analytics, accounts or connection to the worker daemon. All device profiles and timing values in the explorer are explicitly fictional.

```sh
node scripts/build.mjs
node --test scripts/simulator.test.mjs
node scripts/check.mjs
```

Run commands from `web/landing`. The build copies the complete approved plan and related source documents into `dist/docs`; the delivered site is self-contained. `dist` is generated and ignored. The development entrypoint builds once and serves the result; rebuild and reload after edits. In managed environments use the approved preview service rather than starting a second server.

## Simulator contract

Every candidate has exact contiguous layer coverage and satisfies each device's own inference budget. Count per-layer weights, context-dependent KV, scratch and fixed input/output tensors. For sharding the laptop is the coordinator and owns the input/output weights even when assigned zero layers. Full-model routes count all weights/state on their chosen host.

The cost model enumerates bounded candidate routes and chooses the lowest fictional decode estimate with deterministic ties. Sharded remote stages add activation round-trip costs. A full-model host streams text without that per-token activation chain. Estimates exclude load, queue and prefill and cannot be presented as measurements. Local and remote examples assume compatible formats/backends and trusted eligible devices.

Tests cover coverage, budgets, fixed-weight overhead, context cost, no-route states, full-model selection and latency sensitivity, including a grid of profile/context/network/budget combinations.

## Status and publishing

The site uses research-stage language, source/roadmap calls to action and no fabricated download, adoption or earnings claims. Production domain/canonical URL, public deployment, model rights, supported-device evidence and signed releases remain separate decisions. This local implementation does not deploy anything.

`dist/_headers` supplies a suggested static-host security policy. A selected host must actually enforce it; placing a file in a bundle is not proof of deployed headers. No production hostname, waitlist or tracking identifier is invented.

## Browser QA

The development server exposes `/_qa.html`, with real 360/768 CSS-pixel iframe viewports and a JavaScript-disabled fallback. It exists only in the local development output; a clean production build excludes it. Development response framing is SAMEORIGIN to permit this harness; the suggested production header remains DENY.

See [the QA report](../../docs/evidence/landing-qa.md) for actual checks and limitations. Semantic checks, selected palette contrast and reduced-motion CSS policy do not replace physical-device or screen-reader review.
