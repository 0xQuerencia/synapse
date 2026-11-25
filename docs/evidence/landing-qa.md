# Research landing QA

Scope: PR-33/53 static landing and fictional resource simulator. This record does not certify the local application, physical devices, production headers or released inference capabilities.

## Executed checks

- `node web/landing/scripts/build.mjs`: complete static build, including the approved master plan and related source documents.
- `node --test web/landing/scripts/simulator.test.mjs`: eight tests passed. Includes a profile/context/network/budget grid, exact coverage, per-device memory, coordinator fixed weights, KV growth, latency sensitivity and full-model routing.
- `node web/landing/scripts/check.mjs`: semantic checks, 12 internal asset/link references, four core color-pair contrast ratios at least 4.5:1, responsive/reduced-motion policy and three deliberately broken semantic regressions passed.
- Browser desktop review: hero, architecture illustration and simulator inspected; sharded/full-model/no-route states observed; checkbox Space interaction worked.
- Browser mobile layout: a real iframe viewport of 360 CSS pixels (345-pixel content width with the browser scrollbar). A discovered heading/min-content overflow was fixed. Final `scrollWidth = clientWidth = 345`; simulator controls switched to a full-model route with no overflow.
- Browser tablet layout: 768 CSS-pixel iframe (753-pixel content width); final `scrollWidth = clientWidth = 753`.
- Mobile/tablet menu: observed `aria-expanded=true` after opening and `false` after Escape. Escape returns focus to the menu control.
- JavaScript-disabled iframe: fallback explanation remained visible, ordinary navigation remained available, and the JavaScript menu stayed hidden. The simulation fieldsets and clipboard button are disabled until their script is ready.

The browser checks exercised real rendered iframe viewports, not physical phones. No touch-device, screen-reader software or operating-system reduced-motion preference emulation was performed. Reduced-motion CSS sets smooth scrolling to auto and suppresses animation/transitions; the site has no continuous decorative animation. Static-host `_headers` enforcement must be verified at deployment time, which this work does not perform.

## Reproduce locally

Build and run the dependency-free development script through the environment's approved preview service. Inspect `/` at desktop width and `/_qa.html` for mobile, tablet and script-disabled views. Exercise Menu, Escape, the GPU checkbox via Space, the model/context/network selects, and individual budgets. Confirm no active requests, accounts or balances exist behind the simulator.

A clean `build.mjs` removes the development QA harness from `dist`. Keep the static page's status claims conservative until corresponding acceptance evidence is published.

## Separate app QA required

The real application additionally needs authentication bootstrap, Host/Origin denial, revoked/invalid credentials, live device/model state, streaming disconnect/cancel cleanup, resource control integration and keyboard/error flows against the actual backend. Landing QA cannot satisfy those tests.
