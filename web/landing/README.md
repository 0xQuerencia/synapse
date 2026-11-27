# Synapse landing

A dependency-free static site for the experimental layer-splitting runtime. The trace records a deterministic laptop-to-Codex Cloud fixture run over private Tailscale. Each machine ran one layer, one prefill pass and seven decode passes. All eight fixture token IDs matched the laptop-only run.

The memory simulator lives at `simulator.html`. Its device profiles and timing estimates are fictional. It evaluates exact contiguous layer coverage, individual memory budgets, cache and working memory, fixed input and output weights, and estimated network overhead.

## Build and preview

From the repository root:

```sh
npm --prefix web/landing test
npm --prefix web/landing run build
npm --prefix web/landing run check
npm --prefix web/landing run dev
```

The development server serves the landing at `/`, the simulator at `/simulator.html`, and a viewport harness at `/_qa.html`. Set `PORT` to override the default port, 4173. Rebuild and reload after edits.

## Publish

The complete static output is `web/landing/dist`. For a Git-connected Cloudflare Pages project, use the repository root with build command `npm --prefix web/landing run build` and output directory `web/landing/dist`. For an existing Direct Upload project, run the same build locally and upload the contents of that output directory. The build needs Node.js and the full checkout because it copies project documentation.

Keep `index.html`, `404.html` and `_headers` at the publish root. A fresh build excludes the development QA harness. Confirm the landing, simulator, discrete 404 response and security headers on the intended hostname after publishing.

The site bundles Instrument Serif Italic locally. Its [license](assets/fonts/OFL-Instrument-Serif.txt) and [upstream source](https://github.com/google/fonts/tree/main/ofl/instrumentserif) are available alongside the build.
