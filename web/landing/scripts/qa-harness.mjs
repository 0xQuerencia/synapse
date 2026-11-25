import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
// Only the development server creates this file. The production build does not.
export async function writeQaHarness() {
  const target = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../dist/_qa.html');
  await writeFile(target, `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synapse local viewport QA</title><style>body{margin:20px;background:#24242b;color:white;font:16px system-ui}main{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}iframe{display:block;border:1px solid #888;background:#0d0f14}h1{font-size:22px}h2{font-size:16px}p{max-width:850px}</style><h1>Local viewport QA</h1><p>Real iframe viewports exercise responsive layout at 360 and 768 CSS pixels. This is layout QA, not a claim of testing physical phones, touch behavior or platform accessibility software.</p><main><section><h2>Mobile layout · 360 CSS px</h2><iframe id="mobile" title="Mobile layout at 360 pixels" src="./" width="360" height="780"></iframe></section><section><h2>Tablet layout · 768 CSS px</h2><iframe id="tablet" title="Tablet layout at 768 pixels" src="./" width="768" height="780"></iframe></section><section><h2>No JavaScript · 360 CSS px</h2><iframe id="nojs" title="No JavaScript fallback at 360 pixels" src="./" width="360" height="780" sandbox="allow-same-origin"></iframe></section></main></html>`);
}
