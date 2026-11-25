import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(site, '../..');
export async function build() {
  const destination = path.join(site, 'dist');
  await rm(destination, { recursive: true, force: true });
  await mkdir(path.join(destination, 'docs'), { recursive: true });
  await cp(path.join(site, 'index.html'), path.join(destination, 'index.html'));
  await cp(path.join(site, 'assets'), path.join(destination, 'assets'), { recursive: true });
  for (const name of ['MASTER_PLAN.md', 'IMPLEMENTATION_STATUS.md', 'quickstart.md', 'upstream-provenance.md']) {
    await cp(path.join(repo, 'docs', name), path.join(destination, 'docs', name));
  }
  await cp(path.join(repo, 'SECURITY.md'), path.join(destination, 'docs', 'SECURITY.md'));
  await cp(path.join(repo, 'LICENSE'), path.join(destination, 'docs', 'LICENSE'));
  await writeFile(path.join(destination, '404.html'), '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Page not found — Synapse</title><h1>Page not found</h1><p><a href="./">Return to Synapse</a></p></html>\n');
  await writeFile(path.join(destination, '_headers'), '/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: no-referrer\n  X-Frame-Options: DENY\n  Permissions-Policy: camera=(), microphone=(), geolocation=()\n  Content-Security-Policy: default-src \'self\'; script-src \'self\'; style-src \'self\'; img-src \'self\' data:; font-src \'self\'; object-src \'none\'; base-uri \'none\'; frame-ancestors \'none\'\n');
  console.log('Built static site and complete planning documents in web/landing/dist.');
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await build();
