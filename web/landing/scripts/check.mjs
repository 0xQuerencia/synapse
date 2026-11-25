import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const checker = path.join(path.dirname(fileURLToPath(import.meta.url)), 'check.py');
const run = spawnSync('python3', [checker], { stdio: 'inherit' });
if (run.error) throw run.error;
process.exit(run.status ?? 1);
