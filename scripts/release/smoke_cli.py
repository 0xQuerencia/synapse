#!/usr/bin/env python3
"""Build and verify a disposable signed artifact containing the real Linux CLI."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from install import host_target
from package_format import digest

RELEASE = Path(__file__).resolve().parent
FIXTURE = RELEASE.parents[1] / 'crates/synapse-core/tests/fixtures/tiny-llama'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def smoke(binary, root):
    home = root / 'home'
    home.mkdir(mode=0o700)
    environment = os.environ | {'HOME': str(home), 'HF_HOME': str(home / 'hf')}

    def run(*args):
        process = subprocess.run(args, cwd=root, env=environment, capture_output=True, text=True, timeout=120)
        require(process.returncode == 0, f'{Path(args[0]).name} exited {process.returncode}: {process.stderr}')
        return process.stdout

    key, public = root / 'signing.pem', root / 'public.pem'
    run('openssl', 'genpkey', '-algorithm', 'ED25519', '-out', key)
    key.chmod(0o600)
    run('openssl', 'pkey', '-in', key, '-pubout', '-out', public)

    version, target = '0.2.0-smoke.1', host_target()
    output = root / 'package'
    run(sys.executable, RELEASE / 'package.py', '--binary', binary, '--version', version,
        '--target', target, '--output-dir', output, '--channel', 'candidate', '--signing-key', key)
    archive = output / f'synapse-{version}-{target}.tar.gz'
    prefix = root / 'unused-prefix'
    manifest = json.loads(run(sys.executable, RELEASE / 'install.py', '--prefix', prefix,
                              'verify-artifact', archive, '--statement', f'{archive}.statement.json',
                              '--signature', f'{archive}.sig', '--public-key', public))
    require(not prefix.exists(), 'artifact verification created an installation')
    require(manifest['version'] == version and manifest['target'] == target, 'wrong package identity')
    binary_hash = digest(binary)
    packaged = next(item for item in manifest['files'] if item['path'] == 'bin/synapse')
    require(packaged['sha256'] == binary_hash, 'verified package binary differs from the build')

    reference = json.loads((FIXTURE / 'reference.json').read_text())
    expected = reference['greedy_ids'][len(reference['input_ids']):]
    require(len(expected) == 8, 'expected eight reference tokens')
    events = root / 'inference.jsonl'
    run(binary, 'infer', '--model-dir', FIXTURE, '--prompt', '<s> </s> Hello',
        '--max-tokens', str(len(expected)), '--events-jsonl', events)
    records = [json.loads(line) for line in events.read_text().splitlines()]
    tokens = [record['data']['token_id'] for record in records if record['event'] == 'token']
    require(tokens == expected, 'CLI inference differed from the reference')
    require(records[-1]['event'] == 'generation_done', 'generation did not finish')
    return {
        'status': 'passed', 'target': target, 'binary_sha256': binary_hash,
        'fixture_manifest_sha256': digest(FIXTURE / 'manifest.json'),
        'archive_sha256': digest(archive), 'reference_token_count': len(tokens),
        'installation_created': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='synapse-cli-package-smoke-') as directory:
        result = smoke(args.binary.resolve(strict=True), Path(directory))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
