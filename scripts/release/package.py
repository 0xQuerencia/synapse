#!/usr/bin/env python3
"""Build a bounded, reproducible candidate package; never publish it."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path
import tarfile
import tempfile

from package_format import canonical, digest, regular_file, sign_statement, validate_manifest, validate_statement, version_key


def make_package(binary, version, target, output_dir, signing_key=None, sbom=None, channel='development'):
    version_key(version)
    if not re.fullmatch(r'[a-z0-9_-]{1,80}', target):
        raise ValueError('invalid package target')
    if channel == 'candidate' and signing_key is None:
        raise ValueError('candidate requires a signing key')
    regular_file(binary)
    root = Path(__file__).resolve().parents[2]
    sources = [('bin/synapse', Path(binary), 0o755), ('LICENSE', root / 'LICENSE', 0o644)]
    if sbom:
        sources.append(('sbom.cdx.json', Path(sbom), 0o644))
    files = []
    for name, path, mode in sources:
        info = regular_file(path)
        files.append({'path': name, 'size_bytes': info.st_size, 'sha256': digest(path), 'mode': mode})
    manifest = {'schema_version': 1, 'version': version, 'target': target, 'channel': channel, 'files': files}
    validate_manifest(manifest)
    raw_manifest = canonical(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'synapse-{version}-{target}.tar.gz'
    destination = output_dir / filename
    statement = output_dir / (filename + '.statement.json')
    signature = output_dir / (filename + '.sig')
    if any(os.path.lexists(path) for path in [destination, statement, signature]):
        raise ValueError('refusing to overwrite an existing package')
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix='.package-', delete=False) as temporary:
        temp = Path(temporary.name)
    try:
        with temp.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                    def add(name, content, size, mode):
                        info = tarfile.TarInfo(name)
                        info.size, info.mode, info.mtime = size, mode, 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ''
                        archive.addfile(info, content)
                    add('manifest.json', io.BytesIO(raw_manifest), len(raw_manifest), 0o644)
                    for name, path, mode in sources:
                        with path.open('rb') as content:
                            add(name, content, path.stat().st_size, mode)
            raw.flush()
            os.fsync(raw.fileno())
        record = {'schema_version': 1, 'filename': filename, 'sha256': digest(temp), 'size_bytes': temp.stat().st_size, 'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(), 'version': version, 'target': target, 'channel': channel}
        validate_statement(record)
        os.link(temp, destination)
        destination.chmod(0o644)
        statement_fd = os.open(statement, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(statement_fd, 'wb') as stream:
            stream.write(canonical(record))
            stream.flush()
            os.fsync(stream.fileno())
        if signing_key:
            with tempfile.TemporaryDirectory(dir=output_dir, prefix='.signature-') as signature_dir:
                temporary_signature = Path(signature_dir) / 'package.sig'
                sign_statement(statement, signing_key, temporary_signature)
                os.link(temporary_signature, signature)
        return destination, statement, signature if signing_key else None
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--signing-key', type=Path, help='Existing private Ed25519 PEM key; never generated or published by this command')
    parser.add_argument('--sbom', type=Path)
    parser.add_argument('--channel', choices=['development', 'candidate'], default='development')
    args = parser.parse_args()
    if args.channel == 'candidate' and args.signing_key is None:
        parser.error('candidate packages require a signing key; release approval is separate')
    try:
        paths = make_package(args.binary, args.version, args.target, args.output_dir, args.signing_key, args.sbom, args.channel)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Package failed: {error}\n')
    for path in paths:
        if path:
            print(path)
    print('Local package only. This command does not establish release acceptance or publish it.')


if __name__ == '__main__':
    main()
