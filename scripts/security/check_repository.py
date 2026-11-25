#!/usr/bin/env python3
"""Small offline checks for accidental credentials and risky workflow settings.

This is a narrow repository hygiene gate, not a vulnerability audit. It does not
inspect history, prove absence of secrets, or replace dependency/adversarial review.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess
import sys

PATTERNS = {
    'private key material': re.compile(rb'-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----'),
    'GitHub token': re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,255}'),
    'AWS access key identifier': re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
    'OpenAI-style secret': re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b'),
}


def inspect_content(path: str, content: bytes) -> list[str]:
    if b'\x00' in content[:8192]:
        return []
    failures = [f'{path}: possible {label}; inspect privately' for label, pattern in PATTERNS.items() if pattern.search(content)]
    if path.startswith('.github/workflows/') and path.endswith(('.yml', '.yaml')):
        text = content.decode('utf-8', errors='replace')
        if re.search(r'^\s*pull_request_target\s*:', text, re.M):
            failures.append(f'{path}: pull_request_target requires a separate explicit security review')
        if re.search(r'permissions:\s*write-all\b', text):
            failures.append(f'{path}: blanket write-all workflow permission')
        if re.search(r'curl\b[^\n]*\|\s*(?:bash|sh)\b', text):
            failures.append(f'{path}: unverified remote script execution')
    return failures


def scan(root: Path) -> list[str]:
    result = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], check=True, capture_output=True)
    failures = []
    for raw in sorted(set(result.stdout.split(b'\0')) - {b''}):
        name = raw.decode('utf-8')
        path = root / name
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if Path(name).name == '.env' or (Path(name).suffix in {'.pem', '.p12', '.pfx'} and 'fixture' not in name):
            failures.append(f'{name}: credential-like filename must not be tracked')
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        failures.extend(inspect_content(name, path.read_bytes()))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    failures = scan(args.root)
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print('Repository hygiene checks passed (working tree; no history audit).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
