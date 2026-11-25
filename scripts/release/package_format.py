"""Versioned package format shared by candidate builder and local installer."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

MAX_ARCHIVE = 256 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
MAX_DOCUMENT = 64 * 1024
ALLOWED_FILES = {'bin/synapse', 'LICENSE', 'sbom.cdx.json'}
VERSION = re.compile(r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?$')


def version_key(value):
    match = VERSION.fullmatch(value) if isinstance(value, str) and len(value) <= 128 else None
    if not match:
        raise ValueError('version must be a bounded semantic version')
    prerelease = match.group(4)
    parts = []
    for item in prerelease.split('.') if prerelease else []:
        if not item or (item.isdigit() and len(item) > 1 and item.startswith('0')):
            raise ValueError('invalid semantic prerelease')
        parts.append((0, int(item)) if item.isdigit() else (1, item))
    if match.group(5) and any(not item for item in match.group(5).split('.')):
        raise ValueError('invalid build metadata')
    return (*map(int, match.group(1, 2, 3)), not bool(prerelease), tuple(parts))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode() + b'\n'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def regular_file(path, maximum=None):
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('expected a regular file, not a link or special file')
    if maximum is not None and info.st_size > maximum:
        raise ValueError('file exceeds package bound')
    return info


def read_document(path):
    regular_file(path, MAX_DOCUMENT)
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('document must be an object')
    return value, raw


def validate_statement(value):
    keys = {'schema_version', 'filename', 'sha256', 'size_bytes', 'manifest_sha256', 'version', 'target', 'channel'}
    if set(value) != keys or value['schema_version'] != 1:
        raise ValueError('unsupported release statement schema')
    version_key(value['version'])
    if not re.fullmatch(r'[a-z0-9_-]{1,80}', value['target']):
        raise ValueError('invalid target')
    expected = f"synapse-{value['version']}-{value['target']}.tar.gz"
    if value['filename'] != expected:
        raise ValueError('unexpected artifact filename')
    if value['channel'] not in {'development', 'candidate'}:
        raise ValueError('unknown package channel')
    if type(value['size_bytes']) is not int or not 0 < value['size_bytes'] <= MAX_ARCHIVE:
        raise ValueError('invalid archive size')
    for key in ['sha256', 'manifest_sha256']:
        if not isinstance(value[key], str) or not re.fullmatch(r'[0-9a-f]{64}', value[key]):
            raise ValueError('invalid digest')


def validate_manifest(value):
    if set(value) != {'schema_version', 'version', 'target', 'channel', 'files'} or value['schema_version'] != 1:
        raise ValueError('unsupported package manifest schema')
    version_key(value['version'])
    if value['channel'] not in {'development', 'candidate'}:
        raise ValueError('invalid package channel')
    if not isinstance(value['files'], list) or not 2 <= len(value['files']) <= 3:
        raise ValueError('unexpected package file count')
    names = set()
    for item in value['files']:
        if set(item) != {'path', 'size_bytes', 'sha256', 'mode'}:
            raise ValueError('invalid package file schema')
        if item['path'] not in ALLOWED_FILES or item['path'] in names:
            raise ValueError('unsupported or duplicate package path')
        names.add(item['path'])
        if type(item['size_bytes']) is not int or not 0 < item['size_bytes'] <= MAX_ARCHIVE:
            raise ValueError('invalid package file size')
        if not isinstance(item['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
            raise ValueError('invalid package file digest')
        if item['mode'] != (0o755 if item['path'] == 'bin/synapse' else 0o644):
            raise ValueError('unsafe package permission')
    if not {'bin/synapse', 'LICENSE'}.issubset(names):
        raise ValueError('binary and license are required')


def openssl(arguments):
    result = subprocess.run(['openssl', *arguments], capture_output=True)
    if result.returncode:
        raise ValueError('OpenSSL signature operation failed')
    return result.stdout


def sign_statement(statement, private_key, signature):
    info = regular_file(private_key)
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError('signing key must be private and owned by the current user')
    public_der = openssl(['pkey', '-in', str(private_key), '-pubout', '-outform', 'DER'])
    if len(public_der) != 44 or not public_der.startswith(bytes.fromhex('302a300506032b6570032100')):
        raise ValueError('signing key must be Ed25519')
    openssl(['pkeyutl', '-sign', '-rawin', '-inkey', str(private_key), '-in', str(statement), '-out', str(signature)])


def verify_signature(statement, signature, public_key):
    if regular_file(signature, 1024).st_size != 64:
        raise ValueError('expected a 64-byte Ed25519 signature')
    if regular_file(public_key, MAX_DOCUMENT).st_mode & 0o022:
        raise ValueError('trusted public key must not be group/world writable')
    public_der = openssl(['pkey', '-pubin', '-in', str(public_key), '-outform', 'DER'])
    if len(public_der) != 44 or not public_der.startswith(bytes.fromhex('302a300506032b6570032100')):
        raise ValueError('trusted public key must be Ed25519')
    openssl(['pkeyutl', '-verify', '-pubin', '-rawin', '-inkey', str(public_key), '-in', str(statement), '-sigfile', str(signature)])
