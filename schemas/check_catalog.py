#!/usr/bin/env python3
"""Check catalog shapes, exact signing bytes, and an independent OpenSSL signature.

Public fixture keys are test data; these checks never enroll them in application
policy. Current-time approval is intentionally not claimed for a golden record.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from jsonschema import Draft202012Validator, ValidationError
from check import validate as validate_existing, rust_fields

HERE = Path(__file__).resolve().parent

def load(name):
    return json.loads((HERE / name).read_text())

def validate(name, value):
    Draft202012Validator(load(name + '.schema.json')).validate(value)
    if name == 'model-catalog-v1':
        validate_existing('model-manifest-v1', value['manifest'])
        if not 0 < value['expires_at_unix'] - value['issued_at_unix'] <= 366 * 86400:
            raise ValueError('invalid catalog lifetime')
        for field, cap in [('license', 256), ('provenance', 4096)]:
            if not value[field].strip() or len(value[field].encode()) > cap or any(ord(c) < 32 or 127 <= ord(c) < 160 for c in value[field]):
                raise ValueError('invalid bounded assertion text')
    if name == 'catalog-trust-root-v1':
        peers = [p['peer_id'] for p in value['publishers']]
        if len(set(peers)) != len(peers):
            raise ValueError('duplicate trusted publisher')

def transcript(record):
    canonical = {k: record[k] for k in ('schema_version', 'publisher', 'public_key', 'issued_at_unix', 'expires_at_unix', 'status', 'license', 'rights', 'provenance', 'manifest', 'signature')}
    canonical['rights'] = {k: record['rights'][k] for k in ('local_inference', 'remote_hosting', 'redistribution')}
    manifest = record['manifest']
    canonical['manifest'] = {k: manifest[k] for k in ('schema_version', 'model_id', 'revision', 'weights')}
    canonical['manifest']['artifacts'] = [{k: a[k] for k in ('filename', 'size_bytes', 'sha256')} for a in sorted(manifest['artifacts'], key=lambda a: a['filename'])]
    canonical['signature'] = []
    return b'synapse-model-catalog-v1\0' + json.dumps(canonical, ensure_ascii=False, separators=(',', ':')).encode()

def verify_signature(record, payload):
    # The golden fixture pins libp2p's Ed25519 protobuf form: field 1 type=1,
    # field 2 32-byte raw key. General key decoding remains the runtime's job.
    key = bytes(record['public_key'])
    assert len(key) == 36 and key[:4] == bytes([8, 1, 18, 32])
    with tempfile.TemporaryDirectory(prefix='synapse-catalog-contract-') as temp:
        path = Path(temp)
        (path / 'key.der').write_bytes(bytes.fromhex('302a300506032b6570032100') + key[4:])
        (path / 'payload').write_bytes(payload)
        (path / 'signature').write_bytes(bytes(record['signature']))
        result = subprocess.run(['openssl', 'pkeyutl', '-verify', '-pubin', '-keyform', 'DER', '-inkey', str(path / 'key.der'), '-rawin', '-in', str(path / 'payload'), '-sigfile', str(path / 'signature')], capture_output=True)
        return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    for name in ('model-catalog-v1', 'catalog-trust-root-v1'):
        Draft202012Validator.check_schema(load(name + '.schema.json'))
        validate(name, load('fixtures/' + name + '.json'))
    record = load('fixtures/model-catalog-v1.json')
    golden = load('fixtures/catalog-signing-v1.json')
    payload = transcript(record)
    assert payload == golden['transcript_utf8'].encode()
    assert hashlib.sha256(payload).hexdigest() == golden['sha256']
    assert verify_signature(record, payload), 'Rust-produced fixture must verify independently'
    shuffled = deepcopy(record)
    shuffled['manifest']['artifacts'].reverse()
    assert transcript(shuffled) == payload
    changed = deepcopy(record)
    changed['rights']['redistribution'] = True
    assert not verify_signature(changed, transcript(changed)), 'altered rights must fail signature check'
    invalid = 0
    for change in [lambda v: v.update(signature=[0] * 63), lambda v: v.update(status='unknown'), lambda v: v.update(expires_at_unix=v['issued_at_unix']), lambda v: v['rights'].pop('redistribution'), lambda v: v['manifest'].update(weights='../secret'), lambda v: v.update(license=''), lambda v: v.update(schema_version=2), lambda v: v.update(unrecognized=True)]:
        bad = deepcopy(record)
        change(bad)
        try:
            validate('model-catalog-v1', bad)
        except (ValueError, ValidationError):
            invalid += 1
        else:
            raise AssertionError('invalid catalog accepted')
    if args.source_root:
        source = (args.source_root / 'crates/synapse-core/src/catalog.rs').read_text()
        for rust, schema in [('SignedModelRecord', 'model-catalog-v1'), ('TrustRoot', 'catalog-trust-root-v1')]:
            assert rust_fields(source, rust) == list(load(schema + '.schema.json')['properties']), 'catalog field drift'
        assert rust_fields(source, 'ModelRights') == ['local_inference', 'remote_hosting', 'redistribution']
        assert rust_fields(source, 'TrustedPublisher') == ['peer_id', 'public_key']
    print(f'Catalog shapes, canonical transcript, independent signature and {invalid} invalid fixtures passed.')

if __name__ == '__main__':
    main()
