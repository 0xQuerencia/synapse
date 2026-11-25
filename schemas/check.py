#!/usr/bin/env python3
"""Validate published examples, cross-field invariants and optional Rust shape drift.

JSON schemas describe serde data shapes; they are not the bincode wire codec.
Run with --source-root after the runtime commits are integrated, and --wire-root
for the 0.4 network implementation. Neither check claims interoperability alone.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import unicodedata

from jsonschema import Draft202012Validator, ValidationError

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


def canonical_manifest(value):
    ordered = {key: value[key] for key in ('schema_version', 'model_id', 'revision', 'weights')}
    ordered['artifacts'] = [
        {key: artifact[key] for key in ('filename', 'size_bytes', 'sha256')}
        for artifact in sorted(value['artifacts'], key=lambda item: item['filename'])
    ]
    return json.dumps(ordered, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def validate(name, value):
    contract = load(name + '.schema.json')
    Draft202012Validator(contract).validate(value)
    if name == 'model-manifest-v1':
        if len(value['model_id'].encode()) > 256 or any(unicodedata.category(c) == 'Cc' for c in value['model_id']):
            raise ValueError('model ID is not bounded printable text')
        artifacts = value['artifacts']
        names = [a['filename'] for a in artifacts]
        if len(set(names)) != len(names):
            raise ValueError('duplicate artifact name')
        if not {'config.json', 'tokenizer.json', value['weights']}.issubset(names):
            raise ValueError('required artifact missing')
        for filename in names + [value['weights']]:
            allowed = filename.endswith(('.safetensors', '.gguf')) or filename in {
                'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'
            }
            if not allowed:
                raise ValueError('unapproved artifact format')
        if not value['weights'].endswith(('.safetensors', '.gguf')) and value['weights'] != 'model.safetensors.index.json':
            raise ValueError('weights must name a supported checkpoint or index')
    elif name in {'shard-plan-v0.4', 'shard-assignment-v1', 'signed-availability-v1'}:
        if not value['layer_start'] < value['layer_end'] <= value['total_layers']:
            raise ValueError('invalid half-open layer interval')
        if name == 'signed-availability-v1' and not 0 < value['expires_at_unix'] - value['issued_at_unix'] <= 300:
            raise ValueError('invalid signed availability lifetime')
    elif name == 'shard-ready-v1':
        validate('shard-assignment-v1', value['assignment'])
        if value['resident_weight_bytes'] > value['assignment']['memory_limit_bytes']:
            raise ValueError('resident weights exceed assigned memory')
        if value['availability'] is not None:
            record = value['availability']
            validate('signed-availability-v1', record)
            for key in ['key', 'layer_start', 'layer_end', 'total_layers']:
                if record[key] != value['assignment'][key]:
                    raise ValueError('availability does not match assignment')
            if record['resident_weight_bytes'] != value['resident_weight_bytes']:
                raise ValueError('availability does not match residency')
    elif name == 'tensor-v0.4':
        width = {'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8}[value['dtype']]
        if math.prod(value['shape']) * width != value['original_size']:
            raise ValueError('shape/dtype byte length mismatch')
        if len(value['data']) != value['compressed_size']:
            raise ValueError('payload byte length mismatch')
        if value['compression'] == 'none' and value['original_size'] != value['compressed_size']:
            raise ValueError('uncompressed length mismatch')
    elif name == 'evidence-v1':
        if value['status'] in {'locally_tested', 'accepted', 'released'} and not value['commands']:
            raise ValueError('tested status requires actual commands')
        if value['status'] in {'accepted', 'released'} and (value['reviewer'] is None or value['source_revision'] is None or value['dirty_worktree']):
            raise ValueError('acceptance needs reviewer and clean identified revision')
        if value['status'] == 'released' and not value['release_artifact']:
            raise ValueError('release needs artifact identifier')


def rejected(name, value):
    try:
        validate(name, value)
    except (ValueError, ValidationError):
        return
    raise AssertionError(f'invalid {name} fixture was accepted')


def rust_fields(source, name):
    match = re.search(r'pub struct ' + re.escape(name) + r'\s*\{(.*?)\n\}', source, re.S)
    if not match:
        raise AssertionError(f'missing Rust struct {name}')
    return re.findall(r'^\s*pub (\w+)\s*:', match.group(1), re.M)


def source_shapes(root):
    model = (root / 'crates/synapse-core/src/inference/model_loader.rs').read_text()
    protocol = (root / 'crates/synapse-core/src/protocol.rs').read_text()
    assignment = (root / 'crates/synapse-core/src/inference/assignment.rs').read_text()
    registry = (root / 'crates/synapse-core/src/inference/registry.rs').read_text()
    pairs = [('ModelManifest', model, 'model-manifest-v1'), ('ShardPlan', protocol, 'shard-plan-v0.4'), ('TensorPayload', protocol, 'tensor-v0.4'), ('ShardAssignment', assignment, 'shard-assignment-v1'), ('ShardReady', assignment, 'shard-ready-v1'), ('SignedAvailability', registry, 'signed-availability-v1')]
    for struct_name, source, contract in pairs:
        assert rust_fields(source, struct_name) == list(load(contract + '.schema.json')['properties']), f'field drift in {struct_name}'
    assert rust_fields(model, 'ModelArtifact') == ['filename', 'size_bytes', 'sha256']
    assert rust_fields(assignment, 'LeaseKey') == ['session_id', 'route_epoch', 'model_digest']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--wire-root', type=Path)
    args = parser.parse_args()
    for path in HERE.glob('*.schema.json'):
        Draft202012Validator.check_schema(json.loads(path.read_text()))
    for name in ['model-manifest-v1', 'shard-plan-v0.4', 'tensor-v0.4', 'shard-assignment-v1', 'shard-ready-v1']:
        validate(name, load('fixtures/' + name + '.json'))
    manifest = load('fixtures/model-manifest-v1.json')
    golden = load('fixtures/model-manifest-digest-v1.json')
    assert canonical_manifest(manifest) == golden['canonical_utf8'].encode()
    assert hashlib.sha256(canonical_manifest(manifest)).hexdigest() == golden['sha256']
    reordered = deepcopy(manifest)
    reordered['artifacts'].reverse()
    assert canonical_manifest(manifest) == canonical_manifest(reordered)
    changed = deepcopy(manifest)
    changed['revision'] = 'fixture-v2'
    assert canonical_manifest(manifest) != canonical_manifest(changed)
    negatives = 0
    for mutate in [lambda v: v.update(schema_version=2), lambda v: v.update(revision='main'), lambda v: v.update(weights='../model.safetensors'), lambda v: v['artifacts'].append(v['artifacts'][0]), lambda v: v['artifacts'][0].update(sha256='bad'), lambda v: v['artifacts'][0].update(size_bytes=0), lambda v: v['artifacts'][0].update(filename='script.py'), lambda v: v.update(unrecognized=True), lambda v: v.update(model_id='bad\nname')]:
        bad = deepcopy(manifest)
        mutate(bad)
        rejected('model-manifest-v1', bad)
        negatives += 1
    for patch in [{'layer_start': 7, 'layer_end': 6}, {'layer_end': 9}, {'max_sequence_len': 4097}, {'dtype': 'F64'}]:
        rejected('shard-plan-v0.4', {**load('fixtures/shard-plan-v0.4.json'), **patch})
        negatives += 1
    for patch in [{'shape': [1, 1, 3]}, {'data': [0]}, {'original_size': 67108865}, {'compression': 'unknown'}, {'data': [256] * 8}]:
        rejected('tensor-v0.4', {**load('fixtures/tensor-v0.4.json'), **patch})
        negatives += 1
    assignment = load('fixtures/shard-assignment-v1.json')
    for patch in [{'protocol_version': 2}, {'layer_end': 9}, {'total_layers': 1025}, {'context_tokens': 4097}, {'memory_limit_bytes': 0}, {'ttl_ms': 999}, {'ttl_ms': 300001}, {'key': {**assignment['key'], 'route_epoch': 0}}, {'key': {**assignment['key'], 'session_id': 'bad'}}]:
        rejected('shard-assignment-v1', {**assignment, **patch})
        negatives += 1
    ready = load('fixtures/shard-ready-v1.json')
    rejected('shard-ready-v1', {**ready, 'resident_weight_bytes': assignment['memory_limit_bytes'] + 1})
    negatives += 1
    # These bytes are shape-only test data, deliberately not valid identity or
    # signature evidence. Cryptographic verification is the Rust handler's job.
    availability = {'schema_version': 1, 'peer_id': 'shape-only', 'key': assignment['key'], 'layer_start': 2, 'layer_end': 6, 'total_layers': 8, 'backend': 'llama-safetensors', 'resident_weight_bytes': ready['resident_weight_bytes'], 'sequence': 1, 'issued_at_unix': 1000, 'expires_at_unix': 1060, 'public_key': [0], 'signature': [0] * 64}
    validate('shard-ready-v1', {**ready, 'availability': availability})
    for patch in [{'expires_at_unix': 1301}, {'expires_at_unix': 1000}, {'sequence': 0}, {'backend': 'unknown'}, {'signature': [0]}]:
        rejected('signed-availability-v1', {**availability, **patch})
        negatives += 1
    rejected('shard-ready-v1', {**ready, 'availability': {**availability, 'layer_start': 1}})
    negatives += 1
    evidence = {'schema_version': 1, 'package': 'PR-39', 'status': 'proposed', 'source_revision': None, 'dirty_worktree': True, 'environment': 'synthetic contract example', 'topology': 'none', 'commands': [], 'limitations': ['Fixture only'], 'reviewer': None, 'release_artifact': None}
    validate('evidence-v1', evidence)
    for status in ['locally_tested', 'accepted', 'released']:
        rejected('evidence-v1', {**evidence, 'status': status})
        negatives += 1
    wire = load('fixtures/wire-v0.4.json')
    for vector in wire['vectors']:
        body = struct.pack('<I', vector['variant_index'])
        frame = bytes([wire['marker']]) + struct.pack('<I', len(body)) + body
        assert body.hex() == vector['body_hex'] and frame.hex() == vector['frame_hex']
    if args.source_root:
        source_shapes(args.source_root)
    if args.wire_root:
        codec = (args.wire_root / 'crates/synapse-core/src/network/codec.rs').read_text()
        assert '"/synapse/0.4.0"' in codec, '0.4 contract cannot describe a 0.3 codec'
    print(f'Schemas and golden fixtures passed; {negatives} invalid cases rejected.')
    print('Rust shape check:', 'passed' if args.source_root else 'not requested')
    print('Wire version check:', 'passed' if args.wire_root else 'not requested')


if __name__ == '__main__':
    main()
