#!/usr/bin/env python3
"""Produce a CycloneDX 1.6 inventory from the actual locked Cargo resolve graph."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote


def build_sbom(metadata):
    if metadata.get('resolve') is None:
        raise ValueError('full Cargo resolve graph required; --no-deps is insufficient')
    packages = metadata['packages']
    refs = {p['id']: 'cargo:' + hashlib.sha256(p['id'].encode()).hexdigest() for p in packages}
    if len(refs) != len(packages):
        raise ValueError('duplicate package IDs')
    workspace = set(metadata.get('workspace_members', []))
    components = []
    for p in sorted(packages, key=lambda p: (p['name'], p['version'], p['id'])):
        item = {'type': 'application' if p['id'] in workspace else 'library', 'bom-ref': refs[p['id']], 'name': p['name'], 'version': p['version'], 'purl': f"pkg:cargo/{quote(p['name'], safe='')}@{quote(p['version'], safe='')}"}
        if p.get('license'):
            item['licenses'] = [{'expression': p['license']}]
        else:
            item['properties'] = [{'name': 'synapse:license-status', 'value': 'requires-review'}]
        components.append(item)
    dependencies = []
    for node in metadata['resolve']['nodes']:
        if node['id'] not in refs or any(d not in refs for d in node['dependencies']):
            raise ValueError('resolve graph references a missing component')
        dependencies.append({'ref': refs[node['id']], 'dependsOn': sorted(refs[d] for d in node['dependencies'])})
    return {'bomFormat': 'CycloneDX', 'specVersion': '1.6', 'version': 1, 'metadata': {'properties': [{'name': 'synapse:provenance', 'value': 'cargo metadata --locked --format-version 1'}, {'name': 'synapse:license-review', 'value': 'Root MIT and workspace Apache-2.0 discrepancy remains unresolved; metadata is not a rights decision.'}]}, 'components': components, 'dependencies': sorted(dependencies, key=lambda d: d['ref'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, help='Existing full cargo metadata JSON; otherwise invoke Cargo')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.metadata:
        metadata = json.loads(args.metadata.read_text())
    else:
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(['cargo', 'metadata', '--locked', '--format-version', '1'], cwd=root, check=True, capture_output=True, text=True)
        metadata = json.loads(result.stdout)
    sbom = build_sbom(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sbom, indent=2) + '\n')
    print(f'Wrote inventory of {len(sbom["components"])} resolved packages to {args.output}')


if __name__ == '__main__':
    main()
