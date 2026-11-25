#!/usr/bin/env python3
"""Check acceptance-record completeness; does not certify referenced evidence."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
record = json.loads((root / 'docs/evidence/gates.json').read_text())
assert record['schema_version'] == 1
implementation = record.get('local_implementation')
if implementation is not None:
    packages = implementation['package_ids']
    assert implementation['package_count'] == len(packages) == len(set(packages))
    assert all(value.startswith('PR-') and value[3:].isdigit() and 1 <= int(value[3:]) <= 56 for value in packages)
    assert implementation['evidence_record'] and implementation['scope']
    for name in ['github_pull_requests_published', 'release_artifacts_published', 'production_deployments']:
        assert isinstance(implementation[name], int) and implementation[name] >= 0
assert [gate['id'] for gate in record['gates']] == [f'G{i}' for i in range(7)]
for gate in record['gates']:
    assert isinstance(gate['accepted'], bool)
    assert gate['state'] in {'in_progress', 'awaiting_external_evidence', 'disabled', 'research', 'accepted'}
    assert gate['required_evidence'] and all(isinstance(value, str) and value for value in gate['required_evidence'])
    for evidence in gate.get('local_evidence', []):
        assert evidence['artifact'] and evidence['scope'], 'local evidence needs its actual scope'
    if gate['accepted']:
        assert gate['state'] == 'accepted' and gate['evidence'], 'acceptance needs actual evidence'
        for evidence in gate['evidence']:
            assert all(evidence.get(key) for key in ['source_revision', 'environment', 'artifact', 'reviewer'])
    else:
        assert gate['state'] != 'accepted'
print('Seven gate records are structurally consistent; no independent acceptance is inferred.')
