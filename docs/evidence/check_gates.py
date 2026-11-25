#!/usr/bin/env python3
"""Check acceptance-record completeness; does not certify referenced evidence."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
record = json.loads((root / 'docs/evidence/gates.json').read_text())
assert record['schema_version'] == 1
assert [gate['id'] for gate in record['gates']] == [f'G{i}' for i in range(7)]
for gate in record['gates']:
    assert isinstance(gate['accepted'], bool)
    assert gate['state'] in {'in_progress', 'awaiting_external_evidence', 'disabled', 'research', 'accepted'}
    assert gate['required_evidence'] and all(isinstance(value, str) and value for value in gate['required_evidence'])
    if gate['accepted']:
        assert gate['state'] == 'accepted' and gate['evidence'], 'acceptance needs actual evidence'
        for evidence in gate['evidence']:
            assert all(evidence.get(key) for key in ['source_revision', 'environment', 'artifact', 'reviewer'])
    else:
        assert gate['state'] != 'accepted'
print('Seven gate records are structurally consistent; no independent acceptance is inferred.')
