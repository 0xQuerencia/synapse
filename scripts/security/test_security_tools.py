import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


hygiene = module('hygiene', 'scripts/security/check_repository.py')
sbom = module('sbom', 'scripts/release/generate_sbom.py')


class RepositoryChecks(unittest.TestCase):
    def test_sensitive_values_never_appear_in_diagnostics(self):
        secret = b'ghp_' + b'a' * 40
        failures = hygiene.inspect_content('example.txt', secret)
        self.assertEqual(len(failures), 1)
        self.assertNotIn(secret.decode(), failures[0])

    def test_private_key_is_rejected(self):
        content = b'-----BEGIN ' + b'PRIVATE KEY-----'
        self.assertTrue(hygiene.inspect_content('file.txt', content))

    def test_risky_workflow_is_rejected(self):
        self.assertEqual(len(hygiene.inspect_content('.github/workflows/test.yml', b'on:\n  pull_request_target:\npermissions: write-all\n')), 2)

    def test_documented_placeholders_are_allowed(self):
        self.assertFalse(hygiene.inspect_content('guide.md', b'Authorization: Bearer <local-token>'))


class InventoryChecks(unittest.TestCase):
    def fixture(self):
        return {'workspace_members': ['local'], 'packages': [{'id': 'local', 'name': 'synapse', 'version': '0.2.0', 'license': 'MIT'}, {'id': 'dep', 'name': 'a-library', 'version': '1.0.0', 'license': None}], 'resolve': {'nodes': [{'id': 'local', 'dependencies': ['dep']}, {'id': 'dep', 'dependencies': []}]}}

    def test_transitive_graph_and_missing_rights_are_preserved(self):
        data = sbom.build_sbom(self.fixture())
        self.assertEqual(len(data['components']), 2)
        self.assertEqual(sum(len(x['dependsOn']) for x in data['dependencies']), 1)
        dependency = next(x for x in data['components'] if x['name'] == 'a-library')
        self.assertEqual(dependency['properties'][0]['value'], 'requires-review')

    def test_partial_metadata_is_rejected(self):
        data = self.fixture()
        data['resolve'] = None
        with self.assertRaises(ValueError):
            sbom.build_sbom(data)

    def test_broken_graph_is_rejected(self):
        data = self.fixture()
        data['resolve']['nodes'][0]['dependencies'] = ['missing']
        with self.assertRaises(ValueError):
            sbom.build_sbom(data)


if __name__ == '__main__':
    unittest.main()
