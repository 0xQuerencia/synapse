"""The checked-in original fixture must be available on a complete checkout."""
import shutil
import tempfile
import unittest
from pathlib import Path
from synapse_testnet.fixture_transfer import deliver_fixture
from synapse_testnet.native_lab import LAB_MODEL, verify_fixture
from synapse_testnet.ledger import Denied

FIXTURE=Path(__file__).resolve().parents[2]/'crates/synapse-core/tests/fixtures/tiny-llama'

@unittest.skipUnless((FIXTURE/'manifest.json').is_file(),'complete native fixture checkout required')
class FixtureTransferTests(unittest.TestCase):
    def test_exact_native_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            path,report=deliver_fixture(FIXTURE,root/'transfer')
            self.assertEqual(verify_fixture(path),LAB_MODEL)
            self.assertEqual(report['model_fingerprint'],LAB_MODEL)
            self.assertEqual(len(report['files']),4)
            for file in report['files']:
                self.assertEqual((path/file['file']).read_bytes(),(FIXTURE/file['file']).read_bytes())
                self.assertTrue(file['complete']);self.assertGreater(file['fetched'],0)

    def test_modified_manifest_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'bad';shutil.copytree(FIXTURE,source)
            path=source/'manifest.json';path.write_text(path.read_text().replace('fixture-v1','unknown'))
            with self.assertRaisesRegex(Denied,'NATIVE_FIXTURE_MISMATCH'):
                deliver_fixture(source,root/'transfer')
            self.assertFalse((root/'transfer').exists())
