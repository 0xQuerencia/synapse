import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from synapse_testnet.demo import setup
from synapse_testnet.ops import backup, restore, summary
from synapse_testnet.service import Service
from synapse_testnet.synthetic import MODEL
from synapse_testnet.ledger import Denied

class OpsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name); self.path=self.root/'db'; self.s,self.t,self.keys=setup(self.path)
    def test_snapshot_restore_acknowledged_reservation(self):
        q=self.s.quote(self.t['client'],MODEL,'demo-v1',2,8); j=self.s.reserve(self.t['client'],q['quote'],'ack')
        b=backup(self.s,self.root/'backup'); self.assertGreater(b['journal_transactions'],0)
        restored=self.root/'restored'; r=restore(self.root/'backup',restored); self.assertEqual(r['open_jobs'],1)
        s=Service(restored); self.assertEqual(s.job(self.t['client'],j['job']),j); self.assertEqual(s.balance(self.t['client'])['held'],20_000_000)
        s.cancel(self.t['client'],j['job']); self.assertEqual(s.balance(self.t['client'])['available'],100_000_000)
    def test_corrupt_backup_rejected(self):
        backup(self.s,self.root/'backup'); p=self.root/'backup'/'ledger.sqlite'; p.write_bytes(p.read_bytes()+b'x')
        with self.assertRaisesRegex(Denied,'BACKUP_HASH_MISMATCH'): restore(self.root/'backup',self.root/'restored')
    def test_crash_before_commit_rolls_back(self):
        before=summary(self.s)
        code='import os,sys; from synapse_testnet.service import Service; s=Service(sys.argv[1]);\nwith s.tx() as db:\n db.execute("UPDATE accounts SET balance=0 WHERE kind=\'available\'"); os._exit(77)'
        p=subprocess.run([sys.executable,'-c',code,str(self.path)],timeout=10); self.assertEqual(p.returncode,77)
        self.assertEqual(summary(Service(self.path)),before); self.assertEqual(self.s.balance(self.t['client'])['available'],100_000_000)
    def test_restore_refuses_overwrite(self):
        backup(self.s,self.root/'backup')
        with self.assertRaisesRegex(Denied,'RESTORE_TARGET_EXISTS'): restore(self.root/'backup',self.path)
