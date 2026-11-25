import tempfile
import unittest
from pathlib import Path
from synapse_testnet.demo import setup
from synapse_testnet.synthetic import MODEL
from synapse_testnet.ledger import Denied,canonical
from synapse_testnet.service import Service

class PresenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.path=Path(self.tmp.name)/'db'; self.s,self.t,self.keys=setup(self.path); self.c=self.t['client']; self.w=self.t['worker-a']
    def q(self): return self.s.quote(self.c,MODEL,'demo-v1',2,8)['quote']
    def test_pause_and_resume(self):
        self.s.worker_control(self.w,'worker-a-device','pause')
        with self.assertRaisesRegex(Denied,'NO_ROUTE'): self.s.reserve(self.c,self.q(),'x')
        self.s.worker_control(self.w,'worker-a-device','resume'); self.assertEqual(self.s.reserve(self.c,self.q(),'x')['state'],'running')
    def test_stop_refunds(self):
        j=self.s.reserve(self.c,self.q(),'x'); self.s.worker_control(self.w,'worker-a-device','stop'); self.assertEqual(self.s.job(self.c,j['job'])['state'],'failed'); self.assertEqual(self.s.balance(self.c)['held'],0)
    def test_heartbeat_expiry_refunds(self):
        j=self.s.reserve(self.c,self.q(),'x'); s=Service(self.path,clock=lambda:self.s.clock()+61); self.assertEqual(s.sweep(),1); self.assertEqual(s.job(self.c,j['job'])['state'],'failed')
    def test_heartbeat_replay_denied(self):
        b={'device':'worker-a-device','sequence':1,'issued':self.s.clock()}; signature=self.keys['worker-a'].sign(b'synapse-testnet-heartbeat-v1\0'+canonical(b)).hex(); self.s.heartbeat(self.w,b,signature)
        with self.assertRaisesRegex(Denied,'STALE_HEARTBEAT'): self.s.heartbeat(self.w,b,signature)
