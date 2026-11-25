import concurrent.futures
import json
import random
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
import urllib.request
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from synapse_testnet.demo import setup,execute,call
from synapse_testnet.ledger import Denied,canonical,split,sha
from synapse_testnet.synthetic import MODEL,receipt_body,observed
from synapse_testnet.service import Service
from synapse_testnet.http_api import Server

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'ledger.db'; self.s,self.t,self.keys=setup(self.path)
        self.client=self.t['client']; self.op=self.t['operator']; self.v=self.t['verifier']
    def quote(self,output=8,token=None): return self.s.quote(token or self.client,MODEL,'demo-v1',2,output)['quote']
    def job(self): return self.s.reserve(self.client,self.quote(),'job-1')
    def receipts(self,corrupt=False,count=4):
        for w,k in self.keys.items():
            a=self.s.assignments(self.t[w])[0]; body=receipt_body(a,count)
            if corrupt: body['evidence']='0'*64
            sig=k.sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex()
            r=self.s.receipt(self.t[w],body,sig); self.assertEqual(r,self.s.receipt(self.t[w],body,sig))
    def validate(self):
        for p in self.s.pending(self.v):
            b=json.loads(p['body']); ref=observed({**b,'first_layer':p['first_layer'],'end_layer':p['end_layer']},b['output_units'])
            self.s.validate(self.v,p['assignment'],p['digest'],True,b['output_units'],ref)
    def assertDenied(self,code,func,*args):
        with self.assertRaises(Denied) as e: func(*args)
        self.assertEqual(e.exception.code,code)
    def test_http_demo_sharded(self):
        with tempfile.TemporaryDirectory() as d:
            r=execute(Path(d)/'db'); self.assertEqual(r['balances']['client']['available'],88_000_000); self.assertEqual(r['balances']['worker-a']['available'],6_000_000)
    def test_http_demo_full_redeem(self):
        with tempfile.TemporaryDirectory() as d: self.assertEqual(execute(Path(d)/'db',False)['redemption']['charged'],5_000_000)
    def test_balance_before_after(self):
        job=self.job(); self.assertEqual(self.s.balance(self.client)['held'],20_000_000)
        self.receipts(); self.assertDenied('WORK_NOT_VERIFIED',self.s.settle,self.client,job['job']); self.validate(); result=self.s.settle(self.client,job['job']); self.assertEqual(result['charged'],12_000_000); self.assertEqual(result,self.s.settle(self.client,job['job']))
        self.assertEqual(self.s.stats(self.t['worker-a'])['verified_jobs'],1)
    def test_concurrent_idempotent_reservation(self):
        q=self.quote()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as p: jobs=list(p.map(lambda _:self.s.reserve(self.client,q,'same'),range(16)))
        self.assertEqual(len({j['job'] for j in jobs}),1); self.assertEqual(self.s.balance(self.client)['held'],20_000_000)
    def test_concurrent_capacity_rejection(self):
        q1,q2=self.quote(),self.quote()
        def run(q):
            try: return self.s.reserve(self.client,q,q)['state']
            except Denied as e: return e.code
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p: self.assertCountEqual(list(p.map(run,[q1,q2])),['running','NO_ROUTE'])
    def test_insufficient_balance_rolls_back(self):
        self.s.tariff(self.op,'costly',99_000_000,1_000_000,2_000_000); q=self.s.quote(self.client,MODEL,'costly',2,8)['quote']
        self.assertDenied('INSUFFICIENT_CREDITS',self.s.reserve,self.client,q,'expensive'); self.assertEqual(self.s.balance(self.client)['held'],0); self.job()
    def test_cancel_idempotent(self):
        j=self.job()['job']; self.assertEqual(self.s.cancel(self.client,j)['state'],'cancelled'); self.s.cancel(self.client,j); self.assertEqual(self.s.balance(self.client)['available'],100_000_000)
    def test_expiry_restart(self):
        j=self.job(); new=Service(self.path,clock=lambda:j['expires']+1); self.assertEqual(new.sweep(),1); self.assertEqual(new.sweep(),0); self.assertEqual(new.balance(self.client)['available'],100_000_000)
    def test_invalid_signature(self):
        self.job(); a=self.s.assignments(self.t['worker-a'])[0]; self.assertDenied('BAD_SIGNATURE',self.s.receipt,self.t['worker-a'],receipt_body(a,4),'00'*64)
    def test_forged_computation_refunded(self):
        j=self.job(); self.receipts(True); pending=self.s.pending(self.v)[0]; b=json.loads(pending['body']); self.assertEqual(self.s.validate(self.v,pending['assignment'],pending['digest'],True,b['output_units'],b['evidence']),'rejected'); self.assertEqual(self.s.job(self.client,j['job'])['state'],'failed'); self.assertEqual(self.s.balance(self.client)['available'],100_000_000)
    def test_scope_rejects_other_worker(self):
        self.job(); a=self.s.assignments(self.t['worker-a'])[0]; body=receipt_body(a,4); sig=self.keys['worker-a'].sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex(); self.assertDenied('NOT_FOUND',self.s.receipt,self.t['worker-b'],body,sig)
    def test_over_limit_receipt(self):
        self.job(); a=self.s.assignments(self.t['worker-a'])[0]; body=receipt_body(a,9); sig=self.keys['worker-a'].sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex(); self.assertDenied('RECEIPT_SCOPE',self.s.receipt,self.t['worker-a'],body,sig)
    def test_duplicate_conflicting_receipt(self):
        self.job(); self.receipts(); a=self.s.assignments(self.t['worker-a'])[0]; body=receipt_body(a,3); sig=self.keys['worker-a'].sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex(); self.assertDenied('RECEIPT_CONFLICT',self.s.receipt,self.t['worker-a'],body,sig)
    def test_device_proof_replay(self):
        body=self.s.device_challenge(self.t['worker-a'],'worker-a-device'); sig=self.keys['worker-a'].sign(b'synapse-testnet-device-v1\0'+canonical(body)).hex(); self.s.prove_device(self.t['worker-a'],body,sig); self.assertDenied('EXPIRED_OR_REPLAYED_PROOF',self.s.prove_device,self.t['worker-a'],body,sig)
    def test_revocation_refunds(self):
        j=self.job()['job']; self.s.revoke(self.op,'worker-a-device','device'); self.assertEqual(self.s.job(self.client,j)['state'],'failed'); self.assertEqual(self.s.balance(self.client)['held'],0); self.assertDenied('NO_ROUTE',self.s.reserve,self.client,self.quote(),'new')
    def test_suspension_refunds(self):
        j=self.job()['job']; self.s.set_pool(self.op,False); self.assertEqual(self.s.job(self.client,j)['state'],'failed'); self.assertDenied('POOL_OR_CONSENT_REQUIRED',self.s.quote,self.client,MODEL,'demo-v1',1,1)
    def test_model_revocation(self):
        j=self.job()['job']; self.s.disable_model(self.op,MODEL); self.assertEqual(self.s.job(self.client,j)['state'],'failed'); self.assertDenied('MODEL_UNSUPPORTED',self.s.quote,self.client,MODEL,'demo-v1',1,1)
    def test_member_and_pool_scope(self):
        j=self.job()['job']; other=self.s.bootstrap('other','op2',100); t=self.s.add_member(other,'other-user','requester'); self.assertDenied('NOT_FOUND',self.s.job,t,j); self.assertDenied('FORBIDDEN',self.s.grant,self.client,'client',1)
    def test_consent(self):
        t=self.s.add_member(self.op,'new','requester'); self.assertDenied('POOL_OR_CONSENT_REQUIRED',self.s.quote,t,MODEL,'demo-v1',1,1); self.assertDenied('EXPLICIT_CONSENT_REQUIRED',self.s.consent,t,'testnet-v1',False)
    def test_no_non_synthetic_model(self): self.assertDenied('TESTNET_SYNTHETIC_MODELS_ONLY',self.s.approve_model,self.op,'f'*64,4,64,True)
    def test_self_dealing(self):
        self.assertDenied('NO_ROUTE',self.s.reserve,self.t['worker-a'],self.quote(token=self.t['worker-a']),'self')
    def test_treasury_exhaustion(self): self.assertDenied('INSUFFICIENT_CREDITS',self.s.grant,self.op,'worker-a',1)
    def test_dispute_independent_review(self):
        j=self.job()['job']; self.receipts(); self.validate(); self.s.settle(self.client,j); d=self.s.dispute(self.client,j,'incorrect_charge'); self.assertDenied('INDEPENDENT_DISPUTE_REVIEW_REQUIRED',self.s.review_dispute,self.v,d,1); self.s.review_dispute(self.t['reviewer'],d,1); self.assertDenied('INSUFFICIENT_CREDITS',self.s.resolve_dispute,self.op,d)
    def test_journal_append_only(self):
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            with self.s.tx() as db: db.execute('DELETE FROM journal')
    def test_projection_mismatch(self):
        with self.assertRaises(Denied):
            with self.s.tx() as db: db.execute("UPDATE accounts SET balance=balance-1 WHERE kind='available'")
        self.assertEqual(self.s.balance(self.client)['available'],100_000_000)
    def test_randomized_rounding(self):
        rnd=random.Random(31)
        for _ in range(1000):
            amount=rnd.randrange(100000); weights={str(i):rnd.randrange(1,256) for i in range(rnd.randrange(1,10))}; got=split(amount,weights); self.assertEqual(sum(got.values()),amount); self.assertTrue(all(v>=0 for v in got.values())); self.assertEqual(got,split(amount,dict(reversed(list(weights.items())))))
    def test_concurrent_settlement(self):
        j=self.job()['job']; self.receipts(); self.validate()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p: out=list(p.map(lambda _: self.s.settle(self.client,j),range(10)))
        self.assertTrue(all(v==out[0] for v in out)); self.assertEqual(self.s.balance(self.t['worker-a'])['available'],6_000_000)
    def test_http_security(self):
        with Server(self.s) as server:
            thread=threading.Thread(target=server.serve_forever); thread.start()
            try:
                url=f'http://127.0.0.1:{server.server_port}/v1/credits/balance'
                for headers,status in [({},401),({'Host':'attacker.invalid'},403),({'Origin':'http://attacker.invalid'},403)]:
                    with self.assertRaises(HTTPError) as cm: urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=3)
                    self.assertEqual(cm.exception.code,status)
                self.assertEqual(call(server,self.client,'/v1/credits/balance')['available'],100_000_000)
            finally: server.shutdown(); thread.join()
    def test_invalid_integer_bool(self): self.assertDenied('INVALID_INTEGER',self.s.quote,self.client,MODEL,'demo-v1',True,1)
    def test_quote_conflict(self):
        self.job(); self.assertDenied('IDEMPOTENCY_CONFLICT',self.s.reserve,self.client,self.quote(),'job-1')

if __name__=='__main__': unittest.main()
