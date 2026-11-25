import json
import tempfile
import unittest
from pathlib import Path
from synapse_testnet.demo import setup
from synapse_testnet.ledger import Denied, canonical
from synapse_testnet.synthetic import MODEL,receipt_body,observed

class ContributorTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.s,self.t,self.keys=setup(Path(tmp.name)/'db');self.client=self.t['client'];self.a=self.t['worker-a'];self.b=self.t['worker-b'];self.op=self.t['operator'];self.v=self.t['verifier']
    def job(self,idem='first'):
        return self.s.reserve(self.client,self.s.quote(self.client,MODEL,'demo-v1',2,8)['quote'],idem)['job']
    def settle(self):
        j=self.job()
        for worker,key in self.keys.items():
            a=self.s.assignments(self.t[worker])[0];body=receipt_body(a,4)
            self.s.receipt(self.t[worker],body,key.sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex())
        for r in self.s.pending(self.v):
            b=json.loads(r['body']);e=observed({**b,'first_layer':r['first_layer'],'end_layer':r['end_layer']},4)
            self.s.validate(self.v,r['assignment'],r['digest'],True,4,e)
        self.s.settle(self.client,j);return j
    def test_only_opted_in_verified_settled_work(self):
        self.assertFalse(self.s.contributor_profile(self.a)['opted_in'])
        self.s.set_contributor_profile(self.a,True,'alias-a');self.assertEqual(self.s.leaderboard(self.client),[])
        self.settle();self.assertEqual(self.s.leaderboard(self.client),[{'alias':'alias-a','verified_jobs':1}])
        j=self.job('cancelled');self.s.cancel(self.client,j);self.assertEqual(self.s.leaderboard(self.client)[0]['verified_jobs'],1)
        self.s.set_contributor_profile(self.b,True,'alias-b');self.assertEqual(len(self.s.leaderboard(self.client)),2)
        self.s.set_contributor_profile(self.a,False,'');self.assertEqual(self.s.leaderboard(self.client),[{'alias':'alias-b','verified_jobs':1}])
    def test_alias_uniqueness_and_scope(self):
        self.s.set_contributor_profile(self.a,True,'alias-a')
        with self.assertRaises(Denied):self.s.set_contributor_profile(self.b,True,'alias-a')
        for alias in ('<script>','Alice@Example.com','x','Xyz'):
            with self.assertRaises(Denied):self.s.set_contributor_profile(self.b,True,alias)
        with self.assertRaises(Denied):self.s.set_contributor_profile(self.client,True,'alias-c')
        self.settle();other=self.s.bootstrap('other','op2',100);new=self.s.add_member(other,'other-worker','worker');self.s.set_pool(other,True);self.s.consent(new,'testnet-v1',True)
        self.s.set_contributor_profile(new,True,'alias-a');self.assertEqual(self.s.leaderboard(new),[])
        self.s.set_pool(self.op,False);self.assertFalse(self.s.set_contributor_profile(self.a,False,'')['opted_in'])
    def test_disputes_private_and_idempotent(self):
        j=self.settle();d=self.s.dispute(self.client,j,'incorrect_charge');self.assertEqual(d,self.s.dispute(self.client,j,'incorrect_charge'))
        self.assertEqual(self.s.disputes(self.a),[]);self.assertEqual(self.s.disputes(self.client),[{'id':d,'job':j,'reason':'incorrect_charge','state':'open','compensation':0}])
        self.s.review_dispute(self.t['reviewer'],d,0);self.assertEqual(self.s.disputes(self.client)[0]['state'],'rejected')

if __name__=='__main__':unittest.main()
