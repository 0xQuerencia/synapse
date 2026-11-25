"""Shape and accounting contract tests; real process proof is exercised in CI."""
import copy
import tempfile
import unittest
from pathlib import Path
from synapse_testnet.sharded_lab import validate_run, run_sharded_lab, EXPECTED_IDS
from synapse_testnet.native_lab import ARTIFACTS
from synapse_testnet.ledger import Denied


def evidence():
    peers=['host','worker-one','worker-two']
    ranges=[{'start':0,'end':1},{'start':1,'end':2}]
    return {'success':True,'reference_parity':True,'topology':'loopback','physical_machine_count':1,
            'worker_process_count':2,'transport':'quic','actual_output_tokens':8,'token_ids':list(EXPECTED_IDS),
            'binary_sha256':'a'*64,'peer_ids':peers,
            'model':{'schema_version':1,'model_id':'synapse/tiny-llama-fixture','revision':'fixture-v1','weights':'model.safetensors',
                     'artifacts':[{'filename':n,'size_bytes':size,'sha256':checksum} for n,(size,checksum) in ARTIFACTS.items()]},
            'route':{'master_layer_range':{'start':0,'end':0},'worker_hops':[{'peer_id':p,'layer_range':r} for p,r in zip(peers[1:],ranges)]},
            'worker_residency':{f'worker-{i+1}':{'role':'worker','owned_layer_range':r,'authenticated_count':1,
                'resident_weight_bytes':11392,'memory_budget_bytes':45000,'model_digest':'b'*64} for i,r in enumerate(ranges)}}

class ShardedLabTests(unittest.TestCase):
    def test_verified_two_stage_settlement_conserves_credits(self):
        with tempfile.TemporaryDirectory() as d:
            result=run_sharded_lab(Path(d)/'db',evidence)
            self.assertTrue(result['parity']);self.assertEqual(result['job_state'],'settled')
            self.assertEqual(result['charged_microcredits'],21_000_000)
            self.assertEqual(result['requester_balance']['available'],79_000_000)
            self.assertEqual([r['available'] for r in result['worker_balances'].values()],[10_500_000,10_500_000])
            self.assertFalse(result['production_enabled'])

    def test_failed_or_invalid_process_refunds(self):
        for change in ('failed','wrong-tokens','wrong-route','wrong-residency','wrong-model'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                run=evidence()
                if change=='failed':run['success']=False;run['injected_fault']={'kind':'worker_terminated'};run['partial_output_contiguous']=True
                if change=='wrong-tokens':run['token_ids'][-1]=0
                if change=='wrong-route':run['route']['worker_hops'][1]['layer_range']={'start':0,'end':2}
                if change=='wrong-residency':run['worker_residency']['worker-1']['resident_weight_bytes']=100_000
                if change=='wrong-model':run['model']['revision']='other'
                result=run_sharded_lab(Path(d)/'db',lambda:run)
                self.assertFalse(result['parity']);self.assertEqual(result['charged_microcredits'],0)
                self.assertEqual(result['requester_balance']['available'],100_000_000)
                self.assertEqual([r['available'] for r in result['worker_balances'].values()],[0,0])

    def test_distinct_peers_and_model_digest_required(self):
        run=evidence();run['peer_ids'][2]=run['peer_ids'][1]
        with self.assertRaisesRegex(Denied,'NATIVE_SHARDED_IDENTITY_MISMATCH'):validate_run(run)
        run=evidence();run['worker_residency']['worker-2']['model_digest']='c'*64
        with self.assertRaisesRegex(Denied,'NATIVE_SHARDED_MODEL_MISMATCH'):validate_run(run)

    def test_route_order_independent_of_worker_process_order(self):
        run=evidence();run['peer_ids'][1:]=reversed(run['peer_ids'][1:])
        run['worker_residency']['worker-1'],run['worker_residency']['worker-2']=run['worker_residency']['worker-2'],run['worker_residency']['worker-1']
        self.assertEqual(validate_run(run)['stages'][0]['first_layer'],0)
