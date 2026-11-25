import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from synapse_testnet.ledger import Denied
from synapse_testnet.native_lab import Endpoint, LAB_MODEL, run_lab, verify_fixture, NativeLabService
from synapse_testnet.service import Service

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_POST(self):
        assert self.path=='/v1/chat/completions'
        assert self.headers['Authorization']=='Bearer '+'x'*32
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert body['model']=='tiny' and body['max_tokens']==8 and body['temperature']==0
        self.server.calls+=1
        result={'model':'tiny','choices':[{'message':{'content':self.server.output},'finish_reason':'length'}],
                'usage':{'prompt_tokens':3,'completion_tokens':8,'total_tokens':11}}
        payload=json.dumps(result).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)

class NativeLabTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.endpoints=[];self.servers=[]
        for name in ('candidate','reference'):
            server=HTTPServer(('127.0.0.1',0),Handler);server.calls=0;server.output='fixture output'
            t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
            def cleanup(server=server,t=t): server.shutdown();t.join(5);server.server_close()
            self.addCleanup(cleanup)
            token=self.root/name;token.write_text('x'*32);token.chmod(0o600)
            self.endpoints.append(Endpoint(server.server_port,token));self.servers.append(server)

    def test_equal_results_settle_once(self):
        result=run_lab(self.root/'ledger.db',*self.endpoints)
        self.assertTrue(result['parity']);self.assertEqual(result['job_state'],'settled')
        self.assertEqual(result['charged_microcredits'],21_000_000)
        self.assertEqual(result['worker_balance']['available'],21_000_000)
        self.assertEqual(result['requester_balance']['available'],79_000_000)
        self.assertEqual([s.calls for s in self.servers],[1,1])
        self.assertFalse(result['production_enabled']);self.assertNotIn('fixture output',str(result))

    def test_mismatched_native_output_refunds(self):
        self.servers[0].output='forged result'
        result=run_lab(self.root/'ledger.db',*self.endpoints)
        self.assertFalse(result['parity']);self.assertEqual(result['job_state'],'failed')
        self.assertEqual(result['charged_microcredits'],0)
        self.assertEqual(result['requester_balance']['available'],100_000_000)
        self.assertEqual(result['worker_balance']['available'],0)

    def test_same_endpoint_refused(self):
        with self.assertRaisesRegex(Denied,'DISTINCT_NATIVE_ENDPOINTS_REQUIRED'):
            run_lab(self.root/'ledger.db',self.endpoints[0],self.endpoints[0])
        self.assertFalse((self.root/'ledger.db').exists())

    def test_untrusted_fixture_and_normal_service_refuse_native_model(self):
        with self.assertRaisesRegex(Denied,'NATIVE_FIXTURE_MISMATCH'):verify_fixture(self.root)
        service=Service(self.root/'ordinary.db');op=service.bootstrap('ordinary','operator',100)
        with self.assertRaisesRegex(Denied,'TESTNET_SYNTHETIC_MODELS_ONLY'):
            service.approve_model(op,LAB_MODEL,2,128,True)
        with self.assertRaisesRegex(Denied,'NEW_ISOLATED_LAB_REQUIRED'):
            NativeLabService(self.root/'ordinary.db',{},isolated_lab=True)
