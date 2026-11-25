"""Contract-double tests only; the native Rust engine is unavailable in this session."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
from synapse_testnet.native_probe import probe
from synapse_testnet.ledger import Denied

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        assert self.path=='/v1/chat/completions'; assert self.headers['Authorization']=='Bearer '+'x'*32
        body=json.loads(self.rfile.read(int(self.headers['Content-Length']))); assert body['temperature']==0 and body['stream'] is False
        response={'model':body['model'],'choices':[{'message':{'content':'synthetic output'},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4,'total_tokens':7 if not self.server.bad else 8}}
        data=json.dumps(response).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(data)
class ProbeTests(unittest.TestCase):
    def test_native_contract(self):
        with tempfile.TemporaryDirectory() as d,HTTPServer(('127.0.0.1',0),Handler) as s:
            p=Path(d)/'token'; p.write_text('x'*32); p.chmod(0o600); s.bad=False; t=threading.Thread(target=s.serve_forever); t.start()
            try:
                r=probe(s.server_port,p,'fixture'); self.assertEqual(r['completion_tokens'],4); self.assertEqual(r['credits_charged'],0); self.assertNotIn('synthetic output',str(r)); s.bad=True
                with self.assertRaisesRegex(Denied,'NATIVE_USAGE_MISMATCH'): probe(s.server_port,p,'fixture')
            finally:s.shutdown();t.join()
