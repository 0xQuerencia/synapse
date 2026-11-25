#!/usr/bin/env python3
"""Serve the unmodified GUI assets with visibly labelled layout-test state.

This is an isolated UI fixture. It neither starts Synapse nor reads credentials,
serves private files, or submits a request to an inference process. Real HTTP,
identity, stream and worker-control checks live in gui_smoke.py.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import json
import os

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'crates/synapse-gui/web'
STATE = {'peer_id':'layout-test-device', 'cached_models':['synapse/tiny-llama-fixture'],
         'bootstrap_count':0, 'relay_count':0, 'node':None, 'api_model':None,
         'session':{'role':'idle','running':False,'peer_id':None,'model_id':None,'pairing_code':None,
         'max_tokens':None,'listen_addr':None,'generating':False,'last_error':None,'drain_at_unix':None,'backend':'none'}}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def reply(self, code, body, content_type='application/json'):
        self.send_response(code); self.send_header('Content-Type',content_type); self.send_header('Cache-Control','no-store')
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/status': return self.reply(200,json.dumps(STATE).encode())
        if path == '/viewport':
            width = parse_qs(urlsplit(self.path).query).get('width',['360'])[0]
            if width not in ['360','768','1280']: return self.reply(400,b'{}')
            return self.reply(200,('''<!doctype html><meta charset="utf-8"><title>Synapse layout QA</title><style>body{margin:0;background:#292b35;color:white;font:14px Arial}p{margin:12px}iframe{display:block;border:1px solid #777;margin:0 auto;height:2000px;max-width:100%}</style><p>Layout QA fixture · '''+width+'''px viewport · no live session</p><iframe title="Synapse device workspace layout fixture" src="/" width="'''+width+'''"></iframe>''').encode(),'text/html')
        if path in ['/','/contribute','/index.html']:
            html=(ASSETS/'index.html').read_text()
            html=html.replace('<body>','<body><div style="padding:10px 20px;background:#342647;color:#fff;text-align:center;font:14px Arial">UI layout fixture · no live session or credentials</div>')
            return self.reply(200,html.encode(),'text/html')
        names = {'/workspace.css':('workspace.css','text/css'),'/workspace.js':('workspace.js','text/javascript')}
        if path in names:
            name,kind=names[path]; return self.reply(200,(ASSETS/name).read_bytes(),kind)
        self.reply(404,b'{"error":"not found in layout fixture"}')
    def do_POST(self):
        self.reply(409,b'{"error":"Layout fixture only; use the authenticated native app to run a session."}')

if __name__ == '__main__':
    port=int(os.environ.get('PORT','4173'))
    print(f'Labelled GUI layout fixture on port {port}',flush=True)
    ThreadingHTTPServer(('0.0.0.0',port),Handler).serve_forever()
