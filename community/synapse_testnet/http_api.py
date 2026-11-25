"""Loopback-only HTTP transport for the isolated testnet. Never expose through a tunnel."""
import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from .ledger import Denied, require
from .service import Service

MAX_BODY=32768
ROUTES={
 'POST /v1/consent': ('consent', ('version','external_compute')),
 'POST /v1/quotes': ('quote', ('model','tariff','input_units','max_output')),
 'POST /v1/jobs': ('reserve', ('quote','idem')),
 'POST /v1/jobs/status': ('job', ('job',)),
 'POST /v1/jobs/cancel': ('cancel', ('job',)),
 'POST /v1/jobs/settle': ('settle', ('job',)),
 'GET /v1/credits/balance': ('balance', ()),
 'GET /v1/credits/history': ('history', ()),
 'GET /v1/contributor/stats': ('stats', ()),
 'GET /v1/worker/assignments': ('assignments', ()),
 'POST /v1/devices/challenge': ('device_challenge', ('device',)),
 'POST /v1/devices/prove': ('prove_device', ('body','signature')),
 'POST /v1/receipts': ('receipt', ('body','signature')),
 'GET /v1/verifier/pending': ('pending', ()),
 'POST /v1/verifier/validate': ('validate', ('assignment','receipt_digest','accepted','observed_output','observed_evidence')),
 'POST /v1/disputes': ('dispute', ('job','reason')),
 'POST /v1/verifier/dispute': ('review_dispute', ('dispute','compensation')),
 'POST /v1/operator/dispute': ('resolve_dispute', ('dispute',)),
 'POST /v1/operator/pool': ('set_pool', ('enabled',)),
 'POST /v1/operator/grant': ('grant', ('member','amount')),
 'POST /v1/operator/revoke': ('revoke', ('target','kind')),
 'POST /v1/operator/model/disable': ('disable_model', ('model',)),
}

def strict_pairs(pairs):
    result={}
    for k,v in pairs:
        require(k not in result,'DUPLICATE_JSON_KEY')
        result[k]=v
    return result

class Server(ThreadingHTTPServer):
    daemon_threads=True
    request_queue_size=16
    def __init__(self, service, port=0):
        self.service=service
        self.slots=threading.BoundedSemaphore(16)
        super().__init__(('127.0.0.1',port), Handler)
    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            try: request.sendall(b'HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n')
            finally: self.shutdown_request(request)
            return
        try: super().process_request(request,address)
        except BaseException:
            self.slots.release(); raise
    def process_request_thread(self, request, address):
        try: super().process_request_thread(request,address)
        finally: self.slots.release()
    def get_request(self):
        sock,addr=super().get_request(); sock.settimeout(5); return sock,addr

class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.0'
    server_version='SynapseTestnet/0.1'
    sys_version=''
    def log_message(self,*args): pass  # never log request paths, headers, credentials or bodies
    def do_GET(self): self.dispatch()
    def do_POST(self): self.dispatch()
    def do_OPTIONS(self): self.reply(403,{'error':'ORIGIN_DENIED'})
    def reply(self,status,data):
        raw=json.dumps(data,separators=(',',':'),allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.send_header('Connection','close'); self.end_headers(); self.wfile.write(raw)
    def dispatch(self):
        try:
            host=f'127.0.0.1:{self.server.server_port}'
            require(self.headers.get_all('Host')==[host] and not self.headers.get('Origin') and not self.headers.get('Forwarded') and not self.headers.get('X-Forwarded-Host'),'ORIGIN_DENIED',403)
            path=urlsplit(self.path)
            require(path.path==self.path and not path.query and not path.fragment,'INVALID_PATH')
            require(not self.headers.get('Transfer-Encoding'),'TRANSFER_ENCODING_UNSUPPORTED')
            if self.command=='GET' and path.path=='/health/live':
                return self.reply(200,{'status':'alive','mode':'local_testnet','real_credits':False})
            route=ROUTES.get(self.command+' '+path.path)
            require(route is not None,'NOT_FOUND',404)
            auth=self.headers.get_all('Authorization') or []
            require(len(auth)==1 and auth[0].startswith('Bearer '),'AUTH_REQUIRED',401)
            token=auth[0][7:]; body={}
            if self.command=='POST':
                require(self.headers.get('Content-Type')=='application/json','JSON_REQUIRED',415)
                lengths=self.headers.get_all('Content-Length') or []
                require(len(lengths)==1 and lengths[0].isascii() and lengths[0].isdigit() and len(lengths[0])<=6,'INVALID_LENGTH')
                length=int(lengths[0]); require(0<length<=MAX_BODY,'BODY_LIMIT',413)
                raw=self.rfile.read(length); require(len(raw)==length,'TRUNCATED_BODY')
                body=json.loads(raw,object_pairs_hook=strict_pairs,parse_constant=lambda _: (_ for _ in ()).throw(Denied('INVALID_JSON')))
            name,fields=route
            require(isinstance(body,dict) and set(body)==set(fields),'INVALID_FIELDS')
            result=getattr(self.server.service,name)(token,**body)
            self.reply(200,{'result':result,'testnet':True})
        except Denied as e: self.reply(e.status,{'error':e.code})
        except (ValueError,TypeError,OverflowError,RecursionError,KeyError): self.reply(400,{'error':'INVALID_REQUEST'})
        except (BrokenPipeError,ConnectionResetError,TimeoutError): return
        except Exception:
            logging.error('testnet request failed; inspect local database integrity without logging request bodies')
            self.reply(503,{'error':'SERVICE_UNAVAILABLE'})

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--db',required=True); p.add_argument('--port',type=int,default=8787); p.add_argument('--testnet',action='store_true',required=True); args=p.parse_args()
    s=Service(args.db); s.sweep()
    with Server(s,args.port) as server:
        stop=threading.Event()
        def sweeper():
            while not stop.wait(2):
                try: s.sweep()
                except Exception: logging.error('testnet sweep failed; new admissions require reconciliation')
        t=threading.Thread(target=sweeper,daemon=True); t.start()
        print(f'SYNAPSE TESTNET ONLY: http://127.0.0.1:{server.server_port}',flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally: stop.set(); t.join(6)
if __name__=='__main__': main()
