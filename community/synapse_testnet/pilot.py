"""Run a multi-process, loopback-only synthetic pilot. Credits are resettable and noncash."""
import argparse
import json
import os
import signal
import stat
import time
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from urllib.error import HTTPError
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .demo import setup
from .ledger import canonical, require, Denied, uid, integer
from .synthetic import MODEL, receipt_body, observed
from .ops import private_dir, exclusive

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

class Client:
    def __init__(self,port,token):
        integer(port,1,65535); self.url=f'http://127.0.0.1:{port}'; self.token=token; self.opener=build_opener(ProxyHandler({}),NoRedirect())
    def call(self,path,body=None):
        require(path.startswith('/v1/') and '?' not in path,'INVALID_PATH')
        req=Request(self.url+path,data=None if body is None else canonical(body),headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        try:
            with self.opener.open(req,timeout=5) as r:
                require(r.headers.get_content_type()=='application/json','UNEXPECTED_RESPONSE'); raw=r.read(1024*1024+1); require(len(raw)<=1024*1024,'RESPONSE_LIMIT'); return json.loads(raw)['result']
        except HTTPError as e:
            try: code=json.loads(e.read(4096)).get('error','HTTP_ERROR')
            except (ValueError,KeyError): code='HTTP_ERROR'
            raise Denied(code,e.code) from None

def write_private(path,value):
    with os.fdopen(exclusive(path),'wb') as f: f.write(canonical(value)+b'\n'); f.flush(); os.fsync(f.fileno())

def read_private(root,role):
    require(role in ('operator','client','worker-a','worker-b','verifier','reviewer'),'INVALID_ROLE')
    p=private_dir(root)/(role+'.json'); st=p.lstat(); require(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_mode&0o077==0 and st.st_size<=1024,'UNSAFE_CREDENTIAL_FILE')
    return json.loads(p.read_bytes())

def provision(root,sharded=True):
    root=private_dir(root); require(not any(root.iterdir()),'NEW_EMPTY_DIRECTORY_REQUIRED')
    s,tokens,keys=setup(root/'ledger.db',sharded)
    for role,token in tokens.items():
        payload={'token':token}
        if role in keys:
            payload['key']=keys[role].private_bytes_raw().hex(); payload['device']=role+'-device'
            # Workers must actually start and prove possession before becoming eligible.
            s.worker_control(token,payload['device'],'stop')
        write_private(root/(role+'.json'),payload)
    return {'testnet':True,'sharded':sharded,'database':str(root/'ledger.db'),'roles':list(tokens),'model':MODEL,'tariff':'demo-v1'}

def worker(root,role,port,stop):
    c=read_private(root,role); require('key' in c,'WORKER_NOT_PROVISIONED'); api=Client(port,c['token']); key=Ed25519PrivateKey.from_private_bytes(bytes.fromhex(c['key'])); device=c['device']
    challenge=api.call('/v1/devices/challenge',{'device':device}); proof=api.call('/v1/devices/prove',{'body':challenge,'signature':key.sign(b'synapse-testnet-device-v1\0'+canonical(challenge)).hex()}); seq=proof['heartbeat_sequence']; seen=set(); next_heartbeat=0
    print('worker ready (synthetic only)',flush=True)
    try:
        while not stop():
            now=time.monotonic()
            if now>=next_heartbeat:
                seq+=1; h={'device':device,'sequence':seq,'issued':int(time.time())}; api.call('/v1/devices/heartbeat',{'body':h,'signature':key.sign(b'synapse-testnet-heartbeat-v1\0'+canonical(h)).hex()}); next_heartbeat=now+5
            assignments=api.call('/v1/worker/assignments')
            seen.intersection_update(a['assignment'] for a in assignments)
            for a in assignments:
                if a['device']!=device or a['model']!=MODEL or a['assignment'] in seen: continue
                body=receipt_body(a,min(a['max_output'],4)); sig=key.sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex()
                try: api.call('/v1/receipts',{'body':body,'signature':sig}); seen.add(a['assignment'])
                except Denied as e:
                    if e.code!='LEASE_EXPIRED': raise
            time.sleep(.1)
    finally:
        try: api.call('/v1/worker/control',{'device':device,'action':'stop'})
        except (Denied,OSError): pass

def verifier(root,port,stop):
    api=Client(port,read_private(root,'verifier')['token']); print('verifier ready (synthetic only)',flush=True)
    while not stop():
        for row in api.call('/v1/verifier/pending'):
            body=json.loads(row['body']); ref=observed({**body,'first_layer':row['first_layer'],'end_layer':row['end_layer']},body['output_units'])
            try: api.call('/v1/verifier/validate',{'assignment':row['assignment'],'receipt_digest':row['digest'],'accepted':ref==body['evidence'],'observed_output':body['output_units'],'observed_evidence':ref})
            except Denied as e:
                if e.code!='LEASE_EXPIRED': raise
        time.sleep(.1)

def request(root,port,role='client',output=8,timeout=15):
    api=Client(port,read_private(root,role)['token']); before=api.call('/v1/credits/balance'); q=api.call('/v1/quotes',{'model':MODEL,'tariff':'demo-v1','input_units':2,'max_output':output}); end=time.monotonic()+timeout; idem=uid()
    while True:
        try: j=api.call('/v1/jobs',{'quote':q['quote'],'idem':idem}); break
        except Denied as e:
            if e.code!='NO_ROUTE' or time.monotonic()>=end: raise
            time.sleep(.2)
    held=api.call('/v1/credits/balance')
    while time.monotonic()<end:
        status=api.call('/v1/jobs/status',{'job':j['job']})
        if status['state']!='running': return {'before':before,'held':held,'job':status,'after':api.call('/v1/credits/balance')}
        try:
            settled=api.call('/v1/jobs/settle',{'job':j['job']}); return {'before':before,'held':held,'job':settled,'after':api.call('/v1/credits/balance')}
        except Denied as e:
            if e.code!='WORK_NOT_VERIFIED': raise
        time.sleep(.2)
    cancelled=api.call('/v1/jobs/cancel',{'job':j['job']}); return {'before':before,'held':held,'job':cancelled,'after':api.call('/v1/credits/balance')}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True); p.add_argument('--port',type=int,default=8787); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('init'); a.add_argument('--full-model',action='store_true'); a=sub.add_parser('worker'); a.add_argument('role',choices=['worker-a','worker-b']); sub.add_parser('verifier'); a=sub.add_parser('request'); a.add_argument('--role',default='client',choices=['client','worker-a','worker-b']); a.add_argument('--output',type=int,default=8); a=sub.add_parser('balance'); a.add_argument('role',choices=['client','worker-a','worker-b']); a=sub.add_parser('control'); a.add_argument('role',choices=['worker-a','worker-b']); a.add_argument('action',choices=['pause','resume','drain','stop'])
    args=p.parse_args(); flag=[False]
    def stop(*_): flag[0]=True
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    if args.cmd=='init': result=provision(args.root,not args.full_model)
    elif args.cmd=='worker': return worker(args.root,args.role,args.port,lambda:flag[0])
    elif args.cmd=='verifier': return verifier(args.root,args.port,lambda:flag[0])
    elif args.cmd=='request': result=request(args.root,args.port,args.role,args.output)
    else:
        c=read_private(args.root,args.role); api=Client(args.port,c['token']); result=api.call('/v1/credits/balance') if args.cmd=='balance' else api.call('/v1/worker/control',{'device':c['device'],'action':args.action})
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
