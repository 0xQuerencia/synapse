"""Execute an original synthetic testnet job via actual loopback HTTP and signed receipts."""
import argparse
import json
import tempfile
import threading
import urllib.request
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .ledger import canonical
from .service import Service
from .http_api import Server
from .synthetic import MODEL, receipt_body, observed

def setup(path, sharded=True):
    service=Service(path)
    operator=service.bootstrap('pilot','operator',100_000_000)
    tokens={r:service.add_member(operator,r,role) for r,role in [('client','requester'),('worker-a','worker'),('worker-b','worker'),('verifier','verifier'),('reviewer','verifier')]}
    tokens['operator']=operator
    service.approve_model(operator,MODEL,4,64,True)
    service.tariff(operator,'demo-v1',2_000_000,1_000_000,2_000_000)
    service.grant(operator,'client',100_000_000)
    service.set_pool(operator,True)
    for k,t in tokens.items(): service.consent(t,'testnet-v1',True)
    keys={}
    for name,start,end in [('worker-a',0,2 if sharded else 4),('worker-b',2,4)] if sharded else [('worker-a',0,4)]:
        key=Ed25519PrivateKey.generate(); keys[name]=key; device=name+'-device'
        service.approve_device(operator,device,name,key.public_key().public_bytes_raw().hex(),MODEL,start,end,1)
        body=service.device_challenge(tokens[name],device)
        service.prove_device(tokens[name],body,key.sign(b'synapse-testnet-device-v1\0'+canonical(body)).hex())
    return service,tokens,keys

def call(server, token, path, body=None):
    data=None if body is None else canonical(body)
    request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+path,data=data,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=8) as r:
        return json.load(r)['result']

def execute(path, sharded=True):
    service,tokens,keys=setup(path,sharded)
    with Server(service) as server:
        t=threading.Thread(target=server.serve_forever,daemon=True); t.start()
        try:
            def api(w,p,b=None): return call(server,tokens[w],p,b)
            q=api('client','/v1/quotes',{'model':MODEL,'tariff':'demo-v1','input_units':2,'max_output':8})
            job=api('client','/v1/jobs',{'quote':q['quote'],'idem':'demo-1'})
            held=api('client','/v1/credits/balance')
            for w,key in keys.items():
                for a in api(w,'/v1/worker/assignments'):
                    body=receipt_body(a,4); signature=key.sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex()
                    api(w,'/v1/receipts',{'body':body,'signature':signature})
            for row in api('verifier','/v1/verifier/pending'):
                body=json.loads(row['body']); reference=observed({**body,'first_layer':row['first_layer'],'end_layer':row['end_layer']},body['output_units'])
                api('verifier','/v1/verifier/validate',{'assignment':row['assignment'],'receipt_digest':row['digest'],'accepted':reference==body['evidence'],'observed_output':body['output_units'],'observed_evidence':reference})
            settled=api('client','/v1/jobs/settle',{'job':job['job']})
            retry=api('client','/v1/jobs/settle',{'job':job['job']}); assert retry==settled
            balances={w:api(w,'/v1/credits/balance') for w in ('client','worker-a','worker-b')}
            badges={w:api(w,'/v1/contributor/stats') for w in keys}
            # Worker spends its earned balance using an unrelated worker, not its own device.
            redemption=None
            if not sharded:
                key=Ed25519PrivateKey.generate(); device='worker-b-device'; op=tokens['operator']
                service.approve_device(op,device,'worker-b',key.public_key().public_bytes_raw().hex(),MODEL,0,4,1)
                challenge=service.device_challenge(tokens['worker-b'],device)
                service.prove_device(tokens['worker-b'],challenge,key.sign(b'synapse-testnet-device-v1\0'+canonical(challenge)).hex())
                q2=api('worker-a','/v1/quotes',{'model':MODEL,'tariff':'demo-v1','input_units':1,'max_output':1})
                j2=api('worker-a','/v1/jobs',{'quote':q2['quote'],'idem':'redemption-1'})
                a2=api('worker-b','/v1/worker/assignments')[0]; b2=receipt_body(a2,1)
                rr=api('worker-b','/v1/receipts',{'body':b2,'signature':key.sign(b'synapse-testnet-receipt-v1\0'+canonical(b2)).hex()})
                ref2=observed(a2,1)
                api('verifier','/v1/verifier/validate',{'assignment':a2['assignment'],'receipt_digest':rr['digest'],'accepted':True,'observed_output':1,'observed_evidence':ref2})
                redemption=api('worker-a','/v1/jobs/settle',{'job':j2['job']})
            with service.read() as db: service.reconcile(db)
            return {'mode':'synthetic HTTP testnet, not real AI','sharded':sharded,'quote_microcredits':q['maximum'],'held':held,'settled':settled,'balances':balances,'badges':badges,'redemption':redemption,'duplicate_settlement_equal':retry==settled}
        finally: server.shutdown(); t.join(5)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--full-model',action='store_true'); args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='synapse-fake-') as d: print(json.dumps(execute(Path(d)/'ledger.db',not args.full_model),indent=2))
if __name__=='__main__': main()
