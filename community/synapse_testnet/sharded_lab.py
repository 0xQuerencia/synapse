"""Trusted local sharded native accounting lab for the original tiny fixture.

The coordinator's process harness records actual token delivery and resident
ranges. Operator-generated receipt keys bind those observations to local fake
credit assignments; they do not prove remote computation or confer real value.
No arbitrary prompts, external peers, production billing or partial payment.
"""
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .ledger import canonical, digest, require, sha, Denied
from .native_lab import LAB_MODEL, ARTIFACTS, INPUT, OUTPUT, LAYERS, MICRO, PROMPT_DIGEST
from .service import Service

EXPECTED_IDS=(45,20,61,20,46,8,60,34)


def validate_run(run):
    """Validate actual local harness evidence before authorizing any reward."""
    require(run.get('success') is True and run.get('reference_parity') is True
            and run.get('topology')=='loopback' and run.get('physical_machine_count')==1
            and run.get('worker_process_count')==2 and run.get('transport') in ('quic','tcp')
            and run.get('actual_output_tokens')==OUTPUT and run.get('token_ids')==list(EXPECTED_IDS),
            'NATIVE_SHARDED_PARITY_FAILED')
    expected_manifest={'schema_version':1,'model_id':'synapse/tiny-llama-fixture',
                       'revision':'fixture-v1','weights':'model.safetensors',
                       'artifacts':[{'filename':n,'size_bytes':size,'sha256':checksum}
                                    for n,(size,checksum) in ARTIFACTS.items()]}
    require(run.get('model')==expected_manifest,'NATIVE_FIXTURE_MISMATCH')
    binary=digest(run['binary_sha256']); peers=run['peer_ids']
    require(isinstance(peers,list) and len(peers)==3 and all(isinstance(p,str) and 1<=len(p)<=128 for p in peers)
            and len(set(peers))==3,'NATIVE_SHARDED_IDENTITY_MISMATCH')
    route=run['route'];hops=route['worker_hops']
    require(route['master_layer_range']=={'start':0,'end':0} and len(hops)==2,'NATIVE_SHARDED_ROUTE_MISMATCH')
    require({h['peer_id'] for h in hops}==set(peers[1:]),'NATIVE_SHARDED_IDENTITY_MISMATCH')
    require([h['layer_range'] for h in hops]==[{'start':0,'end':1},{'start':1,'end':2}], 'NATIVE_SHARDED_ROUTE_MISMATCH')
    ranges={};model_digests=set()
    for index,peer in enumerate(peers[1:],start=1):
        resident=run['worker_residency'][f'worker-{index}']
        hop=next(h for h in hops if h['peer_id']==peer)
        require(resident.get('role')=='worker' and resident.get('owned_layer_range')==hop['layer_range']
                and type(resident.get('authenticated_count')) is int and resident['authenticated_count']>=1
                and type(resident.get('resident_weight_bytes')) is int and resident['resident_weight_bytes']>0
                and type(resident.get('memory_budget_bytes')) is int
                and resident['resident_weight_bytes']<=resident['memory_budget_bytes'], 'NATIVE_SHARDED_RESIDENCY_MISMATCH')
        model_digests.add(digest(resident['model_digest']))
        r=hop['layer_range'];ranges[r['start']]={'first_layer':r['start'],'end_layer':r['end'],
            'peer_hash':sha(peer.encode()),'resident_weight_bytes':resident['resident_weight_bytes'],
            'memory_budget_bytes':resident['memory_budget_bytes']}
    require(len(model_digests)==1,'NATIVE_SHARDED_MODEL_MISMATCH')
    return {'model':LAB_MODEL,'native_model_digest':model_digests.pop(),'binary_sha256':binary,
            'prompt_sha256':PROMPT_DIGEST,'output_token_ids_sha256':sha(canonical(list(EXPECTED_IDS))),
            'input_units':INPUT,'output_units':OUTPUT,'transport':run['transport'],
            'stages':[ranges[i] for i in range(LAYERS)],'coordinator_transformer_layers':0}


def receipt_evidence(assignment,run_proof,stage):
    return sha(canonical(['synapse-native-sharded-known-answer-v1',assignment['assignment'],
                          assignment['job'],assignment['epoch'],run_proof,stage]))


class ShardedLabService(Service):
    def __init__(self,path,*,isolated_lab=False):
        require(isolated_lab is True and not Path(path).exists() and not Path(path).is_symlink(), 'NEW_ISOLATED_LAB_REQUIRED')
        self.proof=None
        super().__init__(path)

    def approve_fixture(self,token):
        self._approve_fixture_model(token,LAB_MODEL,LAYERS,128,'native_sharded_lab_fixture_approved')

    def _reference_evidence(self,body,stage):
        require(self.proof is not None and body['model']==LAB_MODEL and body['epoch']==1
                and stage['input_units']==INPUT and stage['output_units']==OUTPUT
                and stage['first_layer'] in (0,1) and stage['end_layer']==stage['first_layer']+1,
                'NATIVE_SHARDED_ASSIGNMENT_SCOPE')
        return receipt_evidence(body,self.proof,self.proof['stages'][stage['first_layer']])


def run_sharded_lab(path,run_native):
    """Reserve before invoking trusted process harness; only completed parity earns.

    `run_native` is an in-process operator callback, never remote/user input.
    The disposable ledger is separate from the standard synthetic service.
    """
    service=ShardedLabService(path,isolated_lab=True)
    op=service.bootstrap('native-sharded','operator',100*MICRO)
    client=service.add_member(op,'requester','requester');verifier=service.add_member(op,'verifier','verifier')
    workers={f'worker-{i}':service.add_member(op,f'worker-{i}','worker') for i in range(LAYERS)}
    service.approve_fixture(op);service.tariff(op,'native-sharded-v1',2*MICRO,MICRO,2*MICRO)
    service.grant(op,'requester',100*MICRO);service.set_pool(op,True)
    for credential in (client,verifier,*workers.values()):service.consent(credential,'testnet-v1',True)
    keys={}
    for i,(worker,token) in enumerate(workers.items()):
        key=Ed25519PrivateKey.generate();keys[worker]=key;device=worker+'-device'
        service.approve_device(op,device,worker,key.public_key().public_bytes_raw().hex(),LAB_MODEL,i,i+1,1)
        challenge=service.device_challenge(token,device)
        service.prove_device(token,challenge,key.sign(b'synapse-testnet-device-v1\0'+canonical(challenge)).hex())
    q=service.quote(client,LAB_MODEL,'native-sharded-v1',INPUT,OUTPUT)
    job=service.reserve(client,q['quote'],'native-sharded-job')
    reserved=service.balance(client)['held'];require(reserved==q['maximum'],'NATIVE_SHARDED_RESERVATION_FAILED')
    report={'scope':'trusted local native process lab; fake credits only','production_enabled':False,
            'physical_wan_verified':False,'reserved_microcredits':reserved,'parity':False}
    try:
        run=run_native()
        try:service.proof=validate_run(run)
        except (Denied,ValueError,TypeError,KeyError,IndexError) as e:
            report['rejection']=e.code if isinstance(e,Denied) else 'INVALID_NATIVE_EVIDENCE'
            report['native_fault_observed']=(run.get('injected_fault') or {}).get('kind')=='worker_terminated' if isinstance(run,dict) else False
            report['partial_output_contiguous']=run.get('partial_output_contiguous') is True if isinstance(run,dict) else False
            service.cancel(client,job['job'])
        else:
            report['native_evidence']=service.proof
            for i,(worker,token) in enumerate(workers.items()):
                assignment,=service.assignments(token)
                require(assignment['job']==job['job'] and assignment['first_layer']==i and assignment['end_layer']==i+1,
                        'NATIVE_SHARDED_ASSIGNMENT_SCOPE')
                body={'assignment':assignment['assignment'],'job':job['job'],'model':LAB_MODEL,
                      'epoch':assignment['epoch'],'input_units':INPUT,'output_units':OUTPUT,
                      'evidence':receipt_evidence(assignment,service.proof,service.proof['stages'][i])}
                service.receipt(token,body,keys[worker].sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex())
            import json
            pending=service.pending(verifier);require(len(pending)==LAYERS,'NATIVE_SHARDED_RECEIPT_COUNT')
            for row in pending:
                body=json.loads(row['body']);stage=service.proof['stages'][row['first_layer']]
                expected=receipt_evidence(body,service.proof,stage)
                require(service.validate(verifier,row['assignment'],row['digest'],True,OUTPUT,expected)=='verified',
                        'NATIVE_SHARDED_RECEIPT_REJECTED')
            settled=service.settle(client,job['job'])
            require(service.settle(client,job['job'])==settled,'NATIVE_LAB_DUPLICATE_SETTLEMENT')
            report['parity']=True
    finally:
        if service.job(client,job['job'])['state']=='running':service.cancel(client,job['job'])
    resolved=service.job(client,job['job'])
    with service.read() as db:service.reconcile(db)
    report.update({'job_state':resolved['state'],'charged_microcredits':resolved['charged'],
                   'requester_balance':service.balance(client),
                   'worker_balances':{name:service.balance(token) for name,token in workers.items()}})
    return report
