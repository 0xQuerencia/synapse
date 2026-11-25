"""Disposable, operator-trusted native known-answer accounting experiment.

Only the original tiny-Llama fixture, fixed prompt and greedy eight-token run
are admitted. This is not a remote proof of computation, a hosted API, a real
reward, an arbitrary model gateway, or a production settlement mechanism.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .ledger import canonical, digest, require, sha
from .native_probe import probe
from .service import Service

ARTIFACTS = {
    'config.json': (413, '6d3185f2b5726c4306287051e7741bf95510b0e4c059342491d345a0837988b3'),
    'model.safetensors': (29000, '947e138695eacd8da8c1e8a27748a3d3182fb660d440756fc6611b03f0621d46'),
    'tokenizer.json': (2183, '1dc60e019bcf944a9e64c484e7887d60e62872412dc18f14ebc61b77364644b5'),
}
LAB_MODEL = sha(canonical(['synapse/native-lab-v1', ARTIFACTS]))
PROMPT_DIGEST = sha(b'<s> </s> Hello')
API_MODEL = 'tiny'
INPUT, OUTPUT, LAYERS = 3, 8, 2
MICRO = 1_000_000

@dataclass(frozen=True)
class Endpoint:
    port: int
    token_file: Path


def verify_fixture(directory):
    """Only allow the original fixture bytes; no download or third-party data."""
    manifest_path=Path(directory)/'manifest.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink() and 0<manifest_path.stat().st_size<=4096,
            'NATIVE_FIXTURE_MISMATCH')
    expected_manifest={'schema_version':1,'model_id':'synapse/tiny-llama-fixture',
                       'revision':'fixture-v1','weights':'model.safetensors',
                       'artifacts':[{'filename':name,'size_bytes':size,'sha256':checksum}
                                    for name,(size,checksum) in ARTIFACTS.items()]}
    require(json.loads(manifest_path.read_bytes())==expected_manifest,'NATIVE_FIXTURE_MISMATCH')
    for name, (size, expected) in ARTIFACTS.items():
        path = Path(directory) / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == size,
                'NATIVE_FIXTURE_MISMATCH')
        require(sha(path.read_bytes()) == expected, 'NATIVE_FIXTURE_MISMATCH')
    return LAB_MODEL


def result_fields(result):
    """Bind known-answer output and exact server-reported token usage."""
    require(result['model'] == API_MODEL and result['prompt_fixture_sha256'] == PROMPT_DIGEST
            and type(result['prompt_tokens']) is int and result['prompt_tokens'] == INPUT
            and type(result['completion_tokens']) is int and result['completion_tokens'] == OUTPUT
            and result['finish_reason'] == 'length' and result['credits_charged'] == 0,
            'NATIVE_FIXTURE_RESULT_MISMATCH')
    return {'model': LAB_MODEL, 'prompt': PROMPT_DIGEST,
            'output': digest(result['output_sha256']), 'input_units': INPUT,
            'output_units': OUTPUT, 'finish_reason': 'length'}


def evidence(assignment, result):
    require(assignment['model'] == LAB_MODEL and assignment['input_units'] == INPUT
            and assignment['epoch'] == 1, 'NATIVE_LAB_ASSIGNMENT_SCOPE')
    return sha(canonical(['synapse-native-known-answer-v1', assignment['assignment'],
                          assignment['job'], assignment['epoch'], result]))


class NativeLabService(Service):
    """A local disposable instance; normal Service/HTTP remain synthetic-only."""
    def __init__(self, path, reference, *, isolated_lab=False):
        require(isolated_lab is True and not Path(path).exists() and not Path(path).is_symlink(),
                'NEW_ISOLATED_LAB_REQUIRED')
        self.reference = result_fields(reference)
        super().__init__(path)

    def approve_fixture(self, token):
        self._approve_fixture_model(token, LAB_MODEL, LAYERS, 128, 'native_lab_fixture_approved')

    def _reference_evidence(self, body, stage):
        require(stage['first_layer'] == 0 and stage['end_layer'] == LAYERS
                and stage['input_units'] == INPUT and stage['output_units'] == OUTPUT,
                'NATIVE_LAB_ASSIGNMENT_SCOPE')
        return evidence(body, self.reference)


def run_lab(path, candidate, reference):
    """Two distinct locally launched endpoints; only temporary fake credits.

    The reference endpoint is trusted by the operator, not a decentralized
    verifier. Matching output demonstrates metering/settlement integration only.
    """
    require(candidate.port != reference.port, 'DISTINCT_NATIVE_ENDPOINTS_REQUIRED')
    ref = probe(reference.port, reference.token_file, API_MODEL, OUTPUT)
    service = NativeLabService(path, ref, isolated_lab=True)
    op = service.bootstrap('native-lab','operator',100*MICRO)
    client = service.add_member(op,'requester','requester')
    worker = service.add_member(op,'worker','worker')
    verifier = service.add_member(op,'verifier','verifier')
    service.approve_fixture(op)
    service.tariff(op,'native-lab-v1',2*MICRO,MICRO,2*MICRO)
    service.grant(op,'requester',100*MICRO)
    service.set_pool(op,True)
    for credential in (client,worker,verifier): service.consent(credential,'testnet-v1',True)
    key = Ed25519PrivateKey.generate()
    service.approve_device(op,'native-device','worker',key.public_key().public_bytes_raw().hex(),LAB_MODEL,0,LAYERS,1)
    challenge = service.device_challenge(worker,'native-device')
    service.prove_device(worker,challenge,key.sign(b'synapse-testnet-device-v1\0'+canonical(challenge)).hex())
    quote = service.quote(client,LAB_MODEL,'native-lab-v1',INPUT,OUTPUT)
    job = service.reserve(client,quote['quote'],'native-lab-job')
    try:
        assignment, = service.assignments(worker)
        require(assignment['job'] == job['job'] and assignment['max_output'] == OUTPUT,
                'NATIVE_LAB_ASSIGNMENT_SCOPE')
        actual = result_fields(probe(candidate.port,candidate.token_file,API_MODEL,OUTPUT))
        body = {'assignment':assignment['assignment'],'job':job['job'],'model':LAB_MODEL,
                'epoch':assignment['epoch'],'input_units':INPUT,'output_units':OUTPUT,
                'evidence':evidence(assignment,actual)}
        signature = key.sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex()
        receipt = service.receipt(worker,body,signature)
        row, = service.pending(verifier)
        expected = evidence(assignment,service.reference)
        status = service.validate(verifier,row['assignment'],receipt['digest'],True,OUTPUT,expected)
        if status == 'verified':
            resolved = service.settle(client,job['job'])
            require(service.settle(client,job['job']) == resolved,'NATIVE_LAB_DUPLICATE_SETTLEMENT')
        else:
            resolved = service.job(client,job['job'])
        with service.read() as db: service.reconcile(db)
        return {'scope':'operator-trusted original native fixture lab; fake credits only',
                'production_enabled':False,'physical_wan_verified':False,'full_model_only':True,
                'model_fingerprint':LAB_MODEL,'parity':status=='verified','reference':service.reference,
                'candidate':actual,'receipt_status':status,'reserved_microcredits':quote['maximum'],
                'job_state':resolved['state'],'charged_microcredits':resolved['charged'],
                'requester_balance':service.balance(client),'worker_balance':service.balance(worker)}
    finally:
        # Any HTTP/native exception or failed validation must not strand funds.
        if service.job(client,job['job'])['state']=='running':
            service.cancel(client,job['job'])
