"""Original deterministic synthetic computation; not an LLM or proof of LLM correctness."""
import hashlib
from .ledger import canonical, sha

MODEL=sha(b'synapse-original-testnet-synthetic-v1')

def observed(assignment, count):
    # A verifier can independently recompute the known-answer challenge, without user text.
    seed=canonical({k:assignment[k] for k in ('assignment','job','model','epoch','first_layer','end_layer')})
    state=hashlib.sha256(seed).digest()
    for step in range(count):
        for layer in range(assignment['first_layer'],assignment['end_layer']):
            state=hashlib.sha256(state+step.to_bytes(4,'big')+layer.to_bytes(4,'big')).digest()
    return state.hex()

def receipt_body(assignment,count):
    return {**{k:assignment[k] for k in ('assignment','job','model','epoch','input_units')},'output_units':count,'evidence':observed(assignment,count)}
