#!/usr/bin/env python3
"""Run real coordinator-only native sharding and account for its completed output.

The trusted local harness launches a reference, coordinator and two workers.
Only known-answer loopback fixture work is admitted. No remote payment or
physical network evidence is inferred from this run.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'community'))
from synapse_testnet.fixture_transfer import deliver_fixture
from synapse_testnet.sharded_lab import run_sharded_lab
from local_cluster import run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--transport',choices=['quic','tcp'],default='quic');p.add_argument('--interrupt-worker',action='store_true')
    args=p.parse_args()
    if args.interrupt_worker and args.transport!='tcp':p.error('fault injection uses delayed TCP')
    args.out.mkdir(parents=True,exist_ok=False)
    report={'status':'failed','transport':args.transport,'fault_injection_requested':args.interrupt_worker}
    try:
        with tempfile.TemporaryDirectory(prefix='synapse-sharded-accounting-') as tmp:
            root=Path(tmp);source=ROOT/'crates/synapse-core/tests/fixtures/tiny-llama'
            delivered,report['artifact_delivery']=deliver_fixture(source,root/'transfer')
            def actual():
                return run(argparse.Namespace(binary=args.binary,model_dir=str(delivered),output=str(root/'native-run'),
                    workers=2,coordinator_only=True,transport=args.transport,tokens=8,prompt='<s> </s> Hello',timeout=60,
                    memory_mib=None,host_memory_bytes=16000,worker_memory_bytes=45000,
                    proxy_delay_ms=50 if args.interrupt_worker else 0,proxy_jitter_ms=0,proxy_bytes_per_second=0,
                    interrupt_worker_after_token=2 if args.interrupt_worker else None))
            lab=run_sharded_lab(root/'ledger.db',actual);report['experiment']=lab
            if args.interrupt_worker:
                ok=not lab['parity'] and lab.get('native_fault_observed') and lab.get('partial_output_contiguous') and lab['charged_microcredits']==0 and lab['requester_balance']['available']==100_000_000
            else:
                ok=lab['parity'] and lab['charged_microcredits']==21_000_000 and all(x['available']==10_500_000 for x in lab['worker_balances'].values())
            if ok:report['status']='passed'
    except Exception as error:report['failure_type']=type(error).__name__
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
