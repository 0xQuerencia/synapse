#!/usr/bin/env python3
"""Exercise actual native CPU inference with a disposable fake-credit ledger.

Uses two independently launched synapse-gui processes on loopback, not two
physical machines. The original known-answer tiny-Llama fixture is immutable
and hash-checked; neither arbitrary prompts nor third-party data are accepted.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'community'))
from synapse_testnet.native_lab import Endpoint, run_lab, verify_fixture
from synapse_testnet.fixture_transfer import deliver_fixture
from gui_smoke import GuiServer, digest, require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-revision')
    args=p.parse_args()
    binary=args.binary.resolve()
    require(binary.is_file() and os.access(binary,os.X_OK),'executable native GUI required')
    fixture=ROOT/'crates/synapse-core/tests/fixtures/tiny-llama'
    verify_fixture(fixture)
    args.out.mkdir(parents=True,exist_ok=False)
    result={'status':'failed','source_revision':args.source_revision,'binary_sha256':digest(binary),
            'evidence_scope':'two actual local native processes, original fixture, no production credits'}
    servers=[]
    try:
        with tempfile.TemporaryDirectory(prefix='synapse-native-credit-lab-') as temporary:
            root=Path(temporary)
            try:
                delivered,result['artifact_delivery']=deliver_fixture(fixture,root/'delivery')
                for label,model_path in (('candidate',delivered),('reference',fixture)):
                    server=GuiServer(label,binary,model_path,root);servers.append(server);server.ready()
                endpoints=[Endpoint(urlparse(s.base_url).port,s.token_path) for s in servers]
                result['experiment']=run_lab(root/'ledger.db',*endpoints)
                require(result['experiment']['parity'],'native reference mismatch; fake-credit reservation refunded')
                require(result['experiment']['charged_microcredits']==21_000_000,'metering mismatch')
                result['status']='passed'
            finally:
                for server in reversed(servers):server.stop()
                # Keep only structured evidence, never credentials, raw logs or ledger.
                for server in servers:server.log_handle.close()
                result['cleanup']={s.label:s.cleanup_result for s in servers}
                require(all(s.cleanup_result=='exited' for s in servers),'native process cleanup failed')
    except Exception as error:
        result['status']='failed'
        result['failure_type']=type(error).__name__
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
