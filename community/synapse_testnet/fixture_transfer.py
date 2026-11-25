"""Deliver only the pinned original tiny-Llama fixture through authenticated chunks.

The receiver stages exact verified files before making a native model directory
visible. This is an operator-local proof of compatibility, not native libp2p
model seeding or third-party redistribution authorization.
"""
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .artifacts import Store, Policy, Seeder, manifest_for, fetch, MAX_ARTIFACT
from .ledger import require
from .native_lab import ARTIFACTS, verify_fixture


def deliver_fixture(source, workspace):
    """Return the new verified native model path and redacted delivery evidence."""
    source, workspace=Path(source),Path(workspace)
    verify_fixture(source)
    workspace.mkdir(mode=0o700,parents=False,exist_ok=False)
    key=Ed25519PrivateKey.generate()
    policy=Policy([key.public_key().public_bytes_raw().hex()])
    token=secrets.token_urlsafe(32)
    origin=Store(workspace/'origin-cache',MAX_ARTIFACT)
    receiver=Store(workspace/'receiver-cache',MAX_ARTIFACT)
    manifests={}
    for name in (*ARTIFACTS,'manifest.json'):
        data=(source/name).read_bytes()
        m=manifest_for(data,key,int(time.time())+600,redistribute=True)
        origin.import_data(policy,m,data);manifests[name]=m
    target=workspace/'native-model'
    stage=Path(tempfile.mkdtemp(prefix='.stage-',dir=workspace))
    reports=[]
    try:
        with Seeder(origin,policy,list(manifests.values()),token,MAX_ARTIFACT) as seed:
            thread=threading.Thread(target=seed.serve_forever,daemon=True);thread.start()
            try:
                for name,m in manifests.items():
                    report=fetch(receiver,policy,m,[(seed.server_port,token)])
                    require(report['complete'],'FIXTURE_TRANSFER_INCOMPLETE')
                    exported=receiver.export(policy,m,stage/name)
                    reports.append({'file':name,**report,**exported})
            finally:seed.shutdown();thread.join(5)
        model=verify_fixture(stage)
        require(not target.exists(),'FIXTURE_TARGET_EXISTS')
        stage.rename(target)
        Store._sync(workspace)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    return target,{'scope':'original pinned fixture, authenticated loopback artifact transfer',
                   'model_fingerprint':model,'files':reports,'receiver_bytes':receiver.bytes(),
                   'native_path_verified':True,'physical_wan_verified':False,
                   'third_party_redistribution':False}
