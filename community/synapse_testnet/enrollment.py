"""Local testnet membership invitation CLI; credentials never printed or passed in argv."""
import argparse
import json
import os
import stat
import tempfile
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .ledger import canonical, require
from .ops import exclusive, private_dir
from .pilot import Client

DOMAIN=b'synapse-testnet-enrollment-v1\0'


def read_secret(path):
    fd=os.open(Path(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        st=os.fstat(f.fileno())
        require(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and not st.st_mode&0o077 and 0<st.st_size<=8192,'UNSAFE_CREDENTIAL_FILE')
        return json.loads(f.read(8193))


def save_secret(path,payload):
    path=Path(path);private_dir(path.parent)
    with os.fdopen(exclusive(path),'wb') as f:
        f.write(canonical(payload)+b'\n');f.flush();os.fsync(f.fileno())
    sync_dir(path.parent)


def sync_dir(path):
    fd=os.open(path,os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def make_key(path):
    key=Ed25519PrivateKey.generate();pub=key.public_key().public_bytes_raw().hex()
    save_secret(path,{'key':key.private_bytes_raw().hex(),'recipient':pub})
    return {'recipient':pub}


def issue(port,operator_file,member,role,recipient,ttl,destination):
    destination=Path(destination);private_dir(destination.parent)
    # Reserve the output path before the server issues a one-time ticket.
    with os.fdopen(exclusive(destination),'wb') as f:
        try:
            api=Client(port,read_secret(operator_file)['token'])
            invite=api.call('/v1/operator/invites',{'member':member,'role':role,'recipient':recipient,'ttl':ttl})
            f.write(canonical(invite)+b'\n');f.flush();os.fsync(f.fileno())
        except BaseException:
            destination.unlink(missing_ok=True);raise
    sync_dir(destination.parent)
    return {'invite':invite['claim']['id'],'member':member,'expires':invite['claim']['expires'],'ticket_written':True}


def redeem(port,invite_file,key_file,destination):
    invitation=read_secret(invite_file);key=Ed25519PrivateKey.from_private_bytes(bytes.fromhex(read_secret(key_file)['key']))
    claim=invitation['claim'];require(key.public_key().public_bytes_raw().hex()==claim['recipient'],'WRONG_RECIPIENT_KEY')
    destination=Path(destination);private_dir(destination.parent)
    with os.fdopen(exclusive(destination),'wb') as f:
        try:
            result=Client(port,invitation['ticket']).call('/v1/auth/enroll',{'claim':claim,'signature':key.sign(DOMAIN+canonical(claim)).hex()})
            f.write(canonical({'token':result['token']})+b'\n');f.flush();os.fsync(f.fileno())
        except BaseException:
            destination.unlink(missing_ok=True);raise
    sync_dir(destination.parent)
    return {'member':result['member'],'role':result['role'],'consent_required':True,'credential_written':True}


def rotate(port,credential_file):
    path=Path(credential_file);private_dir(path.parent);payload=read_secret(path)
    # Do not print secrets. A failure after server-side rotation may still need
    # operator recovery; do not pretend the old credential remains valid.
    fd,name=tempfile.mkstemp(prefix='.credential-',dir=path.parent);tmp=Path(name)
    try:
        with os.fdopen(fd,'wb') as f:
            result=Client(port,payload['token']).call('/v1/auth/rotate',{})
            f.write(canonical({**payload,'token':result['token']})+b'\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);sync_dir(path.parent)
    finally:tmp.unlink(missing_ok=True)
    return {'member':result['member'],'credential_rotated':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8787)
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('generate-key');a.add_argument('--out',required=True)
    a=sub.add_parser('issue');a.add_argument('--operator-file',required=True);a.add_argument('--member',required=True);a.add_argument('--role',choices=['requester','worker','verifier'],required=True);a.add_argument('--recipient',required=True);a.add_argument('--ttl',type=int,default=300);a.add_argument('--out',required=True)
    a=sub.add_parser('redeem');a.add_argument('--invite-file',required=True);a.add_argument('--key-file',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('rotate');a.add_argument('--credential-file',required=True)
    a=p.parse_args()
    if a.action=='generate-key':result=make_key(a.out)
    elif a.action=='issue':result=issue(a.port,a.operator_file,a.member,a.role,a.recipient,a.ttl,a.out)
    elif a.action=='redeem':result=redeem(a.port,a.invite_file,a.key_file,a.out)
    else:result=rotate(a.port,a.credential_file)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
