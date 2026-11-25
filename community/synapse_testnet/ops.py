"""Operator-only local backup, recovery and integrity checks. No external deployment."""
import argparse
import contextlib
import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path
from .service import Service
from .ledger import require, sha, canonical

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(65536),b''): h.update(chunk)
    return h.hexdigest()

def private_dir(path):
    path=Path(path); path.mkdir(parents=True,exist_ok=True,mode=0o700)
    s=path.lstat(); require(stat.S_ISDIR(s.st_mode) and s.st_uid==os.getuid() and s.st_mode&0o077==0,'PRIVATE_BACKUP_DIRECTORY_REQUIRED')
    return path

def exclusive(path):
    return os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)

def summary(service):
    with service.read() as db:
        service.reconcile(db)
        require(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok','SQLITE_INTEGRITY_FAILURE',503)
        return {'testnet':True,'journal_transactions':db.execute('SELECT COUNT(*) FROM journal').fetchone()[0],'open_jobs':db.execute("SELECT COUNT(*) FROM jobs WHERE state='running'").fetchone()[0],'pending_outbox':db.execute('SELECT COUNT(*) FROM outbox WHERE delivered=0').fetchone()[0]}

def backup(service,destination):
    root=private_dir(destination); path=root/'ledger.sqlite'; meta=root/'backup.json'
    require(not path.exists() and not meta.exists(),'BACKUP_ALREADY_EXISTS',409)
    os.close(exclusive(path))
    try:
        with contextlib.closing(service.connect()) as src, contextlib.closing(sqlite3.connect(path)) as dest:
            src.backup(dest); dest.commit()
            require(dest.execute('PRAGMA integrity_check').fetchone()[0]=='ok','BACKUP_INTEGRITY_FAILURE')
            # Verify that the exact snapshot, not the moving live DB, reconciles.
            dest.row_factory=sqlite3.Row; service.reconcile(dest)
            count=dest.execute('SELECT COUNT(*) FROM journal').fetchone()[0]
        with open(path,'rb') as f: os.fsync(f.fileno())
        record={'version':1,'mode':'local-fake-credit-testnet','sha256':file_hash(path),'size':path.stat().st_size,'journal_transactions':count}
        with os.fdopen(exclusive(meta),'wb') as f: f.write(canonical(record)+b'\n'); f.flush(); os.fsync(f.fileno())
        dirfd=os.open(root,os.O_DIRECTORY)
        try: os.fsync(dirfd)
        finally: os.close(dirfd)
        return record
    except BaseException:
        path.unlink(missing_ok=True); meta.unlink(missing_ok=True); raise

def restore(source,destination):
    root=private_dir(source); dst=Path(destination); private_dir(dst.parent)
    require(not dst.exists(),'RESTORE_TARGET_EXISTS',409)
    p=root/'ledger.sqlite'; m=root/'backup.json'
    for f in (p,m):
        st=f.lstat(); require(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_mode&0o077==0,'UNSAFE_BACKUP_FILE')
    require(m.stat().st_size<4096,'INVALID_BACKUP_MANIFEST'); record=json.loads(m.read_text())
    require(record.get('mode')=='local-fake-credit-testnet' and record.get('size')==p.stat().st_size and record.get('sha256')==file_hash(p),'BACKUP_HASH_MISMATCH')
    try:
        with os.fdopen(exclusive(dst),'wb') as out, open(p,'rb') as inp:
            for chunk in iter(lambda:inp.read(65536),b''): out.write(chunk)
            out.flush(); os.fsync(out.fileno())
        require(file_hash(dst)==record['sha256'],'BACKUP_HASH_MISMATCH')
        s=Service(dst); result=summary(s)
        require(result['journal_transactions']==record['journal_transactions'],'BACKUP_JOURNAL_MISMATCH'); return result
    except BaseException:
        dst.unlink(missing_ok=True); raise

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('check'); a.add_argument('db'); a=sub.add_parser('backup'); a.add_argument('db'); a.add_argument('directory'); a=sub.add_parser('restore'); a.add_argument('directory'); a.add_argument('db')
    a=sub.add_parser('sweep'); a.add_argument('db'); args=p.parse_args()
    if args.cmd=='restore': result=restore(args.directory,args.db)
    else:
        s=Service(args.db)
        if args.cmd=='check': result=summary(s)
        elif args.cmd=='backup': result=backup(s,args.directory)
        else: result={'expired_refunded':s.sweep()}
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
