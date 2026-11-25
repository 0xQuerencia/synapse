"""Crash-recoverable operator event export to an isolated local SQLite sink.

Delivery is at-least-once over HTTP; a unique event ID and payload fingerprint
make insertion into this specific sink idempotent. The ledger acknowledgement
is sent only after the sink commits. No arbitrary side effects or network fanout.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from .enrollment import read_secret
from .ledger import canonical, digest, ident, integer, require, sha
from .ops import private_dir, exclusive
from .pilot import Client

FIELDS={'id','pool','kind','subject','created'}

class Sink:
    def __init__(self,path):
        self.path=Path(path);private_dir(self.path.parent)
        try: os.close(exclusive(self.path))
        except FileExistsError: pass
        st=self.path.lstat()
        require(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and not st.st_mode&0o077,'UNSAFE_EVENT_SINK')
        with contextlib.closing(self.connect()) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS delivered(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL)')
            db.commit()

    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.execute('PRAGMA synchronous=FULL');return db

    def store(self,event):
        require(isinstance(event,dict) and set(event)==FIELDS,'INVALID_EVENT')
        for field in ('id','pool','kind','subject'):ident(event[field])
        integer(event['created'],0,2**62)
        payload=canonical(event);fingerprint=sha(payload)
        with contextlib.closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT fingerprint,payload FROM delivered WHERE id=?',(event['id'],)).fetchone()
            require(row is None or row==(fingerprint,payload.decode()),'EVENT_FINGERPRINT_MISMATCH',409)
            if row is None:db.execute('INSERT INTO delivered VALUES(?,?,?)',(event['id'],fingerprint,payload.decode()))
        return fingerprint

    def count(self):
        with contextlib.closing(self.connect()) as db:return db.execute('SELECT COUNT(*) FROM delivered').fetchone()[0]


def drain(client,sink,limit=50):
    integer(limit,1,100)
    events=client.call('/v1/operator/events',{'limit':limit})
    require(isinstance(events,list) and len(events)<=limit,'INVALID_EVENT_BATCH')
    for event in events:
        fingerprint=sink.store(event)
        ack=client.call('/v1/operator/events/ack',{'event':event['id'],'fingerprint':fingerprint})
        require(ack=={'acknowledged':True,'event':event['id']},'EVENT_ACK_FAILED')
    return {'acknowledged':len(events),'sink_rows':sink.count(),'fake_credits':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8787);p.add_argument('--operator-file',required=True);p.add_argument('--sink',required=True);p.add_argument('--limit',type=int,default=50);args=p.parse_args()
    print(json.dumps(drain(Client(args.port,read_secret(args.operator_file)['token']),Sink(args.sink),args.limit)))

if __name__=='__main__':main()
