"""Unqualified PostgreSQL candidate for the same synthetic-only service.
Requires an isolated empty PostgreSQL database/schema and psycopg 3. No hosted enablement.
SQLite remains the actually tested backend in this tranche.
"""
import contextlib
from pathlib import Path
from .service import Service

class Row(dict):
    def __getitem__(self,key):
        return tuple(self.values())[key] if isinstance(key,int) else super().__getitem__(key)

def row_factory(cursor):
    names=tuple(c.name for c in cursor.description) if cursor.description else ()
    return lambda values: Row(zip(names,values))

class Connection:
    def __init__(self,db): self.db=db
    def execute(self,sql,args=()):
        sql=sql.replace('?','%s')
        if sql.startswith('INSERT OR IGNORE INTO '): sql=sql.replace('INSERT OR IGNORE INTO ','INSERT INTO ',1)+' ON CONFLICT DO NOTHING'
        return self.db.execute(sql,args)
    def close(self): self.db.close()
    def commit(self): self.db.commit()
    def rollback(self): self.db.rollback()

class PostgresService(Service):
    def __init__(self,dsn,clock=None,*,isolated_testnet=False):
        if isolated_testnet is not True: raise ValueError('explicit isolated testnet required')
        import psycopg
        import time
        self.psycopg=psycopg; self.dsn=dsn; self.clock=clock or (lambda:int(time.time()))
        with psycopg.connect(dsn) as db:
            db.execute('SELECT pg_advisory_xact_lock(834763921)')
            db.execute((Path(__file__).parent.parent/'migrations/001_postgres.sql').read_text(),prepare=False)
    def connect(self): return Connection(self.psycopg.connect(self.dsn,row_factory=row_factory,autocommit=False))
    @contextlib.contextmanager
    def tx(self):
        with contextlib.closing(self.connect()) as db:
            try:
                # Serialize all testnet writers before inspecting or mutating accounting.
                # Default READ COMMITTED sees the previous writer after the lock is acquired.
                db.execute('SELECT pg_advisory_xact_lock(834763921)')
                yield db
                self.reconcile(db); db.commit()
            except BaseException:
                db.rollback(); raise
    @contextlib.contextmanager
    def read(self):
        with contextlib.closing(self.connect()) as db:
            db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            yield db
