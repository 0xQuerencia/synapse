"""Isolated PostgreSQL reference backend; never a hosted production control plane."""
import contextlib
from decimal import Decimal
from pathlib import Path
import re
import time
from .service import Service

_SCHEMA = re.compile(r'synapse_lab_[0-9a-f]{32}\Z')

class Row(dict):
    def __getitem__(self, key):
        return tuple(self.values())[key] if isinstance(key, int) else super().__getitem__(key)

def _value(value):
    # SUM(bigint) in PostgreSQL produces numeric, not bigint. The ledger is
    # integer-only and the HTTP API must not serialize decimal or float values.
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError('non-integral PostgreSQL ledger value')
        return int(value)
    return value

def row_factory(cursor):
    names = tuple(c.name for c in cursor.description) if cursor.description else ()
    return lambda values: Row(zip(names, map(_value, values)))

class Connection:
    def __init__(self, db): self.db = db
    def execute(self, sql, args=()):
        # These are static application queries, not caller-supplied SQL.
        sql = sql.replace('?', '%s')
        if sql.startswith('INSERT OR IGNORE INTO '):
            sql = sql.replace('INSERT OR IGNORE INTO ', 'INSERT INTO ', 1) + ' ON CONFLICT DO NOTHING'
        return self.db.execute(sql, args)
    def close(self): self.db.close()
    def commit(self): self.db.commit()
    def rollback(self): self.db.rollback()

class PostgresService(Service):
    def __init__(self, dsn, *, schema, isolated_testnet=False, clock=None):
        if isolated_testnet is not True or not isinstance(schema, str) or not _SCHEMA.fullmatch(schema):
            raise ValueError('explicit isolated testnet and unique synapse_lab_<32 hex> schema required')
        import psycopg
        from psycopg import sql
        self.psycopg, self.dsn, self.schema = psycopg, dsn, schema
        self.clock = clock or (lambda: int(time.time()))
        with psycopg.connect(dsn, connect_timeout=5) as db:
            # Prevent accidental migrations into public or another application's
            # schema. The test role must not be a superuser or own arbitrary DBs.
            role = db.execute('SELECT rolsuper, rolcreaterole, rolcreatedb FROM pg_roles WHERE rolname=current_user').fetchone()
            if role is None or any(role):
                raise ValueError('restricted non-superuser testnet role required')
            db.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))', (schema,))
            db.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
            owned = db.execute('SELECT nspowner = (SELECT oid FROM pg_roles WHERE rolname=current_user) FROM pg_namespace WHERE nspname=%s', (schema,)).fetchone()
            if not owned or owned[0] is not True:
                raise ValueError('testnet schema must be owned by testnet role')
            db.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(schema)))
            db.execute((Path(__file__).parent.parent/'migrations/001_postgres.sql').read_text(), prepare=False)

    def connect(self):
        # schema validated before interpolation; connection startup sets the
        # search_path before any query. No public-schema fallback.
        db = self.psycopg.connect(self.dsn, row_factory=row_factory, autocommit=False,
                                  connect_timeout=5, options=f'-c search_path={self.schema},pg_catalog -c lock_timeout=10000 -c statement_timeout=15000')
        return Connection(db)

    @contextlib.contextmanager
    def tx(self):
        with contextlib.closing(self.connect()) as db:
            try:
                db.execute('SET TRANSACTION ISOLATION LEVEL READ COMMITTED')
                db.execute('SELECT pg_advisory_xact_lock(hashtextextended(?, 0))', (self.schema,))
                yield db
                self.reconcile(db)
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @contextlib.contextmanager
    def read(self):
        with contextlib.closing(self.connect()) as db:
            db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            yield db
