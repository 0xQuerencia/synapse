"""Transactional testnet accounting. All mutations use one SQLite writer transaction.
The journal is authoritative; projection and escrow invariants are checked before commit.
Never expose database write access to requesters or workers.
"""
import contextlib
import hashlib
import json
import secrets
import sqlite3
import time
from pathlib import Path

CAP = 10**12

class Denied(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def require(condition, code, status=400):
    if not condition:
        raise Denied(code, status)


def integer(value, low=0, high=CAP):
    require(type(value) is int and low <= value <= high, 'INVALID_INTEGER')
    return value


def ident(value):
    require(isinstance(value, str) and 0 < len(value) <= 80 and all(c.isascii() and (c.isalnum() or c in '-_.') for c in value), 'INVALID_ID')
    return value


def digest(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'INVALID_DIGEST')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def uid():
    return secrets.token_hex(16)


def split(amount, weights):
    """Largest remainder with deterministic ID tie-break; exact integer conservation."""
    integer(amount)
    require(bool(weights) and all(type(v) is int and v > 0 for v in weights.values()), 'INVALID_WEIGHTS')
    total = sum(weights.values())
    values = {k: amount*v//total for k,v in weights.items()}
    order = sorted(weights, key=lambda k: (-(amount*weights[k] % total), k))
    for key in order[:amount-sum(values.values())]:
        values[key] += 1
    return values


class Ledger:
    def __init__(self, path, clock=None):
        self.path = Path(path)
        self.clock = clock or (lambda: int(time.time()))
        require(self.path != Path(':memory:'), 'PERSISTENT_DB_REQUIRED')
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Only use an operator-owned private directory; never accept a DB path from HTTP.
        require(not self.path.is_symlink(), 'UNSAFE_DB_PATH')
        import os, stat
        parent = self.path.parent
        require(parent.stat().st_uid == os.getuid() and not parent.is_symlink() and parent.stat().st_mode & 0o077 == 0, 'PRIVATE_DB_DIRECTORY_REQUIRED')
        if not self.path.exists():
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                os.close(fd)
            except FileExistsError:
                pass
        st = self.path.lstat()
        require(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_mode & 0o077 == 0, 'UNSAFE_DB_FILE')
        with contextlib.closing(self.connect()) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript((Path(__file__).parent.parent/'migrations/001_sqlite.sql').read_text())

    def connect(self):
        db = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        return db

    @contextlib.contextmanager
    def tx(self):
        with contextlib.closing(self.connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                yield db
                self.reconcile(db)
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @contextlib.contextmanager
    def read(self):
        with contextlib.closing(self.connect()) as db:
            db.execute('BEGIN')
            yield db

    @staticmethod
    def row(db, sql, args=(), code='NOT_FOUND'):
        r = db.execute(sql, args).fetchone()
        require(r is not None, code, 404)
        return r

    @staticmethod
    def account(db, pool, kind, owner):
        aid = sha(canonical([pool,kind,owner]))
        db.execute('INSERT OR IGNORE INTO accounts(id,pool,kind,owner) VALUES(?,?,?,?)',(aid,pool,kind,owner))
        return aid

    def post(self, db, pool, idem, kind, amounts, request):
        rh = sha(canonical([kind,request,amounts]))
        previous = db.execute('SELECT * FROM journal WHERE pool=? AND idem=?',(pool,idem)).fetchone()
        if previous:
            require(previous['request_hash']==rh, 'IDEMPOTENCY_CONFLICT',409)
            return previous['id']
        clean = {a:v for a,v in amounts.items() if v}
        require(len(clean)>=2 and sum(clean.values())==0, 'UNBALANCED_POSTING')
        jid = uid()
        db.execute('INSERT INTO journal VALUES(?,?,?,?,?,?)',(jid,pool,idem,kind,rh,self.clock()))
        for aid,amount in sorted(clean.items()):
            integer(amount,-CAP,CAP)
            row = self.row(db,'SELECT * FROM accounts WHERE id=? AND pool=?',(aid,pool))
            balance = row['balance']+amount
            require(-CAP<=balance<=CAP and (row['kind']=='issuance' or balance>=0),'INSUFFICIENT_CREDITS',409)
            db.execute('UPDATE accounts SET balance=? WHERE id=?',(balance,aid))
            db.execute('INSERT INTO entries VALUES(?,?,?)',(jid,aid,amount))
        return jid

    def reconcile(self, db):
        require(not db.execute('SELECT j.id FROM journal j LEFT JOIN entries e ON j.id=e.tx GROUP BY j.id HAVING COUNT(e.account)<2 OR COALESCE(SUM(e.amount),0)!=0').fetchone(),'JOURNAL_IMBALANCE',503)
        require(not db.execute('SELECT a.id FROM accounts a LEFT JOIN entries e ON a.id=e.account GROUP BY a.id HAVING a.balance!=COALESCE(SUM(e.amount),0)').fetchone(),'PROJECTION_MISMATCH',503)
        require(not db.execute('SELECT e.tx FROM entries e JOIN accounts a ON a.id=e.account JOIN journal j ON j.id=e.tx WHERE a.pool!=j.pool').fetchone(),'CROSS_POOL_JOURNAL',503)
        require(not db.execute("SELECT p.id FROM pools p LEFT JOIN accounts a ON p.id=a.pool GROUP BY p.id HAVING COALESCE(SUM(CASE WHEN a.kind='issuance' THEN -a.balance ELSE 0 END),0)!=p.issued OR COALESCE(SUM(a.balance),0)!=0").fetchone(),'SUPPLY_MISMATCH',503)
        require(not db.execute("SELECT j.id FROM jobs j JOIN quotes q ON j.quote=q.id LEFT JOIN accounts a ON a.pool=j.pool AND a.kind='escrow' AND a.owner=j.id WHERE a.id IS NULL OR (j.state='running' AND (a.balance!=q.maximum OR j.resolution_tx IS NOT NULL)) OR (j.state!='running' AND (a.balance!=0 OR j.resolution_tx IS NULL))").fetchone(),'ESCROW_MISMATCH',503)
        require(not db.execute("SELECT a.id FROM accounts a LEFT JOIN jobs j ON a.owner=j.id AND a.pool=j.pool WHERE a.kind='escrow' AND j.id IS NULL").fetchone(),'ORPHAN_ESCROW',503)
        return True

    def audit(self, db, pool, actor, kind, subject, detail=None):
        db.execute('INSERT INTO events(pool,actor,kind,subject,created,detail) VALUES(?,?,?,?,?,?)',(pool,actor,kind,subject,self.clock(),canonical(detail or {}).decode()))

    def event(self, db, pool, kind, subject):
        db.execute('INSERT INTO outbox(id,pool,kind,subject,created) VALUES(?,?,?,?,?)',(uid(),pool,kind,subject,self.clock()))
