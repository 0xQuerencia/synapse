"""Real isolated PostgreSQL qualification, opt-in via SYNAPSE_TEST_POSTGRES_DSN.
The CI service is disposable; never point this at an existing hosted database.
"""
import concurrent.futures
from decimal import Decimal
import json
import os
import secrets
import unittest
from synapse_testnet.demo import setup, execute
from synapse_testnet.ledger import Denied, canonical, sha
from synapse_testnet.postgres import PostgresService, _value
from synapse_testnet.synthetic import MODEL, receipt_body, observed

DSN = os.environ.get('SYNAPSE_TEST_POSTGRES_DSN')

class PostgresContractTests(unittest.TestCase):
    def test_integral_numeric_conversion(self):
        self.assertEqual(_value(Decimal('1000000000000')), 1000000000000)
        self.assertIs(type(_value(Decimal('1.0'))), int)
        for value in ('0.1', 'NaN', 'Infinity'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _value(Decimal(value))

    def test_requires_explicit_isolation(self):
        for schema in ('public', 'synapse_lab_wrong', 'a;DROP SCHEMA public'):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                PostgresService('', schema=schema, isolated_testnet=True)
        with self.assertRaises(ValueError):
            PostgresService('', schema='synapse_lab_'+'1'*32)

@unittest.skipUnless(DSN, 'requires isolated PostgreSQL CI service')
class PostgresIntegrationTests(unittest.TestCase):
    def service(self):
        import psycopg
        from psycopg import sql
        schema = 'synapse_lab_' + secrets.token_hex(16)
        def cleanup():
            with psycopg.connect(DSN, connect_timeout=5) as db:
                db.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
        self.addCleanup(cleanup)
        return PostgresService(DSN, schema=schema, isolated_testnet=True)

    def setUp(self):
        self.s = self.service()
        self.s, self.t, self.keys = setup(None, service=self.s)
        self.client, self.op, self.v = (self.t[k] for k in ('client','operator','verifier'))

    def quote(self):
        return self.s.quote(self.client, MODEL, 'demo-v1', 2, 8)['quote']

    def receipts(self, corrupt=False):
        for worker, key in self.keys.items():
            assignment = self.s.assignments(self.t[worker])[0]
            body = receipt_body(assignment, 4)
            if corrupt: body['evidence'] = '0'*64
            signature = key.sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex()
            self.s.receipt(self.t[worker], body, signature)
        for row in self.s.pending(self.v):
            body = json.loads(row['body'])
            evidence = observed({**body, 'first_layer':row['first_layer'], 'end_layer':row['end_layer']}, 4)
            self.s.validate(self.v, row['assignment'], row['digest'], True, 4, evidence)
            if corrupt: break

    def test_http_sharded_and_full_model(self):
        for sharded in (True, False):
            with self.subTest(sharded=sharded):
                result = execute(None, sharded, service=self.service())
                self.assertTrue(result['duplicate_settlement_equal'])
                self.assertEqual(result['settled']['charged'], 12_000_000)
                self.assertEqual(result['balances']['client']['available'], 88_000_000)
                self.assertIs(type(result['held']['held']), int)
                if not sharded: self.assertEqual(result['redemption']['charged'], 5_000_000)

    def test_concurrent_reservation_and_settlement(self):
        quote = self.quote()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(lambda _: self.s.reserve(self.client, quote, 'same'), range(16)))
        self.assertEqual(len({j['job'] for j in jobs}), 1)
        self.assertEqual(self.s.balance(self.client)['held'], 20_000_000)
        self.receipts()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            settlements = list(pool.map(lambda _: self.s.settle(self.client, jobs[0]['job']), range(16)))
        self.assertTrue(all(x == settlements[0] for x in settlements))
        self.assertEqual(self.s.balance(self.client)['available'], 88_000_000)
        self.assertEqual(self.s.balance(self.t['worker-a'])['available'], 6_000_000)

    def test_concurrent_capacity(self):
        quotes = [self.quote(), self.quote()]
        def reserve(q):
            try: return self.s.reserve(self.client, q, q)['state']
            except Denied as e: return e.code
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            self.assertCountEqual(list(pool.map(reserve, quotes)), ['running', 'NO_ROUTE'])

    def test_reconnect_refund_and_reconcile(self):
        j = self.s.reserve(self.client, self.quote(), 'expire')
        restored = PostgresService(DSN, schema=self.s.schema, isolated_testnet=True,
                                   clock=lambda: j['expires']+1)
        self.assertEqual(restored.sweep(), 1)
        self.assertEqual(restored.sweep(), 0)
        self.assertEqual(restored.balance(self.client)['available'], 100_000_000)
        with restored.read() as db: self.assertTrue(restored.reconcile(db))

    def test_forgery_and_pool_isolation(self):
        j = self.s.reserve(self.client, self.quote(), 'forged')
        self.receipts(corrupt=True)
        self.assertEqual(self.s.job(self.client, j['job'])['state'], 'failed')
        self.assertEqual(self.s.balance(self.client)['held'], 0)
        other = self.s.bootstrap('other','op2',100)
        token = self.s.add_member(other, 'other-user', 'requester')
        with self.assertRaisesRegex(Denied,'NOT_FOUND'):
            self.s.job(token, j['job'])

    def test_db_constraints_and_rollback(self):
        import psycopg
        with self.assertRaises(psycopg.Error):
            with self.s.tx() as db: db.execute('DELETE FROM journal')
        # A direct malformed transaction must fail even without reconcile().
        with self.assertRaises(psycopg.Error):
            with self.s.psycopg.connect(DSN, options=f'-c search_path={self.s.schema},pg_catalog') as db:
                db.execute('INSERT INTO journal VALUES(%s,%s,%s,%s,%s,%s)',
                           ('bad','pilot','bad','grant',sha(b'bad'),1))
        with self.assertRaisesRegex(Denied, 'PROJECTION_MISMATCH'):
            with self.s.tx() as db: db.execute("UPDATE accounts SET balance=0 WHERE kind='available'")
        self.assertEqual(self.s.balance(self.client)['available'], 100_000_000)
        with self.s.read() as db: self.assertTrue(self.s.reconcile(db))
