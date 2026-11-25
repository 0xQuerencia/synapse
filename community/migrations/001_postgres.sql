CREATE TABLE IF NOT EXISTS pools(id TEXT PRIMARY KEY, enabled BIGINT NOT NULL CHECK(enabled IN(0,1)), consent_version TEXT NOT NULL, grant_cap BIGINT NOT NULL CHECK(grant_cap BETWEEN 0 AND 1000000000000), issued BIGINT NOT NULL DEFAULT 0 CHECK(issued>=0 AND issued<=grant_cap));
CREATE TABLE IF NOT EXISTS members(id TEXT PRIMARY KEY, pool TEXT NOT NULL REFERENCES pools(id), role TEXT NOT NULL CHECK(role IN('requester','worker','verifier','operator')), token_hash TEXT UNIQUE NOT NULL, enabled BIGINT NOT NULL DEFAULT 1 CHECK(enabled IN(0,1)), consent TEXT);
CREATE TABLE IF NOT EXISTS enrollment_invites(id TEXT PRIMARY KEY, pool TEXT NOT NULL REFERENCES pools(id), issuer TEXT NOT NULL REFERENCES members(id), member TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN('requester','worker','verifier')), recipient TEXT NOT NULL, ticket_hash TEXT UNIQUE NOT NULL, expires BIGINT NOT NULL, state TEXT NOT NULL DEFAULT 'issued' CHECK(state IN('issued','redeemed','revoked')));
CREATE INDEX IF NOT EXISTS enrollment_invites_scope ON enrollment_invites(pool,state,expires);
CREATE TABLE IF NOT EXISTS contributor_profiles(member TEXT PRIMARY KEY REFERENCES members(id), pool TEXT NOT NULL REFERENCES pools(id), alias TEXT NOT NULL, UNIQUE(pool,alias));
CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY, owner TEXT NOT NULL REFERENCES members(id), pubkey TEXT UNIQUE NOT NULL, enabled BIGINT NOT NULL DEFAULT 1 CHECK(enabled IN(0,1)), proved BIGINT NOT NULL DEFAULT 0 CHECK(proved IN(0,1)), challenge TEXT, challenge_expires BIGINT, ready_until BIGINT NOT NULL DEFAULT 0, heartbeat_seq BIGINT NOT NULL DEFAULT 0, mode TEXT NOT NULL DEFAULT 'paused' CHECK(mode IN('ready','paused','draining','stopped')), model TEXT NOT NULL, first_layer BIGINT NOT NULL CHECK(first_layer>=0), end_layer BIGINT NOT NULL CHECK(end_layer>first_layer), max_jobs BIGINT NOT NULL CHECK(max_jobs BETWEEN 1 AND 4));
CREATE TABLE IF NOT EXISTS models(pool TEXT NOT NULL REFERENCES pools(id), digest TEXT NOT NULL, layers BIGINT NOT NULL CHECK(layers BETWEEN 1 AND 256), max_context BIGINT NOT NULL CHECK(max_context BETWEEN 1 AND 32768), enabled BIGINT NOT NULL CHECK(enabled IN(0,1)), PRIMARY KEY(pool,digest));
CREATE TABLE IF NOT EXISTS tariffs(pool TEXT NOT NULL REFERENCES pools(id), version TEXT NOT NULL, base BIGINT NOT NULL CHECK(base>=0), prefill BIGINT NOT NULL CHECK(prefill>=0), decode BIGINT NOT NULL CHECK(decode>=0), PRIMARY KEY(pool,version));
CREATE TABLE IF NOT EXISTS accounts(id TEXT PRIMARY KEY, pool TEXT NOT NULL REFERENCES pools(id), kind TEXT NOT NULL CHECK(kind IN('issuance','treasury','available','escrow')), owner TEXT NOT NULL, balance BIGINT NOT NULL DEFAULT 0 CHECK(balance BETWEEN -1000000000000 AND 1000000000000 AND (kind='issuance' OR balance>=0)), UNIQUE(pool,kind,owner));
CREATE TABLE IF NOT EXISTS journal(id TEXT PRIMARY KEY, pool TEXT NOT NULL REFERENCES pools(id), idem TEXT NOT NULL, kind TEXT NOT NULL, request_hash TEXT NOT NULL, created BIGINT NOT NULL, UNIQUE(pool,idem));
CREATE TABLE IF NOT EXISTS entries(tx TEXT NOT NULL REFERENCES journal(id), account TEXT NOT NULL REFERENCES accounts(id), amount BIGINT NOT NULL CHECK(amount!=0 AND amount BETWEEN -1000000000000 AND 1000000000000), PRIMARY KEY(tx,account));
CREATE TABLE IF NOT EXISTS quotes(id TEXT PRIMARY KEY, pool TEXT NOT NULL REFERENCES pools(id), requester TEXT NOT NULL REFERENCES members(id), model TEXT NOT NULL, tariff TEXT NOT NULL, input_units BIGINT NOT NULL CHECK(input_units>0), max_output BIGINT NOT NULL CHECK(max_output>0), maximum BIGINT NOT NULL CHECK(maximum>0), expires BIGINT NOT NULL, used BIGINT NOT NULL DEFAULT 0 CHECK(used IN(0,1)), FOREIGN KEY(pool,model) REFERENCES models(pool,digest), FOREIGN KEY(pool,tariff) REFERENCES tariffs(pool,version));
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, pool TEXT NOT NULL REFERENCES pools(id), requester TEXT NOT NULL REFERENCES members(id), quote TEXT UNIQUE NOT NULL REFERENCES quotes(id), idem TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN('running','settled','cancelled','expired','failed')), expires BIGINT NOT NULL, delivered BIGINT, charged BIGINT, resolution_tx TEXT REFERENCES journal(id), UNIQUE(requester,idem));
CREATE TABLE IF NOT EXISTS assignments(id TEXT PRIMARY KEY, job TEXT NOT NULL REFERENCES jobs(id), device TEXT NOT NULL REFERENCES devices(id), first_layer BIGINT NOT NULL, end_layer BIGINT NOT NULL CHECK(end_layer>first_layer), epoch BIGINT NOT NULL CHECK(epoch=1), UNIQUE(job,first_layer), UNIQUE(job,device));
CREATE TABLE IF NOT EXISTS receipts(assignment TEXT PRIMARY KEY REFERENCES assignments(id), body TEXT NOT NULL, signature TEXT NOT NULL, digest TEXT NOT NULL, output_units BIGINT NOT NULL, input_units BIGINT NOT NULL, status TEXT NOT NULL CHECK(status IN('pending','verified','rejected')), verifier TEXT REFERENCES members(id));
CREATE TABLE IF NOT EXISTS disputes(id TEXT PRIMARY KEY, job TEXT NOT NULL REFERENCES jobs(id), owner TEXT NOT NULL REFERENCES members(id), reason TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN('open','approved','rejected','resolved')), reviewer TEXT REFERENCES members(id), compensation BIGINT NOT NULL DEFAULT 0 CHECK(compensation>=0), UNIQUE(job,owner));
CREATE TABLE IF NOT EXISTS grants(pool TEXT NOT NULL REFERENCES pools(id), member TEXT NOT NULL REFERENCES members(id), amount BIGINT NOT NULL CHECK(amount>0), PRIMARY KEY(pool,member));
CREATE TABLE IF NOT EXISTS events(seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, pool TEXT NOT NULL REFERENCES pools(id), actor TEXT NOT NULL, kind TEXT NOT NULL, subject TEXT NOT NULL, created BIGINT NOT NULL, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outbox(id TEXT PRIMARY KEY, pool TEXT NOT NULL REFERENCES pools(id), kind TEXT NOT NULL, subject TEXT NOT NULL, created BIGINT NOT NULL, delivered BIGINT NOT NULL DEFAULT 0 CHECK(delivered IN(0,1)));
-- Apply only to a new isolated testnet schema with a restricted owner.
CREATE OR REPLACE FUNCTION reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'immutable accounting record'; END; $$;

DROP TRIGGER IF EXISTS immutable_journal ON journal;
CREATE TRIGGER immutable_journal BEFORE UPDATE OR DELETE ON journal FOR EACH ROW EXECUTE FUNCTION reject_mutation();
DROP TRIGGER IF EXISTS immutable_entries ON entries;
CREATE TRIGGER immutable_entries BEFORE UPDATE OR DELETE ON entries FOR EACH ROW EXECUTE FUNCTION reject_mutation();
DROP TRIGGER IF EXISTS immutable_tariffs ON tariffs;
CREATE TRIGGER immutable_tariffs BEFORE UPDATE OR DELETE ON tariffs FOR EACH ROW EXECUTE FUNCTION reject_mutation();
DROP TRIGGER IF EXISTS immutable_events ON events;
CREATE TRIGGER immutable_events BEFORE UPDATE OR DELETE ON events FOR EACH ROW EXECUTE FUNCTION reject_mutation();
CREATE OR REPLACE FUNCTION validate_journal_transaction() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE transaction_id text; n bigint; total numeric; bad bigint;
BEGIN
    IF TG_TABLE_NAME = 'journal' THEN transaction_id := NEW.id; ELSE transaction_id := NEW.tx; END IF;
    SELECT count(*),coalesce(sum(e.amount),0),count(*) FILTER (WHERE a.pool<>j.pool)
      INTO n,total,bad FROM entries e JOIN accounts a ON a.id=e.account JOIN journal j ON j.id=e.tx WHERE e.tx=transaction_id;
    IF n<2 OR total<>0 OR bad<>0 THEN RAISE EXCEPTION 'unbalanced or cross-pool journal'; END IF;
    RETURN NULL;
END; $$;
DROP TRIGGER IF EXISTS balanced_journal ON journal;
CREATE CONSTRAINT TRIGGER balanced_journal AFTER INSERT ON journal DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION validate_journal_transaction();
DROP TRIGGER IF EXISTS balanced_entries ON entries;
CREATE CONSTRAINT TRIGGER balanced_entries AFTER INSERT ON entries DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION validate_journal_transaction();

