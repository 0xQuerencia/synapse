import concurrent.futures
import json
import tempfile
import threading
import unittest
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from synapse_testnet.demo import setup
from synapse_testnet.ledger import canonical, Denied, sha
from synapse_testnet.http_api import Server
from synapse_testnet.pilot import Client
from synapse_testnet.service import Service
from synapse_testnet.synthetic import MODEL

DOMAIN=b'synapse-testnet-enrollment-v1\0'

class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'db';self.s,self.tokens,_=setup(self.path)
        self.op=self.tokens['operator'];self.key=Ed25519PrivateKey.generate()
        self.public=self.key.public_key().public_bytes_raw().hex()

    def issue(self,member='new-member',ttl=60):return self.s.issue_invite(self.op,member,'requester',self.public,ttl)
    def sign(self,invite,key=None):return (key or self.key).sign(DOMAIN+canonical(invite['claim'])).hex()
    def redeem(self,invite,key=None):return self.s.redeem_invite(invite['ticket'],invite['claim'],self.sign(invite,key))

    def test_recipient_bound_http_redemption_requires_consent(self):
        with Server(self.s) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                api=Client(server.server_port,self.op)
                invite=api.call('/v1/operator/invites',{'member':'invitee','role':'requester','recipient':self.public,'ttl':60})
                guest=Client(server.server_port,invite['ticket'])
                with self.assertRaisesRegex(Denied,'BAD_SIGNATURE'):
                    guest.call('/v1/auth/enroll',{'claim':invite['claim'],'signature':self.sign(invite,Ed25519PrivateKey.generate())})
                result=guest.call('/v1/auth/enroll',{'claim':invite['claim'],'signature':self.sign(invite)})
                self.assertTrue(result['consent_required'])
                member=Client(server.server_port,result['token'])
                with self.assertRaisesRegex(Denied,'POOL_OR_CONSENT_REQUIRED'):member.call('/v1/catalog')
                member.call('/v1/consent',{'version':'testnet-v1','external_compute':True})
                self.assertEqual(member.call('/v1/catalog')['models'][0]['digest'],MODEL)
                with self.assertRaisesRegex(Denied,'INVITE_DENIED'):
                    guest.call('/v1/auth/enroll',{'claim':invite['claim'],'signature':self.sign(invite)})
            finally:server.shutdown();thread.join(5)

    def test_concurrent_redeem_only_one_wins(self):
        invite=self.issue()
        def attempt(_):
            try:return self.redeem(invite)['token']
            except Denied as e:return e.code
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(attempt,range(12)))
        self.assertEqual(results.count('INVITE_DENIED'),11)
        self.assertEqual(len([r for r in results if r!='INVITE_DENIED']),1)

    def test_expired_revoked_stolen_tampered(self):
        invite=self.issue(); tampered={**invite['claim'],'role':'worker'}
        with self.assertRaisesRegex(Denied,'INVITE_DENIED'):
            self.s.redeem_invite(invite['ticket'],tampered,self.sign(invite))
        with self.assertRaisesRegex(Denied,'INVITE_DENIED'):
            self.s.redeem_invite('x'*43,invite['claim'],self.sign(invite))
        new=Service(self.path,clock=lambda:invite['claim']['expires'])
        with self.assertRaisesRegex(Denied,'INVITE_DENIED'):
            new.redeem_invite(invite['ticket'],invite['claim'],self.sign(invite))
        self.s.revoke_invite(self.op,invite['claim']['id'])
        with self.assertRaisesRegex(Denied,'INVITE_DENIED'):self.redeem(invite)
        self.assertEqual(self.redeem(self.issue())['member'],'new-member')

    def test_invite_scope_and_nonoperator_denial(self):
        invite=self.issue()
        with self.assertRaisesRegex(Denied,'INVITE_ALREADY_ISSUED'):self.issue()
        with self.assertRaisesRegex(Denied,'FORBIDDEN'):
            self.s.issue_invite(self.tokens['client'],'nope','requester',self.public,60)
        other=self.s.bootstrap('other','op2',100)
        with self.assertRaisesRegex(Denied,'NOT_FOUND'):self.s.revoke_invite(other,invite['claim']['id'])

    def test_atomic_rotation_and_no_credential_in_events(self):
        token=self.tokens['client']
        def attempt(_):
            try:return self.s.rotate_token(token)['token']
            except Denied as e:return e.code
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(attempt,range(12)))
        self.assertEqual(results.count('AUTH_REQUIRED'),11)
        replacement=next(r for r in results if r!='AUTH_REQUIRED')
        with self.assertRaisesRegex(Denied,'AUTH_REQUIRED'):self.s.balance(token)
        self.assertEqual(self.s.balance(replacement)['available'],100_000_000)
        with self.s.read() as db:
            self.assertEqual(db.execute("SELECT token_hash FROM members WHERE id='client'").fetchone()[0],sha(replacement.encode()))
            events=json.dumps([dict(r) for r in db.execute('SELECT * FROM events')])
            self.assertNotIn(token,events);self.assertNotIn(replacement,events)

    def test_private_file_enrollment_and_rotation(self):
        from synapse_testnet.enrollment import make_key,issue,redeem,rotate,read_secret,save_secret
        root=self.path.parent;opfile=root/'operator.json';save_secret(opfile,{'token':self.op})
        keyfile=root/'recipient.json';pub=make_key(keyfile)['recipient']
        with Server(self.s) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                ticketfile=root/'invite.json';credfile=root/'member.json'
                result=issue(server.server_port,opfile,'file-member','requester',pub,60,ticketfile)
                self.assertNotIn(read_secret(ticketfile)['ticket'],str(result))
                result=redeem(server.server_port,ticketfile,keyfile,credfile)
                old=read_secret(credfile)['token'];self.assertNotIn(old,str(result))
                result=rotate(server.server_port,credfile);new=read_secret(credfile)['token']
                self.assertNotEqual(old,new);self.assertNotIn(new,str(result))
                for f in (keyfile,ticketfile,credfile):self.assertEqual(f.stat().st_mode&0o777,0o600)
                with self.assertRaisesRegex(Denied,'AUTH_REQUIRED'):self.s.balance(old)
                self.assertEqual(self.s.balance(new)['available'],0)
            finally:server.shutdown();thread.join(5)
