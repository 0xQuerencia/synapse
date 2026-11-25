import secrets
import tempfile
import threading
import time
import unittest
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from synapse_testnet.artifacts import *

class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); root=Path(self.tmp.name)
        self.key=Ed25519PrivateKey.generate(); self.data=bytes(range(256))*(CHUNK//256)+b'other'*(CHUNK//5)+b'last'
        self.m=manifest_for(self.data,self.key,int(time.time())+600,True); self.p=Policy([self.m['publisher']]); self.source=Store(root/'source',MAX_ARTIFACT); self.target=Store(root/'target',MAX_ARTIFACT); self.source.import_data(self.p,self.m,self.data)
        self.token=secrets.token_urlsafe(32)
    @contextlib.contextmanager
    def seeder(self,limit=MAX_ARTIFACT):
        with Seeder(self.source,self.p,[self.m],self.token,limit) as s:
            t=threading.Thread(target=s.serve_forever); t.start()
            try: yield s
            finally: s.shutdown(); t.join()
    def test_resume_verified_transfer(self):
        with self.seeder() as s:
            r=fetch(self.target,self.p,self.m,[(s.server_port,self.token)],max_new=1); self.assertFalse(r['complete']); self.assertEqual(r['fetched'],1)
            r=fetch(self.target,self.p,self.m,[(s.server_port,self.token)]); self.assertTrue(r['complete']); self.assertGreaterEqual(r['reused'],1); self.assertEqual(self.target.assemble(self.p,self.m),self.data)
    def test_fallback_wrong_token(self):
        with self.seeder() as s:
            r=fetch(self.target,self.p,self.m,[(s.server_port,secrets.token_urlsafe(32)),(s.server_port,self.token)]); self.assertTrue(r['complete']); self.assertGreater(r['peer_failures'],0)
    def test_unsigned_or_tampered_manifest(self):
        with self.assertRaises(Denied): self.p.check({**self.m,'size':self.m['size']-1})
    def test_rights_denied(self):
        m=manifest_for(self.data,self.key,int(time.time())+500,False)
        with self.assertRaisesRegex(Denied,'REDISTRIBUTION_DENIED'): self.p.check(m)
    def test_revocation(self):
        self.p.revoked.add(self.m['artifact'])
        with self.assertRaisesRegex(Denied,'UNTRUSTED_OR_REVOKED'): self.source.assemble(self.p,self.m)
    def test_quota(self):
        small=Store(Path(self.tmp.name)/'small',10)
        with self.assertRaisesRegex(Denied,'CACHE_QUOTA'): small.import_data(self.p,self.m,self.data)
    def test_corrupt_cache(self):
        (self.source.root/self.m['chunks'][0]).write_bytes(b'wrong')
        with self.assertRaisesRegex(Denied,'CACHE_CORRUPT'): self.source.assemble(self.p,self.m)
    def test_egress_cap(self):
        with self.seeder(limit=CHUNK) as s:
            with self.assertRaisesRegex(Denied,'NO_VALID_CHUNK_SOURCE'): fetch(self.target,self.p,self.m,[(s.server_port,self.token)])
            self.assertEqual(s.remaining,0)
    def test_untrusted_publisher(self):
        with self.assertRaisesRegex(Denied,'UNTRUSTED_OR_REVOKED'): Policy([]).check(self.m)
    def test_corrupt_chunk_not_published(self):
        with self.assertRaisesRegex(Denied,'CHUNK_HASH_MISMATCH'): self.target.put(self.m['chunks'][0],b'bad')
        self.assertEqual(self.target.bytes(),0)
