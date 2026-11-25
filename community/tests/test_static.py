import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from synapse_testnet.demo import setup
from synapse_testnet.http_api import Server
class StaticTests(unittest.TestCase):
    def test_static_security_and_same_origin(self):
        with tempfile.TemporaryDirectory() as d:
            s,tokens,_=setup(Path(d)/'db')
            with Server(s) as server:
                t=threading.Thread(target=server.serve_forever);t.start();url=f'http://127.0.0.1:{server.server_port}'
                try:
                    with urllib.request.urlopen(url,timeout=3) as r:
                        self.assertIn(b'SYNTHETIC ONLY',r.read());self.assertIn("frame-ancestors 'none'",r.headers['Content-Security-Policy']);self.assertEqual(r.headers['Cache-Control'],'no-store')
                    with urllib.request.urlopen(url+'/static/app.js',timeout=3) as r:self.assertIn(b'textContent',r.read())
                    request=urllib.request.Request(url+'/v1/catalog',headers={'Authorization':'Bearer '+tokens['client'],'Origin':url})
                    with urllib.request.urlopen(request,timeout=3) as r:self.assertEqual(r.status,200)
                finally:server.shutdown();t.join()
