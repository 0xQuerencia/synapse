import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path
from synapse_testnet.demo import setup
from synapse_testnet.http_api import Server
from synapse_testnet.ledger import Denied, canonical, sha
from synapse_testnet.outbox import Sink, drain
from synapse_testnet.pilot import Client
from synapse_testnet.synthetic import MODEL

class OutboxTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.s,self.t,_=setup(self.root/'ledger.db');self.op=self.t['operator'];self.sink=Sink(self.root/'events.db')
        q=self.s.quote(self.t['client'],MODEL,'demo-v1',2,8)['quote']
        j=self.s.reserve(self.t['client'],q,'outbox-test')['job'];self.s.cancel(self.t['client'],j)
        self.server=Server(self.s);thread=threading.Thread(target=self.server.serve_forever);thread.start()
        def close():self.server.shutdown();thread.join();self.server.server_close()
        self.addCleanup(close);self.client=Client(self.server.server_port,self.op)

    def test_http_export_and_idempotent_retry(self):
        batch=self.s.pending_events(self.op);self.assertGreaterEqual(len(batch),2)
        # Crash boundary: durable sink write before ack; restart repeats safely.
        self.sink.store(batch[0]);reopened=Sink(self.root/'events.db')
        r=drain(self.client,reopened);self.assertEqual(r['acknowledged'],len(batch));self.assertEqual(r['sink_rows'],len(batch))
        self.assertEqual(drain(self.client,reopened)['acknowledged'],0)
        self.assertEqual(self.s.operator_status(self.op)['pending_events'],0)
        self.assertEqual(self.s.acknowledge_event(self.op,batch[0]['id'],sha(canonical(batch[0])))['acknowledged'],True)

    def test_parallel_export(self):
        n=len(self.s.pending_events(self.op))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:list(p.map(lambda _:drain(self.client,Sink(self.root/'events.db')),range(3)))
        self.assertEqual(self.sink.count(),n);self.assertFalse(self.s.pending_events(self.op))

    def test_pool_scope_and_corruption(self):
        e=self.s.pending_events(self.op)[0];other=self.s.bootstrap('other','op2',100)
        for token,code in [(self.t['client'],'FORBIDDEN'),(other,'NOT_FOUND')]:
            with self.assertRaises(Denied) as cm:self.s.acknowledge_event(token,e['id'],sha(canonical(e)))
            self.assertEqual(cm.exception.code,code)
        with self.assertRaises(Denied):self.s.acknowledge_event(self.op,e['id'],'0'*64)
        self.sink.store(e)
        with self.assertRaises(Denied):self.sink.store({**e,'kind':'changed'})
        self.assertEqual(self.sink.count(),1);self.assertIn(e,self.s.pending_events(self.op))

    def test_sink_rejects_public_file_and_symlink(self):
        bad=self.root/'bad';bad.write_text('');bad.chmod(0o644)
        with self.assertRaises(Denied):Sink(bad)
        link=self.root/'link';link.symlink_to(self.sink.path)
        with self.assertRaises(Denied):Sink(link)

if __name__=='__main__':unittest.main()
