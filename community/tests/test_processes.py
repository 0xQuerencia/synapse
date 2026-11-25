import json
import os
import re
import select
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from synapse_testnet.pilot import provision,request,read_private,Client
from synapse_testnet.ops import summary
from synapse_testnet.service import Service

class ProcessTests(unittest.TestCase):
    def test_independent_worker_and_verifier_processes(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); provision(root); processes=[]
            def start(module,*args):
                p=subprocess.Popen([sys.executable,'-m',module,*map(str,args)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True); processes.append(p); return p
            def ready(p,expected):
                end=time.monotonic()+8
                while time.monotonic()<end:
                    if p.poll() is not None: self.fail('process exited: '+p.stderr.read()[:1000])
                    if select.select([p.stdout],[],[],.1)[0]:
                        line=p.stdout.readline().strip()
                        if expected in line:return line
                self.fail('process startup timeout')
            try:
                server=start('synapse_testnet.http_api','--db',root/'ledger.db','--port',0,'--testnet'); line=ready(server,'SYNAPSE TESTNET ONLY:'); port=int(line.rsplit(':',1)[1])
                for role in ('worker-a','worker-b'): ready(start('synapse_testnet.pilot','--root',root,'--port',port,'worker',role),'worker ready')
                ready(start('synapse_testnet.pilot','--root',root,'--port',port,'verifier'),'verifier ready')
                r=request(root,port); self.assertEqual(r['job']['state'],'settled'); self.assertEqual(r['after']['available'],88_000_000)
                for role in ('worker-a','worker-b'):
                    c=Client(port,read_private(root,role)['token']); self.assertEqual(c.call('/v1/credits/balance')['available'],6_000_000)
                self.assertEqual(summary(Service(root/'ledger.db'))['open_jobs'],0)
                # Graceful worker shutdown ends admission. No public endpoint is opened.
                processes[1].terminate(); processes[1].wait(8); self.assertEqual(processes[1].returncode,0)
                self.assertEqual(len([p for p in processes if p.poll() is None]),3)
            finally:
                for p in reversed(processes):
                    if p.poll() is None: p.terminate()
                    try: p.communicate(timeout=5)
                    except subprocess.TimeoutExpired: p.kill(); p.communicate()
