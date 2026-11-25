"""Experimental peer delivery of original synthetic artifacts over loopback.
Separate trust format from the Rust catalog; no third-party weights or arbitrary URLs.
"""
import contextlib
import hashlib
import json
import os
import re
import secrets
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from .ledger import canonical, digest, integer, require, sha, Denied
from .service import verify_sig

CHUNK=64*1024
MAX_ARTIFACT=16*1024*1024
MAX_CHUNKS=MAX_ARTIFACT//CHUNK
DOMAIN=b'synapse-testnet-artifact-v1\0'
MANIFEST_FIELDS={'version','kind','publisher','artifact','size','chunks','redistribute','expires','signature'}

def manifest_for(data,key,expires,redistribute=False):
    require(isinstance(data,bytes) and 0<len(data)<=MAX_ARTIFACT,'ARTIFACT_LIMIT')
    m={'version':1,'kind':'original-synthetic-test-data','publisher':key.public_key().public_bytes_raw().hex(),'artifact':sha(data),'size':len(data),'chunks':[sha(data[i:i+CHUNK]) for i in range(0,len(data),CHUNK)],'redistribute':redistribute,'expires':expires}
    return {**m,'signature':key.sign(DOMAIN+canonical(m)).hex()}

class Policy:
    def __init__(self,publishers,revoked=None,clock=None):
        self.publishers=frozenset(digest(x) for x in publishers); self.revoked=set(revoked or []); self.clock=clock or (lambda:int(time.time()))
        require(len(self.publishers)<=64,'PUBLISHER_LIMIT')
    def check(self,m):
        require(isinstance(m,dict) and set(m)==MANIFEST_FIELDS,'BAD_MANIFEST')
        require(type(m['version']) is int and m['version']==1 and m['kind']=='original-synthetic-test-data','UNSUPPORTED_ARTIFACT')
        publisher=digest(m['publisher']); artifact=digest(m['artifact']); size=integer(m['size'],1,MAX_ARTIFACT)
        require(publisher in self.publishers and publisher not in self.revoked and artifact not in self.revoked,'UNTRUSTED_OR_REVOKED',403)
        require(type(m['redistribute']) is bool and m['redistribute'],'REDISTRIBUTION_DENIED',403)
        expiry=integer(m['expires'],1,2**62); require(self.clock()<expiry<=self.clock()+86400,'ARTIFACT_EXPIRED_OR_TOO_LONG',403)
        chunks=m['chunks']; require(isinstance(chunks,list) and len(chunks)==(size+CHUNK-1)//CHUNK and len(chunks)<=MAX_CHUNKS,'BAD_CHUNK_INDEX')
        for chunk in chunks: digest(chunk)
        verify_sig(publisher,{k:v for k,v in m.items() if k!='signature'},m['signature'],DOMAIN)
        return m

class Store:
    """Single-process private content-addressed store, with resumable verified chunks."""
    def __init__(self,root,quota):
        self.root=Path(root); integer(quota,1,64*MAX_ARTIFACT); self.quota=quota
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        st=self.root.lstat(); require(stat.S_ISDIR(st.st_mode) and st.st_uid==os.getuid() and not st.st_mode&0o077,'PRIVATE_ARTIFACT_DIR_REQUIRED')
        self.lock=threading.RLock()
        # No adoption of arbitrary files or same-account hostile process support.
        require(all(re.fullmatch(r'[0-9a-f]{64}',p.name) for p in self.root.iterdir()),'UNKNOWN_CACHE_CONTENT')
    def _read(self,d):
        digest(d); path=self.root/d
        try: fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        except FileNotFoundError: return None
        except OSError: raise Denied('CACHE_CORRUPT') from None
        with os.fdopen(fd,'rb') as f:
            st=os.fstat(f.fileno()); require(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_size<=CHUNK,'CACHE_CORRUPT')
            data=f.read(CHUNK+1)
        require(sha(data)==d,'CACHE_CORRUPT'); return data
    def bytes(self):
        total=0
        for p in self.root.iterdir():
            digest(p.name); st=p.lstat(); require(stat.S_ISREG(st.st_mode) and st.st_size<=CHUNK,'CACHE_CORRUPT'); total+=st.st_size
        return total
    def get(self,d):
        with self.lock: return self._read(d)
    def put(self,d,data):
        digest(d); require(isinstance(data,bytes) and 0<len(data)<=CHUNK and sha(data)==d,'CHUNK_HASH_MISMATCH')
        with self.lock:
            if self._read(d) is not None: return False
            require(self.bytes()+len(data)<=self.quota,'CACHE_QUOTA',507)
            tmp=self.root.parent/('.synapse-chunk-'+secrets.token_hex(16))
            try:
                fd=os.open(tmp,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
                # Same filesystem, atomic publish. Private local root, no external concurrent writers.
                os.link(tmp,self.root/d)
                dirfd=os.open(self.root,os.O_DIRECTORY)
                try: os.fsync(dirfd)
                finally: os.close(dirfd)
            finally: tmp.unlink(missing_ok=True)
            return True
    def import_data(self,policy,m,data):
        policy.check(m); require(len(data)==m['size'] and sha(data)==m['artifact'],'ARTIFACT_HASH_MISMATCH')
        for i,d in enumerate(m['chunks']): self.put(d,data[i*CHUNK:(i+1)*CHUNK])
    def assemble(self,policy,m):
        policy.check(m); out=bytearray()
        with self.lock:
            for i,d in enumerate(m['chunks']):
                piece=self._read(d); require(piece is not None,'CHUNK_MISSING',404)
                require(len(piece)==min(CHUNK,m['size']-i*CHUNK),'CHUNK_SIZE_MISMATCH'); out.extend(piece)
        require(len(out)==m['size'] and sha(out)==m['artifact'],'ARTIFACT_HASH_MISMATCH'); return bytes(out)

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

class Seeder(ThreadingHTTPServer):
    daemon_threads=True; request_queue_size=8
    def __init__(self,store,policy,manifests,token,egress_limit,port=0):
        require(isinstance(token,str) and 32<=len(token)<=128,'TOKEN_REQUIRED'); integer(egress_limit,1,64*MAX_ARTIFACT)
        require(len(manifests)<=32,'MANIFEST_LIMIT')
        self.store=store; self.policy=policy; self.manifests={m['artifact']:policy.check(m).copy() for m in manifests}; self.token=token; self.remaining=egress_limit
        self.guard=threading.Lock(); self.slots=threading.BoundedSemaphore(4)
        super().__init__(('127.0.0.1',port),SeedHandler)
    def get_request(self):
        sock,addr=super().get_request(); sock.settimeout(4); return sock,addr
    def process_request(self,request,address):
        if not self.slots.acquire(False): self.shutdown_request(request); return
        try: super().process_request(request,address)
        except BaseException: self.slots.release(); raise
    def process_request_thread(self,*args):
        try: super().process_request_thread(*args)
        finally: self.slots.release()

class SeedHandler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.0'
    def log_message(self,*args): pass
    def do_GET(self):
        try:
            require(self.headers.get_all('Host')==[f'127.0.0.1:{self.server.server_port}'] and not self.headers.get('Origin'),'ORIGIN_DENIED',403)
            auth=self.headers.get_all('Authorization') or []
            require(len(auth)==1 and secrets.compare_digest(auth[0],'Bearer '+self.server.token),'AUTH_REQUIRED',401)
            match=re.fullmatch(r'/v1/artifacts/([0-9a-f]{64})/chunks/([0-9]{1,3})',self.path); require(match is not None,'NOT_FOUND',404)
            art,index=match.group(1),int(match.group(2)); m=self.server.manifests.get(art); require(m is not None,'NOT_FOUND',404); self.server.policy.check(m)
            require(index<len(m['chunks']),'NOT_FOUND',404)
            data=self.server.store.get(m['chunks'][index]); require(data is not None,'NOT_FOUND',404)
            require(len(data)==min(CHUNK,m['size']-index*CHUNK),'CHUNK_SIZE_MISMATCH')
            with self.server.guard:
                require(self.server.remaining>=len(data),'EGRESS_QUOTA',429); self.server.remaining-=len(data)
            self.send_response(200); self.send_header('Content-Type','application/octet-stream'); self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
        except Denied as e: self.send_error(e.status,e.code)
        except (OSError,TimeoutError): return

def fetch(store,policy,m,peers,max_new=None):
    """Try explicitly configured loopback peers, not URLs from a manifest. No redirects or proxy."""
    policy.check(m); require(0<len(peers)<=4,'PEER_LIMIT')
    if max_new is not None: integer(max_new,0,MAX_CHUNKS)
    for port,token in peers:
        integer(port,1,65535); require(isinstance(token,str) and 32<=len(token)<=128,'TOKEN_REQUIRED')
    opener=build_opener(ProxyHandler({}),NoRedirect()); fetched=0; reused=0; failures=0
    for i,d in enumerate(m['chunks']):
        policy.check(m)
        if store.get(d) is not None: reused+=1; continue
        if max_new is not None and fetched>=max_new: return {'complete':False,'fetched':fetched,'reused':reused,'peer_failures':failures}
        for port,token in peers:
            try:
                req=Request(f'http://127.0.0.1:{port}/v1/artifacts/{m["artifact"]}/chunks/{i}',headers={'Authorization':'Bearer '+token})
                with opener.open(req,timeout=4) as r:
                    expected=min(CHUNK,m['size']-i*CHUNK); require(r.headers.get('Content-Length')==str(expected),'CHUNK_SIZE_MISMATCH'); data=r.read(expected+1)
                require(len(data)==expected and sha(data)==d,'CHUNK_HASH_MISMATCH')
            except (Denied,OSError,ValueError): failures+=1; continue
            store.put(d,data); fetched+=1; break
        else: raise Denied('NO_VALID_CHUNK_SOURCE',503)
    store.assemble(policy,m)
    return {'complete':True,'fetched':fetched,'reused':reused,'peer_failures':failures}
