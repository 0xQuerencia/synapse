"""Local-only, fake-credit pilot control service. No production identity or payment."""
import json
import secrets
import re
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from .ledger import Ledger, Denied, CAP, require, integer, ident, digest, canonical, sha, uid, split
from .synthetic import MODEL, observed

RECEIPT_FIELDS = {'assignment','job','model','epoch','input_units','output_units','evidence'}

def verify_sig(pubkey, body, sig, domain):
    try:
        require(isinstance(sig,str) and len(sig)==128, 'BAD_SIGNATURE',403)
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey)).verify(bytes.fromhex(sig),domain+canonical(body))
    except (ValueError, TypeError, InvalidSignature):
        raise Denied('BAD_SIGNATURE',403) from None

class Service(Ledger):
    def bootstrap(self, pool, operator, cap, consent='testnet-v1'):
        ident(pool); ident(operator); ident(consent); integer(cap,1)
        token=secrets.token_urlsafe(32)
        with self.tx() as db:
            require(not db.execute('SELECT 1 FROM pools WHERE id=?',(pool,)).fetchone(),'POOL_EXISTS',409)
            db.execute('INSERT INTO pools VALUES(?,0,?,?,?)',(pool,consent,cap,cap))
            db.execute('INSERT INTO members(id,pool,role,token_hash,consent) VALUES(?,?,?,?,?)',(operator,pool,'operator',sha(token.encode()),consent))
            issuance=self.account(db,pool,'issuance','system'); treasury=self.account(db,pool,'treasury','system')
            self.post(db,pool,'initial-issuance','issuance',{issuance:-cap,treasury:cap},[cap])
            self.audit(db,pool,operator,'bootstrap',pool,{'fake_credit_cap':cap})
        return token

    def actor(self, db, token, roles=None, consent=False):
        require(isinstance(token,str) and 32<=len(token)<=100,'AUTH_REQUIRED',401)
        a=db.execute('SELECT m.*, p.enabled AS pool_enabled,p.consent_version FROM members m JOIN pools p ON p.id=m.pool WHERE m.token_hash=?',(sha(token.encode()),)).fetchone()
        require(a is not None and a['enabled'],'AUTH_REQUIRED',401)
        require(roles is None or a['role'] in roles,'FORBIDDEN',403)
        if consent:
            require(a['pool_enabled'] and a['consent']==a['consent_version'],'POOL_OR_CONSENT_REQUIRED',403)
        return a

    def add_member(self, token, member, role):
        ident(member); require(role in ('requester','worker','verifier'),'INVALID_ROLE')
        credential=secrets.token_urlsafe(32)
        with self.tx() as db:
            a=self.actor(db,token,{'operator'})
            require(db.execute('SELECT COUNT(*) FROM members WHERE pool=?',(a['pool'],)).fetchone()[0]<128,'MEMBER_LIMIT',429)
            require(not db.execute('SELECT 1 FROM members WHERE id=?',(member,)).fetchone(),'MEMBER_EXISTS',409)
            db.execute('INSERT INTO members(id,pool,role,token_hash) VALUES(?,?,?,?)',(member,a['pool'],role,sha(credential.encode())))
            self.audit(db,a['pool'],a['id'],'member_enrolled',member,{'role':role})
        return credential

    def consent(self, token, version, external_compute):
        require(external_compute is True,'EXPLICIT_CONSENT_REQUIRED',403)
        with self.tx() as db:
            a=self.actor(db,token)
            require(version==a['consent_version'],'CONSENT_VERSION_MISMATCH',409)
            db.execute('UPDATE members SET consent=? WHERE id=?',(version,a['id']))
            self.audit(db,a['pool'],a['id'],'consent',a['id'],{'version':version})

    def set_pool(self, token, enabled):
        require(type(enabled) is bool,'INVALID_BOOLEAN')
        with self.tx() as db:
            a=self.actor(db,token,{'operator'})
            db.execute('UPDATE pools SET enabled=? WHERE id=?',(int(enabled),a['pool']))
            if not enabled:
                for job in db.execute("SELECT * FROM jobs WHERE pool=? AND state='running'",(a['pool'],)).fetchall():
                    self._refund(db,job,'failed')
            self.audit(db,a['pool'],a['id'],'pool_enabled' if enabled else 'pool_suspended',a['pool'])

    def approve_model(self, token, model, layers, max_context, synthetic_only):
        digest(model); integer(layers,1,256); integer(max_context,2,32768)
        require(synthetic_only is True and model==MODEL and layers==4 and max_context<=256,'TESTNET_SYNTHETIC_MODELS_ONLY',403)
        with self.tx() as db:
            a=self.actor(db,token,{'operator'})
            require(not db.execute('SELECT 1 FROM models WHERE pool=? AND digest=?',(a['pool'],model)).fetchone(),'MODEL_EXISTS',409)
            db.execute('INSERT INTO models VALUES(?,?,?,?,1)',(a['pool'],model,layers,max_context))
            self.audit(db,a['pool'],a['id'],'synthetic_model_approved',model)

    def disable_model(self, token, model):
        with self.tx() as db:
            a=self.actor(db,token,{'operator'}); self.row(db,'SELECT * FROM models WHERE pool=? AND digest=?',(a['pool'],model))
            db.execute('UPDATE models SET enabled=0 WHERE pool=? AND digest=?',(a['pool'],model))
            for j in db.execute("SELECT j.* FROM jobs j JOIN quotes q ON j.quote=q.id WHERE j.pool=? AND q.model=? AND j.state='running'",(a['pool'],model)).fetchall():
                self._refund(db,j,'failed')
            self.audit(db,a['pool'],a['id'],'model_disabled',model)

    def tariff(self, token, version, base, prefill, decode):
        ident(version); integer(base); integer(prefill); integer(decode)
        require(base+prefill+decode>0,'ZERO_TARIFF')
        with self.tx() as db:
            a=self.actor(db,token,{'operator'})
            require(not db.execute('SELECT 1 FROM tariffs WHERE pool=? AND version=?',(a['pool'],version)).fetchone(),'TARIFF_EXISTS',409)
            db.execute('INSERT INTO tariffs VALUES(?,?,?,?,?)',(a['pool'],version,base,prefill,decode))
            self.audit(db,a['pool'],a['id'],'tariff_created',version)

    def grant(self, token, member, amount):
        integer(amount,1)
        with self.tx() as db:
            a=self.actor(db,token,{'operator'}); p=a['pool']
            self.row(db,'SELECT * FROM members WHERE id=? AND pool=? AND enabled=1 AND role IN(\'requester\',\'worker\')',(member,p))
            prior=db.execute('SELECT amount FROM grants WHERE pool=? AND member=?',(p,member)).fetchone()
            if prior:
                require(prior['amount']==amount,'GRANT_ALREADY_USED',409)
                return
            treasury=self.account(db,p,'treasury','system'); target=self.account(db,p,'available',member)
            self.post(db,p,'grant:'+member,'grant',{treasury:-amount,target:amount},[member,amount])
            db.execute('INSERT INTO grants VALUES(?,?,?)',(p,member,amount))
            self.audit(db,p,a['id'],'grant',member,{'amount':amount})

    def approve_device(self, token, device, owner, pubkey, model, first, end, max_jobs=1):
        ident(device); digest(pubkey); digest(model); integer(first,0,255); integer(end,1,256); integer(max_jobs,1,4)
        with self.tx() as db:
            a=self.actor(db,token,{'operator'}); p=a['pool']
            self.row(db,"SELECT * FROM members WHERE id=? AND pool=? AND role='worker' AND enabled=1",(owner,p))
            m=self.row(db,'SELECT * FROM models WHERE pool=? AND digest=? AND enabled=1',(p,model))
            require(first<end<=m['layers'],'INVALID_RANGE')
            require(not db.execute('SELECT 1 FROM devices WHERE id=? OR pubkey=?',(device,pubkey)).fetchone(),'DEVICE_EXISTS',409)
            db.execute('INSERT INTO devices(id,owner,pubkey,model,first_layer,end_layer,max_jobs) VALUES(?,?,?,?,?,?,?)',(device,owner,pubkey,model,first,end,max_jobs))
            self.audit(db,p,a['id'],'device_approved',device)

    def device_challenge(self, token, device):
        with self.tx() as db:
            a=self.actor(db,token,{'worker'},True)
            d=self.row(db,'SELECT * FROM devices WHERE id=? AND owner=? AND enabled=1',(device,a['id']))
            nonce=uid(); expiry=self.clock()+60
            db.execute('UPDATE devices SET challenge=?,challenge_expires=? WHERE id=?',(nonce,expiry,device))
            return {'device':device,'owner':a['id'],'pool':a['pool'],'nonce':nonce,'expires':expiry}

    def prove_device(self, token, body, signature):
        require(isinstance(body,dict) and set(body)=={'device','owner','pool','nonce','expires'},'INVALID_PROOF')
        with self.tx() as db:
            a=self.actor(db,token,{'worker'},True)
            d=self.row(db,'SELECT * FROM devices WHERE id=? AND owner=? AND enabled=1',(body['device'],a['id']))
            expected={'device':d['id'],'owner':a['id'],'pool':a['pool'],'nonce':d['challenge'],'expires':d['challenge_expires']}
            require(body==expected and d['challenge'] and self.clock()<d['challenge_expires'],'EXPIRED_OR_REPLAYED_PROOF',403)
            verify_sig(d['pubkey'],body,signature,b'synapse-testnet-device-v1\0')
            db.execute("UPDATE devices SET proved=1,challenge=NULL,challenge_expires=NULL,mode='ready',ready_until=? WHERE id=?",(self.clock()+60,d['id']))
            self.audit(db,a['pool'],a['id'],'device_proved',d['id'])
            return {'heartbeat_sequence':d['heartbeat_seq'],'ready_until':self.clock()+60}

    def heartbeat(self, token, body, signature):
        require(isinstance(body,dict) and set(body)=={'device','sequence','issued'},'INVALID_HEARTBEAT')
        ident(body['device']); integer(body['sequence'],1,CAP); integer(body['issued'],1,2**62)
        with self.tx() as db:
            a=self.actor(db,token,{'worker'},True)
            d=self.row(db,'SELECT * FROM devices WHERE id=? AND owner=? AND enabled=1 AND proved=1',(body['device'],a['id']))
            require(abs(self.clock()-body['issued'])<=15 and body['sequence']>d['heartbeat_seq'],'STALE_HEARTBEAT',409)
            verify_sig(d['pubkey'],body,signature,b'synapse-testnet-heartbeat-v1\0')
            require(d['mode']!='stopped','DEVICE_STOPPED',409)
            db.execute('UPDATE devices SET ready_until=?,heartbeat_seq=? WHERE id=?',(self.clock()+60,body['sequence'],d['id']))
            return {'mode':d['mode'],'ready_until':self.clock()+60}

    def worker_control(self, token, device, action):
        require(action in ('pause','resume','drain','stop'),'INVALID_ACTION')
        with self.tx() as db:
            a=self.actor(db,token,{'worker'},True)
            d=self.row(db,'SELECT * FROM devices WHERE id=? AND owner=? AND enabled=1 AND proved=1',(device,a['id']))
            mode={'pause':'paused','resume':'ready','drain':'draining','stop':'stopped'}[action]
            if action=='resume':
                require(d['mode']!='stopped' and d['ready_until']>self.clock(),'FRESH_DEVICE_PROOF_REQUIRED',409)
            db.execute('UPDATE devices SET mode=? WHERE id=?',(mode,device))
            if action=='stop':
                jobs=db.execute("SELECT j.* FROM jobs j JOIN assignments s ON j.id=s.job WHERE s.device=? AND j.state='running'",(device,)).fetchall()
                for j in jobs: self._refund(db,j,'failed')
            self.audit(db,a['pool'],a['id'],'worker_'+action,device)
            return {'mode':mode}

    def revoke(self, token, target, kind):
        with self.tx() as db:
            a=self.actor(db,token,{'operator'}); p=a['pool']
            if kind=='member':
                self.row(db,"SELECT * FROM members WHERE id=? AND pool=? AND role!='operator'",(target,p))
                db.execute('UPDATE members SET enabled=0 WHERE id=?',(target,))
                db.execute('UPDATE devices SET enabled=0 WHERE owner=?',(target,))
                jobs=db.execute("SELECT DISTINCT j.* FROM jobs j LEFT JOIN assignments s ON s.job=j.id LEFT JOIN devices d ON d.id=s.device LEFT JOIN receipts r ON r.assignment=s.id WHERE j.pool=? AND j.state='running' AND (j.requester=? OR d.owner=? OR r.verifier=?)",(p,target,target,target)).fetchall()
            elif kind=='device':
                self.row(db,'SELECT d.* FROM devices d JOIN members m ON d.owner=m.id WHERE d.id=? AND m.pool=?',(target,p))
                db.execute('UPDATE devices SET enabled=0 WHERE id=?',(target,))
                jobs=db.execute("SELECT j.* FROM jobs j JOIN assignments s ON s.job=j.id WHERE j.state='running' AND s.device=?",(target,)).fetchall()
            else:
                raise Denied('INVALID_TARGET')
            for j in jobs: self._refund(db,j,'failed')
            self.audit(db,p,a['id'],'revoked_'+kind,target)

    def quote(self, token, model, tariff, input_units, max_output):
        digest(model); ident(tariff); integer(input_units,1,32768); integer(max_output,1,32768)
        with self.tx() as db:
            a=self.actor(db,token,{'requester','worker'},True); p=a['pool']
            require(db.execute('SELECT COUNT(*) FROM quotes WHERE pool=?',(p,)).fetchone()[0]<10000,'TESTNET_QUOTE_LIMIT',429)
            require(db.execute('SELECT COUNT(*) FROM quotes WHERE requester=? AND used=0 AND expires>?',(a['id'],self.clock())).fetchone()[0]<16,'ACTIVE_QUOTE_LIMIT',429)
            m=self.row(db,'SELECT * FROM models WHERE pool=? AND digest=? AND enabled=1',(p,model),'MODEL_UNSUPPORTED')
            require(input_units+max_output<=m['max_context'],'CONTEXT_LIMIT')
            t=self.row(db,'SELECT * FROM tariffs WHERE pool=? AND version=?',(p,tariff))
            maximum=integer(t['base']+t['prefill']*input_units+t['decode']*max_output,1)
            q=uid(); expiry=self.clock()+60
            db.execute('INSERT INTO quotes VALUES(?,?,?,?,?,?,?,?,?,0)',(q,p,a['id'],model,tariff,input_units,max_output,maximum,expiry))
            return {'quote':q,'maximum':maximum,'expires':expiry,'fake_credits':True}

    def _route(self, db, a, q):
        model=self.row(db,'SELECT * FROM models WHERE pool=? AND digest=? AND enabled=1',(a['pool'],q['model']))
        # One worker device per stage; full-model preferred, then fewest stages.
        rows=db.execute("SELECT d.* FROM devices d JOIN members m ON d.owner=m.id WHERE m.pool=? AND m.enabled=1 AND m.consent=? AND d.enabled=1 AND d.proved=1 AND d.mode='ready' AND d.ready_until>? AND d.model=? AND m.id!=? AND (SELECT COUNT(*) FROM assignments s JOIN jobs j ON s.job=j.id WHERE s.device=d.id AND j.state='running')<d.max_jobs ORDER BY d.id LIMIT 128",(a['pool'],a['consent_version'],self.clock(),q['model'],a['id'])).fetchall()
        routes={0: []}
        for start in range(model['layers']):
            if start not in routes or len(routes[start])>=4: continue
            for d in rows:
                if d['first_layer']!=start or d['end_layer']>model['layers']: continue
                candidate=routes[start]+[d]
                old=routes.get(d['end_layer'])
                if old is None or (len(candidate),[r['id'] for r in candidate])<(len(old),[r['id'] for r in old]): routes[d['end_layer']]=candidate
        require(model['layers'] in routes,'NO_ROUTE',503)
        return routes[model['layers']]

    def reserve(self, token, quote, idem):
        ident(quote); ident(idem)
        with self.tx() as db:
            a=self.actor(db,token,{'requester','worker'},True); p=a['pool']
            prev=db.execute('SELECT * FROM jobs WHERE requester=? AND idem=?',(a['id'],idem)).fetchone()
            if prev:
                require(prev['quote']==quote,'IDEMPOTENCY_CONFLICT',409)
                return self._job_view(db,prev)
            q=self.row(db,'SELECT * FROM quotes WHERE id=? AND pool=? AND requester=?',(quote,p,a['id']))
            require(not q['used'] and q['expires']>self.clock(),'QUOTE_EXPIRED_OR_USED',409)
            route=self._route(db,a,q)
            job=uid(); expiration=self.clock()+300
            db.execute("INSERT INTO jobs(id,pool,requester,quote,idem,state,expires) VALUES(?,?,?,?,?,'running',?)",(job,p,a['id'],quote,idem,expiration))
            source=self.account(db,p,'available',a['id']); escrow=self.account(db,p,'escrow',job)
            self.post(db,p,'reserve:'+job,'reserve',{source:-q['maximum'],escrow:q['maximum']},[quote,idem])
            for d in route:
                db.execute('INSERT INTO assignments VALUES(?,?,?,?,?,1)',(uid(),job,d['id'],d['first_layer'],d['end_layer']))
            db.execute('UPDATE quotes SET used=1 WHERE id=?',(quote,))
            self.event(db,p,'job_reserved',job); self.audit(db,p,a['id'],'job_reserved',job)
            return self._job_view(db,self.row(db,'SELECT * FROM jobs WHERE id=?',(job,)))

    def _job_view(self, db, job):
        q=self.row(db,'SELECT * FROM quotes WHERE id=?',(job['quote'],))
        stages=[dict(s) for s in db.execute('SELECT id,first_layer,end_layer,epoch FROM assignments WHERE job=? ORDER BY first_layer',(job['id'],))]
        return {'job':job['id'],'state':job['state'],'expires':job['expires'],'model':q['model'],'maximum':q['maximum'],'charged':job['charged'],'delivered':job['delivered'],'stages':stages,'fake_credits':True}

    def job(self, token, job):
        with self.read() as db:
            a=self.actor(db,token,{'requester','worker'}); j=self.row(db,'SELECT * FROM jobs WHERE id=? AND requester=?',(job,a['id']))
            return self._job_view(db,j)

    def assignments(self, token):
        with self.read() as db:
            a=self.actor(db,token,{'worker'},True)
            return [dict(r) for r in db.execute("SELECT s.id AS assignment,s.job,s.device,s.epoch,s.first_layer,s.end_layer,q.model,q.input_units,q.max_output,j.expires FROM assignments s JOIN devices d ON s.device=d.id JOIN jobs j ON s.job=j.id JOIN quotes q ON j.quote=q.id WHERE d.owner=? AND d.enabled=1 AND d.ready_until>? AND j.state='running' AND j.expires>? ORDER BY j.id,s.first_layer",(a['id'],self.clock(),self.clock()))]

    def receipt(self, token, body, signature):
        require(isinstance(body,dict) and set(body)==RECEIPT_FIELDS,'INVALID_RECEIPT')
        for key in ('assignment','job'): ident(body[key])
        digest(body['model']); digest(body['evidence']); integer(body['epoch'],1,1); integer(body['input_units'],1,32768); integer(body['output_units'],1,32768)
        with self.tx() as db:
            a=self.actor(db,token,{'worker'},True)
            r=self.row(db,'SELECT s.*,d.pubkey,d.enabled,d.ready_until,j.expires,j.state,q.model,q.input_units,q.max_output FROM assignments s JOIN devices d ON s.device=d.id JOIN jobs j ON s.job=j.id JOIN quotes q ON j.quote=q.id WHERE s.id=? AND d.owner=?',(body['assignment'],a['id']))
            require(r['enabled'] and body['job']==r['job'] and body['model']==r['model'] and body['epoch']==r['epoch'] and body['input_units']==r['input_units'] and body['output_units']<=r['max_output'],'RECEIPT_SCOPE',403)
            verify_sig(r['pubkey'],body,signature,b'synapse-testnet-receipt-v1\0')
            rh=sha(canonical(body)); prev=db.execute('SELECT * FROM receipts WHERE assignment=?',(r['id'],)).fetchone()
            if prev:
                require(prev['digest']==rh and prev['signature']==signature,'RECEIPT_CONFLICT',409)
                return {'status':prev['status'],'digest':rh}
            require(r['state']=='running' and min(r['expires'],r['ready_until'])>self.clock(),'LEASE_EXPIRED',409)
            db.execute("INSERT INTO receipts(assignment,body,signature,digest,output_units,input_units,status) VALUES(?,?,?,?,?,?,'pending')",(r['id'],canonical(body).decode(),signature,rh,body['output_units'],body['input_units']))
            self.audit(db,a['pool'],a['id'],'receipt_pending',r['id']); self.event(db,a['pool'],'receipt_pending',r['id'])
            return {'status':'pending','digest':rh}

    def pending(self, token):
        with self.read() as db:
            a=self.actor(db,token,{'verifier'},True)
            return [dict(r) for r in db.execute("SELECT r.assignment,r.digest,r.body,s.first_layer,s.end_layer FROM receipts r JOIN assignments s ON s.id=r.assignment JOIN jobs j ON j.id=s.job WHERE j.pool=? AND j.state='running' AND r.status='pending' AND j.expires>? ORDER BY r.assignment LIMIT 100",(a['pool'],self.clock()))]

    def validate(self, token, assignment, receipt_digest, accepted, observed_output, observed_evidence):
        digest(receipt_digest); digest(observed_evidence); integer(observed_output,0,32768); require(type(accepted) is bool,'INVALID_BOOLEAN')
        with self.tx() as db:
            a=self.actor(db,token,{'verifier'},True)
            r=self.row(db,'SELECT r.*,s.job,s.first_layer,s.end_layer,j.state,j.expires FROM receipts r JOIN assignments s ON s.id=r.assignment JOIN jobs j ON j.id=s.job WHERE r.assignment=? AND j.pool=?',(assignment,a['pool']))
            require(r['digest']==receipt_digest,'RECEIPT_CONFLICT',409)
            body=json.loads(r['body'])
            reference=observed({**body,'first_layer':r['first_layer'],'end_layer':r['end_layer']},r['output_units'])
            status='verified' if accepted and observed_output==r['output_units'] and observed_evidence==body['evidence']==reference else 'rejected'
            if r['status']!='pending':
                require(r['status']==status and r['verifier']==a['id'],'VALIDATION_CONFLICT',409)
                return r['status']
            require(r['state']=='running' and r['expires']>self.clock(),'LEASE_EXPIRED',409)
            db.execute('UPDATE receipts SET status=?,verifier=? WHERE assignment=?',(status,a['id'],assignment))
            self.audit(db,a['pool'],a['id'],'receipt_'+status,assignment)
            if status=='rejected': self._refund(db,self.row(db,'SELECT * FROM jobs WHERE id=?',(r['job'],)),'failed')
            return status

    def settle(self, token, job):
        with self.tx() as db:
            a=self.actor(db,token,{'requester','worker'},True)
            j=self.row(db,'SELECT * FROM jobs WHERE id=? AND requester=?',(job,a['id']))
            if j['state']=='settled': return self._job_view(db,j)
            require(j['state']=='running' and j['expires']>self.clock(),'LEASE_EXPIRED',409)
            q=self.row(db,'SELECT * FROM quotes WHERE id=?',(j['quote'],))
            rows=db.execute('SELECT s.*,d.owner,r.status,r.output_units,r.verifier,m.enabled AS owner_enabled,d.enabled AS device_enabled,d.ready_until,v.enabled AS verifier_enabled FROM assignments s JOIN devices d ON s.device=d.id JOIN members m ON d.owner=m.id LEFT JOIN receipts r ON s.id=r.assignment LEFT JOIN members v ON r.verifier=v.id WHERE s.job=? ORDER BY s.first_layer',(job,)).fetchall()
            require(rows and all(r['status']=='verified' and r['owner_enabled'] and r['device_enabled'] and r['ready_until']>self.clock() and r['verifier_enabled'] for r in rows),'WORK_NOT_VERIFIED',409)
            counts={r['output_units'] for r in rows}; require(len(counts)==1,'INCONSISTENT_DELIVERY',409)
            count=counts.pop(); t=self.row(db,'SELECT * FROM tariffs WHERE pool=? AND version=?',(j['pool'],q['tariff']))
            cost=integer(t['base']+t['prefill']*q['input_units']+t['decode']*count,1,q['maximum'])
            escrow=self.account(db,j['pool'],'escrow',job); requester=self.account(db,j['pool'],'available',j['requester'])
            amounts={escrow:-q['maximum'],requester:q['maximum']-cost}
            shares=split(cost,{r['id']:r['end_layer']-r['first_layer'] for r in rows})
            for r in rows:
                account=self.account(db,j['pool'],'available',r['owner']); amounts[account]=amounts.get(account,0)+shares[r['id']]
            tx=self.post(db,j['pool'],'settle:'+job,'settle',amounts,[job,count])
            db.execute("UPDATE jobs SET state='settled',delivered=?,charged=?,resolution_tx=? WHERE id=?",(count,cost,tx,job))
            self.audit(db,j['pool'],a['id'],'job_settled',job,{'charge':cost}); self.event(db,j['pool'],'job_settled',job)
            return self._job_view(db,self.row(db,'SELECT * FROM jobs WHERE id=?',(job,)))

    def _refund(self, db, j, state):
        if j['state']!='running': return
        q=self.row(db,'SELECT * FROM quotes WHERE id=?',(j['quote'],)); p=j['pool']; job=j['id']
        escrow=self.account(db,p,'escrow',job); requester=self.account(db,p,'available',j['requester'])
        tx=self.post(db,p,'refund:'+job,'refund',{escrow:-q['maximum'],requester:q['maximum']},[job,state])
        db.execute("UPDATE jobs SET state=?,delivered=0,charged=0,resolution_tx=? WHERE id=?",(state,tx,job))
        self.event(db,p,'job_'+state,job); self.audit(db,p,'system','job_'+state,job)

    def cancel(self, token, job):
        with self.tx() as db:
            a=self.actor(db,token,{'requester','worker'})
            j=self.row(db,'SELECT * FROM jobs WHERE id=? AND requester=?',(job,a['id']))
            self._refund(db,j,'cancelled')
            return self._job_view(db,self.row(db,'SELECT * FROM jobs WHERE id=?',(job,)))

    def sweep(self):
        with self.tx() as db:
            jobs=db.execute("SELECT DISTINCT j.* FROM jobs j JOIN assignments a ON a.job=j.id JOIN devices d ON d.id=a.device WHERE j.state='running' AND (j.expires<=? OR d.ready_until<=?)",(self.clock(),self.clock())).fetchall()
            for j in jobs: self._refund(db,j,'expired' if j['expires']<=self.clock() else 'failed')
            return len(jobs)

    def balance(self, token):
        with self.read() as db:
            a=self.actor(db,token,{'requester','worker'})
            bal=db.execute("SELECT balance FROM accounts WHERE pool=? AND kind='available' AND owner=?",(a['pool'],a['id'])).fetchone()
            held=db.execute("SELECT COALESCE(SUM(q.maximum),0) FROM jobs j JOIN quotes q ON j.quote=q.id WHERE j.requester=? AND j.state='running'",(a['id'],)).fetchone()[0]
            pending=db.execute("SELECT COUNT(*) FROM receipts r JOIN assignments s ON r.assignment=s.id JOIN devices d ON s.device=d.id JOIN jobs j ON s.job=j.id WHERE d.owner=? AND r.status='pending' AND j.state='running'",(a['id'],)).fetchone()[0]
            return {'available':bal[0] if bal else 0,'held':held,'pending_receipts':pending,'fake_credits':True}

    def history(self, token, limit=50):
        integer(limit,1,100)
        with self.read() as db:
            a=self.actor(db,token,{'requester','worker'})
            return [dict(r) for r in db.execute("SELECT j.id,j.kind,j.created,e.amount FROM journal j JOIN entries e ON e.tx=j.id JOIN accounts ac ON ac.id=e.account WHERE ac.owner=? AND ac.pool=? AND ac.kind='available' ORDER BY j.created DESC,j.id LIMIT ?",(a['id'],a['pool'],limit))]

    def stats(self, token):
        with self.read() as db:
            a=self.actor(db,token,{'worker'})
            n=db.execute("SELECT COUNT(DISTINCT j.id) FROM jobs j JOIN assignments s ON j.id=s.job JOIN devices d ON d.id=s.device JOIN receipts r ON r.assignment=s.id WHERE d.owner=? AND j.state='settled' AND r.status='verified'",(a['id'],)).fetchone()[0]
            return {'verified_jobs':n,'badges':['First Verified Job'] if n else [],'fake_credits':True}

    def dispute(self, token, job, reason):
        require(reason in ('incorrect_charge','incorrect_reward','incorrect_outcome'),'INVALID_REASON')
        with self.tx() as db:
            a=self.actor(db,token,{'requester','worker'}); j=self.row(db,'SELECT * FROM jobs WHERE id=? AND pool=?',(job,a['pool']))
            require(j['state']!='running','JOB_NOT_TERMINAL',409)
            eligible=j['requester']==a['id'] or db.execute('SELECT 1 FROM assignments s JOIN devices d ON d.id=s.device WHERE s.job=? AND d.owner=?',(job,a['id'])).fetchone()
            require(eligible,'NOT_FOUND',404)
            prev=db.execute('SELECT * FROM disputes WHERE job=? AND owner=?',(job,a['id'])).fetchone()
            if prev: return prev['id']
            dispute=uid(); db.execute("INSERT INTO disputes(id,job,owner,reason,state) VALUES(?,?,?,?,'open')",(dispute,job,a['id'],reason))
            self.audit(db,a['pool'],a['id'],'dispute_open',dispute); return dispute

    def review_dispute(self, token, dispute, compensation):
        integer(compensation,0)
        with self.tx() as db:
            a=self.actor(db,token,{'verifier'},True)
            d=self.row(db,'SELECT d.*,j.charged,j.requester,q.maximum FROM disputes d JOIN jobs j ON j.id=d.job JOIN quotes q ON q.id=j.quote WHERE d.id=? AND j.pool=?',(dispute,a['pool']))
            require(d['state']=='open' and compensation<=d['maximum'],'DISPUTE_SCOPE',409)
            require(not db.execute('SELECT 1 FROM receipts r JOIN assignments s ON s.id=r.assignment WHERE s.job=? AND r.verifier=?',(d['job'],a['id'])).fetchone(),'INDEPENDENT_DISPUTE_REVIEW_REQUIRED',403)
            db.execute('UPDATE disputes SET state=?,reviewer=?,compensation=? WHERE id=?',('approved' if compensation else 'rejected',a['id'],compensation,dispute))
            self.audit(db,a['pool'],a['id'],'dispute_reviewed',dispute)

    def resolve_dispute(self, token, dispute):
        with self.tx() as db:
            a=self.actor(db,token,{'operator'}); p=a['pool']
            d=self.row(db,'SELECT d.* FROM disputes d JOIN jobs j ON j.id=d.job WHERE d.id=? AND j.pool=?',(dispute,p))
            if d['state']=='resolved': return
            require(d['state']=='approved','DISPUTE_NOT_APPROVED',409)
            treasury=self.account(db,p,'treasury','system'); dest=self.account(db,p,'available',d['owner'])
            self.post(db,p,'dispute:'+dispute,'compensation',{treasury:-d['compensation'],dest:d['compensation']},[dispute,d['reviewer']])
            db.execute("UPDATE disputes SET state='resolved' WHERE id=?",(dispute,))
            self.audit(db,p,a['id'],'dispute_resolved',dispute)
