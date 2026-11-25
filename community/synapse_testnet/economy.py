"""Seeded fictional capacity and credit simulation. Never treated as pilot telemetry."""
import argparse
import heapq
import json
import random
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class Scenario:
    name: str
    requesters: int=20
    providers: int=8
    ticks: int=300
    arrivals: int=3
    churn: float=.1
    failure: float=.08
    queue_timeout: int=8
    treasury: int=50_000
    intro_grant: int=2_000
    audit_rate: float=.1
    duplicate_attempt_rate: float=.1
    fraud_rate: float=.02
    relay_fraction: float=.25
    relay_bytes_per_job: int=50_000
    relay_cost_per_gb: float=0.0
    provider_watts: float=0.0
    electricity_per_kwh: float=0.0

SCENARIOS=[Scenario('baseline'),Scenario('zero-demand',arrivals=0),Scenario('zero-supply',providers=0),Scenario('burst',arrivals=20),Scenario('churn-50pct',churn=.5),Scenario('fraud-and-duplicates',fraud_rate=.4,duplicate_attempt_rate=.8),Scenario('grant-exhaustion',treasury=4000)]

def simulate(s,seed=42):
    rng=random.Random(seed); treasury=s.treasury; job_start={}
    clients=[0]*s.requesters; providers=[0]*s.providers; completions=Counter(); stats=Counter(); wait=[]; escrow=0; outstanding=[]; queued=[]; active=set(); sequence=0; relay_bytes=0; provider_ticks=0; low_power=Counter(); high_power=Counter()
    for i in range(s.requesters):
        grant=min(s.intro_grant,treasury); clients[i]=grant; treasury-=grant
        if grant<s.intro_grant: stats['grant_shortfall']+=1
    def conserve():
        assert min([treasury,escrow,*clients,*providers])>=0
        assert treasury+escrow+sum(clients)+sum(providers)==s.treasury
    for tick in range(s.ticks):
        while outstanding and outstanding[0][0]<=tick:
            end,job,client,worker,quote,actual,failed,relayed=heapq.heappop(outstanding)
            escrow-=quote; active.remove(worker); provider_ticks+=end-job_start[job]
            if failed or rng.random()<s.fraud_rate:
                clients[client]+=quote; stats['failed_or_rejected']+=1
            else:
                providers[worker]+=actual; clients[client]+=quote-actual; stats['completed']+=1; completions[worker]+=1
                (low_power if worker%2==0 else high_power)['completed']+=1
                if rng.random()<s.audit_rate: stats['audited']+=1
                if rng.random()<s.duplicate_attempt_rate: stats['duplicate_rejected']+=1
            if relayed: relay_bytes+=s.relay_bytes_per_job
            conserve()
        for _ in range(rng.randrange(s.arrivals+1)):
            if not clients: break
            sequence+=1; client=rng.randrange(len(clients)); queued.append((tick,sequence,client)); stats['arrivals']+=1
        remaining=[]
        for created,job,client in queued:
            if tick-created>=s.queue_timeout: stats['queue_expired']+=1; continue
            if clients[client]<100: stats['insufficient_credit']+=1; continue
            available=[w for w in range(len(providers)) if w not in active and rng.random()>=s.churn]
            if not available: remaining.append((created,job,client)); continue
            worker=min(available,key=lambda w:(completions[w],w)); quote=100; actual=rng.randrange(30,101); duration=rng.randrange(1,5)+(worker%2==0)
            clients[client]-=quote; escrow+=quote; active.add(worker); wait.append(tick-created); stats['admitted']+=1
            job_start[job]=tick
            heapq.heappush(outstanding,(tick+duration,job,client,worker,quote,actual,rng.random()<s.failure,rng.random()<s.relay_fraction)); conserve()
        queued=remaining
    for _,job,client,worker,quote,_,_,_ in outstanding: clients[client]+=quote; escrow-=quote; stats['shutdown_refunds']+=1
    stats['queue_expired']+=len(queued); conserve()
    waits=sorted(wait); p95=waits[min(len(waits)-1,(95*len(waits)+99)//100-1)] if waits else None
    return {'scenario':s.name,'seed':seed,'assumptions':s.__dict__,'simulated_only':True,'outcomes':dict(stats),'fulfillment':stats['completed']/stats['arrivals'] if stats['arrivals'] else None,'p95_queue_ticks':p95,'treasury':treasury,'requester_credits':sum(clients),'provider_credits':sum(providers),'escrow':escrow,'issued':s.treasury,'conserved':True,'provider_top_share':max(completions.values(),default=0)/stats['completed'] if stats['completed'] else None,'low_power_completed':low_power['completed'],'high_power_completed':high_power['completed'],'relay_bytes_assumed':relay_bytes,'estimated_relay_cost':relay_bytes/1e9*s.relay_cost_per_gb,'estimated_provider_electricity_cost':provider_ticks/3600*s.provider_watts/1000*s.electricity_per_kwh,'tick_seconds_assumed':1}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,default=42); p.add_argument('--relay-cost-per-gb',type=float,default=0.0); p.add_argument('--provider-watts',type=float,default=0.0); p.add_argument('--electricity-per-kwh',type=float,default=0.0); args=p.parse_args()
    from dataclasses import replace
    import math
    if not all(math.isfinite(x) and x>=0 for x in (args.relay_cost_per_gb,args.provider_watts,args.electricity_per_kwh)): p.error('cost assumptions must be finite and nonnegative')
    print(json.dumps([simulate(replace(s,relay_cost_per_gb=args.relay_cost_per_gb,provider_watts=args.provider_watts,electricity_per_kwh=args.electricity_per_kwh),args.seed) for s in SCENARIOS],indent=2))
if __name__=='__main__': main()
