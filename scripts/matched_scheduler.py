"""Auditable revision of RACS with a common candidate pool for every selector.

Time is integer seconds, volume is decimal GB, and rate is Gbit/s. Supplied
contact endpoints are hard bounds. This is a batch planning model: all requests
are known in advance; their release times constrain execution, not knowledge.
The original experiment scripts remain a record of the submitted version.
"""
from __future__ import annotations

from collections import defaultdict
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np


# Inherited heuristic coefficients, not literature-derived or calibrated optima.
# Sensitivity experiments assess these choices; feasibility is checked separately.
DEFAULT_WEIGHTS = dict(margin=2.8, spread=0.4, congestion=2.2,
                       risk=1.6, relay=0.15, delay=0.15)
SELECTORS = ('earliest', 'margin', 'congestion', 'random', 'racs')


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


@dataclass
class Ledger:
    busy: dict = field(default_factory=dict)
    storage: dict = field(default_factory=dict)

    def clone(self):
        return Ledger({k: list(v) for k, v in self.busy.items()},
                      {k: list(v) for k, v in self.storage.items()})


@dataclass
class Plan:
    task_id: str
    observations: list
    deliveries: list
    relays: list
    reservations: list
    events: list
    completion: int
    volume: float
    capacity: float
    margin: float
    congestion: float
    risk: float
    spread: float
    delay: float
    variant: str

    def features(self):
        return [self.margin, self.congestion, self.risk, self.spread,
                len(self.relays), self.delay, self.volume,
                sum(e-b for _, b, e in self.reservations) / 3600,
                len(self.deliveries)]


def load_instance(path, stress='base'):
    data = json.loads(Path(path).read_text())
    start = timestamp(data['config']['start_utc'])
    factors = {'base': (1, 1, 1), 'tight_deadline': (.70, 1, 1),
               'reduced_bandwidth': (1, .65, 1), 'high_volume': (1, 1, 1.35),
               'combined': (.65, .60, 1.35)}
    deadline_factor, bandwidth_factor, volume_factor = factors[stress]
    tasks = []
    atoms = {}
    for raw in data['tasks']:
        arrival = int(timestamp(raw['arrival_time']) - start)
        deadline = int(timestamp(raw['deadline']) - start)
        task = dict(id=raw['id'], priority=int(raw['priority']), arrival=arrival,
                    deadline=arrival + int(max(60, (deadline-arrival)*deadline_factor)),
                    atoms=[dict(id=a['id'], duration=int(a['duration']),
                                volume=float(a['vol_gb'])*volume_factor)
                           for a in raw['atomic_tasks']])
        tasks.append(task)
        atoms.update({a['id']: a for a in task['atoms']})
    obs, downlinks, isl = defaultdict(list), defaultdict(list), defaultdict(list)
    discarded = 0
    for w in data['windows']['obs']:
        b, e = int(timestamp(w['start'])-start), int(timestamp(w['end'])-start)
        if e-b < atoms[w['atomic_task_id']]['duration']:
            discarded += 1
            continue
        obs[w['atomic_task_id']].append((w['sat_id'], b, e))
    for w in data['windows']['s2g']:
        b, e = int(timestamp(w['start'])-start), int(timestamp(w['end'])-start)
        rate = float(w['bandwidth_gbps'])*bandwidth_factor
        if e > b and rate > 0:
            downlinks[w['sat_id']].append((str(w['gs_id']), b, e, rate))
    for w in data['windows']['isl']:
        b, e = int(timestamp(w['start'])-start), int(timestamp(w['end'])-start)
        rate = float(w.get('capacity_gbps', 0))
        if e > b and rate > 0:
            isl[w['s1']].append((w['s2'], b, e, rate))
            isl[w['s2']].append((w['s1'], b, e, rate))
    for rows in obs.values(): rows.sort(key=lambda x: (x[1], x[0], x[2]))
    for rows in downlinks.values(): rows.sort(key=lambda x: (x[1], x[0]))
    for rows in isl.values(): rows.sort(key=lambda x: (x[1], x[0]))
    return dict(name=data['instance_name'], stress=stress, tasks=tasks, obs=obs,
                downlinks=downlinks, isl=isl, raw=data,
                storage={s['id']:float(s['storage_gb']) for s in data['satellites']},
                satellites=len(data['satellites']), discarded_obs=discarded)


def available(ledger, keys, b, e):
    return all(not (b < y and x < e) for k in keys for x,y in ledger.busy.get(k, []))


def first_slot(ledger, keys, b, e, duration):
    """Find an uninterrupted interval without extending a contact endpoint."""
    for x,y in sorted(pair for k in keys for pair in ledger.busy.get(k, [])):
        if b+duration <= x: break
        if b < y and x < b+duration: b = y
    return (b, b+duration) if b+duration <= e else None


def reserve(ledger, keys, b, e, records):
    for k in keys:
        ledger.busy.setdefault(k, []).append((b,e))
        records.append((k,b,e))


def buffer_valid(instance, ledger, events):
    merged = {k:list(v) for k,v in ledger.storage.items()}
    for sat,t,amount in events: merged.setdefault(sat, []).append((t,amount))
    for sat, rows in merged.items():
        # All simultaneous events are aggregated; intervals use [start,end).
        at_time = defaultdict(float)
        for t,amount in rows: at_time[t] += amount
        level = 0.
        for t in sorted(at_time):
            level += at_time[t]
            if level < -1e-7 or level > instance['storage'][sat]+1e-7: return False
    return True


def direct_delivery(instance, sat, volume, ready, deadline, ledger, mode, single):
    local = ledger.clone()
    chosen, records, events = [], [], []
    remaining = volume
    windows = instance['downlinks'].get(sat, [])
    if mode == 'capacity': windows = sorted(windows, key=lambda w: (-w[3]*(w[2]-w[1]),w[1]))
    for gs,b,e,rate in windows:
        # Whole-window reservation is conservative and identical across policies.
        keys = ('sat:'+sat, 'gs:'+gs)
        if b < ready or e > deadline or not available(local,keys,b,e): continue
        capacity = rate*(e-b)/8
        if single and capacity+1e-9 < remaining: continue
        amount = min(remaining,capacity)
        reserve(local,keys,b,e,records)
        chosen.append((sat,gs,b,e,rate,amount))
        events.append((sat,e,-amount))
        remaining -= amount
        if remaining <= 1e-9: return local,chosen,records,events
    return None


def delivery(instance, sat, volume, ready, deadline, ledger, mode, single):
    direct = direct_delivery(instance,sat,volume,ready,deadline,ledger,mode,single)
    if direct is not None: return (*direct, [])
    if single: return None
    # Store-and-forward one-hop fallback consumes both satellites during ISL.
    for relay,b,e,rate in instance['isl'].get(sat, []):
        keys = ('sat:'+sat, 'sat:'+relay, 'isl:'+':'.join(sorted((sat,relay))))
        interval = first_slot(ledger, keys, max(ready,b), min(deadline,e),
                              math.ceil(8*volume/rate))
        if interval is None: continue
        rb,re = interval
        local, records = ledger.clone(), []
        reserve(local,keys,rb,re,records)
        relayed = direct_delivery(instance,relay,volume,re,deadline,local,mode,False)
        if relayed is not None:
            next_ledger,chosen,dlrecords,events = relayed
            events = [(sat,re,-volume),(relay,re,volume)] + events
            return next_ledger,chosen,records+dlrecords,events,[(sat,relay,rb,re,rate,volume)]
    return None


def build_plan(instance, task, ledger, variant, force_sat=None):
    """One bounded greedy construction; alternatives share the same generator."""
    obs_mode, dl_mode = variant
    local = ledger.clone()
    records, observations, events, relays, deliveries = [], [], [], [], []
    volumes, ready = defaultdict(float), defaultdict(int)
    for atom in sorted(task['atoms'], key=lambda a: (-a['duration'],a['id'])):
        choices = []
        # Scan all supplied contacts for this atom: no top-k or fleet-size cap.
        # Batch planning includes future windows, not just visibility at arrival.
        for sat,b,e in instance['obs'].get(atom['id'], []):
            if force_sat is not None and sat != force_sat: continue
            interval = first_slot(local, ('sat:'+sat,), max(b,task['arrival']),
                                  min(e,task['deadline']), atom['duration'])
            if interval is None: continue
            x,y = interval
            load = sum(v-u for u,v in local.busy.get('sat:'+sat, []))
            if obs_mode == 'earliest': key = (y,load,sat)
            elif obs_mode == 'load': key = (load,y,sat)
            else:
                next_dl = min((de for _,db,de,_ in instance['downlinks'].get(sat, [])
                               if db >= y and de <= task['deadline']), default=10**12)
                key = (next_dl,y,load,sat)
            choices.append((key,sat,x,y))
        if not choices: return None
        _,sat,b,e = min(choices)
        reserve(local,('sat:'+sat,),b,e,records)
        observations.append((atom['id'],sat,b,e,atom['volume']))
        events.append((sat,e,atom['volume']))
        volumes[sat] += atom['volume']; ready[sat] = max(ready[sat],e)
    for sat in sorted(volumes):
        found = delivery(instance,sat,volumes[sat],ready[sat],task['deadline'],
                         local,dl_mode,force_sat is not None)
        if found is None: return None
        local,ds,rs,es,rels = found
        records.extend(rs); events.extend(es); relays.extend(rels); deliveries.extend(ds)
    if not deliveries or not buffer_valid(instance,ledger,events): return None
    completion = max(d[3] for d in deliveries)
    volume = sum(volumes.values()); capacity = sum((d[3]-d[2])*d[4]/8 for d in deliveries)
    sigmoid = lambda x: 1/(1+math.exp(-min(60,max(-60,x))))
    margin = .55*sigmoid((task['deadline']-completion)/1800)+.45*sigmoid((capacity-volume)/2)
    congestion = volume/max(capacity,1e-9)
    return Plan(task['id'],observations,deliveries,relays,records,events,completion,
                volume,capacity,margin,congestion,min(1,congestion)*(1-margin),
                len(volumes)/len(task['atoms']), (completion-task['arrival'])/3600,
                '-'.join(variant)+(('-'+force_sat) if force_sat else ''))


def candidate_plans(instance, task, ledger, single=False):
    plans, seen = [], set()
    variants = [('earliest','early'),('load','early'),('delivery','early'),
                ('earliest','capacity'),('load','capacity'),('delivery','capacity')]
    if single:
        satellites = sorted(set(s for s,_,_ in instance['obs'].get(task['atoms'][0]['id'],[])))
        specs = [(('earliest','early'),s) for s in satellites]
    else: specs = [(v,None) for v in variants]
    for variant,sat in specs:
        plan = build_plan(instance,task,ledger,variant,sat)
        if plan is None: continue
        key = (tuple(plan.observations),tuple(plan.deliveries),tuple(plan.relays))
        if key not in seen: plans.append(plan); seen.add(key)
    return plans


def score(plan, weights=None):
    w = DEFAULT_WEIGHTS if weights is None else weights
    # Priority and urgency are constant across plans of one task, so they cancel.
    return (w['margin']*plan.margin + w['spread']*plan.spread
            -w['congestion']*plan.congestion -w['risk']*plan.risk
            -w['relay']*len(plan.relays) -w['delay']*plan.delay)


def select(plans, policy, rng, weights=None, model=None):
    if policy == 'random': return plans[int(rng.integers(len(plans)))]
    if policy == 'learned': values = model.predict([p.features() for p in plans])
    else:
        fn = {'earliest':lambda p:-p.completion, 'margin':lambda p:p.margin,
              'congestion':lambda p:-p.congestion, 'racs':lambda p:score(p,weights)}[policy]
        values = [fn(p) for p in plans]
    return max(zip(values,plans),key=lambda vp:(vp[0],-vp[1].completion,vp[1].variant))[1]


def commit(plan, ledger):
    for key,b,e in plan.reservations: ledger.busy.setdefault(key, []).append((b,e))
    for sat,t,amount in plan.events: ledger.storage.setdefault(sat, []).append((t,amount))


def ordered_tasks(instance):
    return sorted(instance['tasks'],key=lambda t:(t['deadline']-t['arrival'],
                  -t['priority'],sum(a['volume'] for a in t['atoms']),t['id']))


def validate_schedule(instance, plans):
    """Independent post-run checks of contacts, payload, buffers and resources."""
    ledger = Ledger()
    tasks = {t['id']:t for t in instance['tasks']}
    assert len({p.task_id for p in plans}) == len(plans)
    for p in plans:
        t = tasks[p.task_id]; atoms = {a['id']:a for a in t['atoms']}
        assert {o[0] for o in p.observations} == set(atoms)
        assert len(p.observations) == len(atoms)
        expected_events,expected_reservations,ready_events = [],[],[]
        for aid,sat,b,e,amount in p.observations:
            assert e-b == atoms[aid]['duration'] and b >= t['arrival']
            assert abs(amount-atoms[aid]['volume']) < 1e-9
            assert any(s==sat and wb<=b and e<=we for s,wb,we in instance['obs'][aid])
            expected_events.append((sat,e,amount));expected_reservations.append(('sat:'+sat,b,e))
            ready_events.append((sat,e,amount))
        assert abs(p.volume-sum(a['volume'] for a in atoms.values())) < 1e-7
        assert abs(sum(d[5] for d in p.deliveries)-p.volume) < 1e-7
        for sat,relay,b,e,rate,volume in p.relays:
            assert any(r==relay and wb<=b and e<=we and rate==wr
                       for r,wb,we,wr in instance['isl'][sat])
            assert volume <= (e-b)*rate/8+1e-9
            expected_events.extend([(sat,e,-volume),(relay,e,volume)])
            ready_events.extend([(sat,b,-volume),(relay,e,volume)])
            for key in ('sat:'+sat,'sat:'+relay,'isl:'+':'.join(sorted((sat,relay)))):
                expected_reservations.append((key,b,e))
        for sat,gs,b,e,rate,amount in p.deliveries:
            assert (gs,b,e,rate) in instance['downlinks'][sat]
            assert amount > 0 and amount <= (e-b)*rate/8+1e-9 and e <= t['deadline']
            ready_events.append((sat,b,-amount))
            expected_events.append((sat,e,-amount))
            expected_reservations.extend([('sat:'+sat,b,e),('gs:'+gs,b,e)])
        assert p.completion == max(d[3] for d in p.deliveries)
        assert Counter(p.events)==Counter(expected_events)
        assert Counter(p.reservations)==Counter(expected_reservations)
        # Debit payload at transmission START to reject sending unavailable data.
        # A satellite may send its own tile before receiving a different tile.
        assert buffer_valid(instance,Ledger(),ready_events)
        # Storage balance rejects delivery before observation and double delivery.
        assert buffer_valid(instance,Ledger(),p.events)
        for key,b,e in p.reservations:
            assert b < e and available(ledger,(key,),b,e), (p.task_id,key,b,e)
            ledger.busy.setdefault(key, []).append((b,e))
        assert buffer_valid(instance,ledger,p.events)
        for sat,tm,v in p.events: ledger.storage.setdefault(sat, []).append((tm,v))
    for events in ledger.storage.values(): assert abs(sum(v for _,v in events)) < 1e-7


def evaluate(instance, policy='racs', single=False, seed=0, weights=None, model=None):
    ledger, accepted = Ledger(), []
    rng = np.random.default_rng(seed)
    candidate_count = 0
    tick = time.perf_counter()
    for task in ordered_tasks(instance):
        plans = candidate_plans(instance,task,ledger,single)
        candidate_count += len(plans)
        if plans:
            chosen = select(plans,policy,rng,weights,model)
            commit(chosen,ledger); accepted.append(chosen)
    elapsed = 1000*(time.perf_counter()-tick)
    validate_schedule(instance,accepted)
    tasks = instance['tasks']; accepted_ids = {p.task_id for p in accepted}
    class_rates = []
    for lo,hi in [(1,3),(4,6),(7,10)]:
        group = [t for t in tasks if lo<=t['priority']<=hi]
        if group: class_rates.append(sum(t['id'] in accepted_ids for t in group)/len(group))
    fairness = sum(class_rates)**2/(len(class_rates)*sum(r*r for r in class_rates)) if any(class_rates) else 0.
    row = dict(instance=instance['name'], stress=instance['stress'], satellites=instance['satellites'],
               policy=('single_' if single else 'full_')+policy, seed=seed,
               tasks=len(tasks), completed=len(accepted), missed=len(tasks)-len(accepted),
               completion=len(accepted)/len(tasks), miss=1-len(accepted)/len(tasks),
               reward=sum(t['priority'] for t in tasks if t['id'] in accepted_ids)/sum(t['priority'] for t in tasks),
               congestion=float(np.mean([p.congestion for p in accepted])) if accepted else np.nan,
               fairness=fairness,runtime_ms=elapsed,candidates=candidate_count,
               relay_uses=sum(len(p.relays) for p in accepted))
    return row,accepted
