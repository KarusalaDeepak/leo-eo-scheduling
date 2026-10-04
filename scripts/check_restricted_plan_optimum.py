"""Exact comparison ONLY within an identical frozen candidate-plan pool.

Unlike the submitted MILP comparison, the greedy and MILP schedules here both
select from the same empty-ledger plans. Neither bounds the dynamic scheduler.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
import matched_scheduler as m


def solve(instance):
    tasks=m.ordered_tasks(instance)
    plans=[p for task in tasks for p in m.candidate_plans(instance,task,m.Ledger())]
    if not plans:
        return dict(instance=instance['name'],stress=instance['stress'],satellites=instance['satellites'],
                    plans=0,milp_status=0,success=True,mip_gap=0.,seconds=0.,milp_reward=0,
                    frozen_greedy_reward=0,milp_completion=0.,frozen_greedy_completion=0.)
    constraints=[]
    # One plan per task; each resource interval clique can admit at most one.
    by_task=defaultdict(list);by_resource=defaultdict(list)
    for idx,p in enumerate(plans):
        by_task[p.task_id].append(idx)
        for key,b,e in p.reservations: by_resource[key].append((idx,b,e))
    for indices in by_task.values(): constraints.append(({j:1. for j in indices},1.))
    seen=set()
    for rows in by_resource.values():
        for t in sorted({b for _,b,_ in rows}):
            clique=tuple(sorted({idx for idx,b,e in rows if b<=t<e}))
            if len(clique)>1 and clique not in seen:
                seen.add(clique);constraints.append(({j:1. for j in clique},1.))
    # Memory occupancy is additive across selected plans at each event epoch.
    for sat,capacity in instance['storage'].items():
        epochs=sorted({t for p in plans for s,t,_ in p.events if s==sat})
        events=defaultdict(list)
        for idx,p in enumerate(plans):
            for s,t,v in p.events:
                if s==sat: events[t].append((idx,v))
        occupancy=np.zeros(len(plans))
        for t in epochs:
            for idx,v in events[t]: occupancy[idx]+=v
            if occupancy.sum()>capacity:
                constraints.append(({idx:float(v) for idx,v in enumerate(occupancy) if v>1e-8},capacity))
    matrix=lil_matrix((len(constraints),len(plans)))
    for r,(coeffs,_) in enumerate(constraints):
        for c,v in coeffs.items():matrix[r,c]=v
    priority={t['id']:t['priority'] for t in tasks}
    objective=np.array([priority[p.task_id] for p in plans],float)
    tick=time.perf_counter()
    result=milp(-objective,integrality=np.ones(len(plans)),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix.tocsr(),-np.inf,[v for _,v in constraints]),
                options={'time_limit':20.,'mip_rel_gap':0.0})
    seconds=time.perf_counter()-tick
    selected=[p for p,x in zip(plans,result.x) if x>.5] if result.x is not None else []
    m.validate_schedule(instance,selected)
    ledger=m.Ledger();greedy=[]
    for task in tasks:
        feasible=[]
        for idx in by_task[task['id']]:
            p=plans[idx]
            if all(m.available(ledger,(key,),b,e) for key,b,e in p.reservations) and m.buffer_valid(instance,ledger,p.events):feasible.append(p)
        if feasible:
            chosen=m.select(feasible,'racs',np.random.default_rng(0));m.commit(chosen,ledger);greedy.append(chosen)
    m.validate_schedule(instance,greedy)
    return dict(instance=instance['name'],stress=instance['stress'],satellites=instance['satellites'],
                plans=len(plans),milp_status=result.status,success=result.success,
                mip_gap=getattr(result,'mip_gap',np.nan),seconds=seconds,
                milp_reward=sum(priority[p.task_id] for p in selected),
                frozen_greedy_reward=sum(priority[p.task_id] for p in greedy),
                milp_completion=len(selected)/len(tasks),frozen_greedy_completion=len(greedy)/len(tasks))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('results/reviewer_revision'))
    args=parser.parse_args();rows=[]
    for stress in ['base','combined']:
        for path in sorted(args.dataset_dir.glob('*.json')):
            row=solve(m.load_instance(path,stress));rows.append(row);print(row,flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir/'restricted_plan_optimum.csv',index=False)
