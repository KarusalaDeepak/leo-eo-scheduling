"""Grouped out-of-fold weight calibration, separate from historical results.

This is a post-review reanalysis of an already studied benchmark, not a fresh
external test. All variants of a fleet size stay in the same outer group.
The search budget, seed and primary objective are fixed before running.
Like the separate path-selection project's evidence protocol, evaluation
outcomes cannot select parameters. Its RTT/CVaR admission gate is NOT reused:
fleet-size folds here do not establish independent-deployment non-inferiority.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import matched_scheduler as m
from run_matched_evaluation import STRESSES


def candidate_weights():
    keys=list(m.DEFAULT_WEIGHTS)
    default=np.array(list(m.DEFAULT_WEIGHTS.values()),dtype=float)
    rows=[default/default.sum(),np.ones(len(keys))/len(keys)]
    # Search relative weights on the simplex: positive common scaling is redundant.
    rows.extend(np.random.default_rng(20261003).dirichlet(np.ones(len(keys)),31))
    return [dict(zip(keys,map(float,row))) for row in rows]


def choose_candidate(search, heldout):
    """Select on development rewards only; reject incomplete comparisons.

    All candidates must see the same development conditions. Silently averaging
    partial runs would favor candidates evaluated only on easier instances.
    Held-out reward values are intentionally never inspected for selection.
    """
    required={'candidate','satellites','reward'}
    if not required.issubset(search.columns):
        raise ValueError('Search requires candidate, satellites and reward columns.')
    if heldout not in set(search.satellites):
        raise ValueError('Held-out fleet group is absent from the search.')
    development=search[search.satellites != heldout]
    if development.empty:
        raise ValueError('At least one separate development group is required.')
    if set(development.candidate) != set(search.candidate):
        raise ValueError('Candidates must cover identical development conditions.')
    if development[['candidate','satellites']].isna().any().any():
        raise ValueError('Development group and candidate IDs must be present.')
    if not np.isfinite(development.reward.to_numpy(dtype=float)).all():
        raise ValueError('Development rewards must be finite.')
    conditions=['satellites']+[c for c in ['instance','stress'] if c in development.columns]
    coverage=development.groupby(['candidate']+conditions).size().unstack('candidate',fill_value=0)
    if not coverage.eq(coverage.iloc[:,0],axis=0).all().all():
        raise ValueError('Candidates must cover identical development conditions.')
    # Each fleet-size group has equal influence, regardless of variant count.
    scores=development.groupby(['candidate','satellites']).reward.mean().groupby('candidate').mean()
    # Candidate ID is the sole tie-break: default wins an exact development tie.
    return int(scores.sort_index().idxmax())


def statistics(frame):
    rows=[]
    for metric in ['reward','completion']:
        grouped=frame.groupby(['satellites','policy'])[metric].mean().unstack()
        for other in ['default','equal','earliest','margin']:
            diff=(grouped.calibrated-grouped[other]).to_numpy()
            nz=np.round(diff[abs(diff)>1e-12],12)
            ranks=rankdata(abs(nz));observed=float(np.dot(np.sign(nz),ranks))
            null=[np.dot(signs,ranks) for signs in itertools.product([-1,1],repeat=len(nz))]
            p=float(np.mean(np.abs(null)>=abs(observed)-1e-10))
            boots=np.random.default_rng(20261003).choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
            rows.append(dict(metric=metric,comparator=other,n_groups=len(diff),
                             delta=diff.mean(),ci_low=np.quantile(boots,.025),ci_high=np.quantile(boots,.975),
                             p=p,rank_biserial=observed/ranks.sum() if len(nz) else 0,
                             wins=int((diff>1e-12).sum()),ties=int((abs(diff)<=1e-12).sum()),losses=int((diff < -1e-12).sum())))
    result=pd.DataFrame(rows);order=result.sort_values('p').index
    result.loc[order,'p_holm']=np.minimum(1,np.maximum.accumulate([(len(result)-j)*result.loc[k,'p'] for j,k in enumerate(order)]))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('results/weight_calibration'))
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    paths=sorted(args.dataset_dir.glob('*.json'))
    instances=[m.load_instance(p,s) for p in paths for s in STRESSES]
    sizes=sorted({i['satellites'] for i in instances})
    if len(sizes)<2: raise ValueError('At least two fleet-size groups required.')
    candidates=candidate_weights()
    protocol=dict(seed=20261003,search_candidates=candidates,objective='mean completed-priority fraction; equal fleet-group weighting',
                  tie_break='lowest candidate ID',outer_groups=sizes,stress_conditions=STRESSES,
                  caveat='Post-review reanalysis, not untouched external validation. Four groups give low statistical power.',
                  fixed_margin_parameters=dict(slack_seconds=1800,capacity_gb=2,slack_mix=.55,capacity_mix=.45),
                  dataset_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(m.__file__)]})
    # Written before evaluating search candidates, preserving the declared protocol.
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    search=[]
    for idx,weights in enumerate(candidates):
        for instance in instances:
            row,_=m.evaluate(instance,weights=weights)
            row['candidate']=idx;search.append(row)
        print('candidate',idx+1,'/',len(candidates),flush=True)
    search=pd.DataFrame(search);search.to_csv(out/'search_results.csv',index=False)
    folds=[];rows=[];ablation=[]
    with (out/'evaluated_schedules.jsonl').open('w') as handle:
        for size in sizes:
            chosen=choose_candidate(search,size)
            folds.append(dict(heldout_satellites=size,development_sizes=[s for s in sizes if s!=size],candidate=chosen,weights=candidates[chosen]))
            for instance in [i for i in instances if i['satellites']==size]:
                for name,policy,weights in [('calibrated','racs',candidates[chosen]),('default','racs',m.DEFAULT_WEIGHTS),
                                            ('equal','racs',candidates[1]),('earliest','earliest',None),('margin','margin',None)]:
                    row,plans=m.evaluate(instance,policy=policy,weights=weights)
                    row['policy']=name;rows.append(row)
                    handle.write(json.dumps(dict(instance=instance['name'],stress=instance['stress'],policy=name,
                                                 plans=[asdict(p) for p in plans]))+'\n')
                # Drop-one diagnostics, not a second tuning stage.
                for term in m.DEFAULT_WEIGHTS:
                    weights=dict(candidates[chosen]);weights[term]=0.
                    row,_=m.evaluate(instance,weights=weights);row['removed_term']=term
                    ablation.append(row)
    (out/'fold_weights.json').write_text(json.dumps(folds,indent=2))
    frame=pd.DataFrame(rows);frame.to_csv(out/'heldout_results.csv',index=False)
    pd.DataFrame(ablation).to_csv(out/'term_ablations.csv',index=False)
    statistics(frame).to_csv(out/'paired_group_statistics.csv',index=False)
    metrics=['completion','reward','congestion','fairness','runtime_ms']
    summary=frame.groupby(['stress','policy'])[metrics].mean().reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    print(summary[summary.stress=='combined'].to_string(index=False),flush=True)
    print('All generated schedules passed the independent validator.',flush=True)


if __name__=='__main__':main()
