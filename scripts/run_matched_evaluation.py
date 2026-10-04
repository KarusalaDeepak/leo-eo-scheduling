"""Reproduce the four-page revision; no submitted-version results are overwritten."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.mpl-cache'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor

import matched_scheduler as m

STRESSES = ['base','tight_deadline','reduced_bandwidth','high_volume','combined']
METRICS = ['completion','miss','reward','congestion','fairness','runtime_ms']


def learning_examples(instance):
    """Offline labels: current reward + greedy reward of the next three tasks.

    Labels may use future requests only in training instances. Every evaluation
    constellation-size group, including all its stress variants, is held out.
    This small learned ranker is a baseline, not a reproduction of published DRL.
    """
    X,y = [],[]
    ledger = m.Ledger()
    tasks = m.ordered_tasks(instance)
    rng = np.random.default_rng(7)
    for idx,task in enumerate(tasks):
        candidates = m.candidate_plans(instance,task,ledger)
        for plan in candidates:
            rollout = ledger.clone(); m.commit(plan,rollout)
            reward = task['priority']/10
            for upcoming in tasks[idx+1:idx+4]:
                next_plans = m.candidate_plans(instance,upcoming,rollout)
                if next_plans:
                    chosen = m.select(next_plans,'earliest',rng)
                    m.commit(chosen,rollout); reward += upcoming['priority']/10
            X.append(plan.features()); y.append(reward)
        if candidates: m.commit(m.select(candidates,'earliest',rng),ledger)
    return X,y


def paired_evidence(frame):
    rows = []
    # Replicate seeds are averaged inside each original instance, never treated
    # as independent samples. Stress transforms are tested separately.
    grouped = frame.groupby(['stress','instance','policy'])[METRICS].mean().reset_index()
    for stress,sub in grouped.groupby('stress'):
        table = sub.pivot(index='instance',columns='policy',values='completion')
        for other in table.columns:
            if other == 'full_racs': continue
            diff = (table.full_racs-table[other]).dropna().to_numpy()
            rng = np.random.default_rng(2026)
            boots = rng.choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
            nonzero = np.round(diff[abs(diff)>1e-12],12)
            ranks = rankdata(abs(nonzero))
            observed = float(np.dot(np.sign(nonzero),ranks))
            # Enumerate all signs (at most 2^9) rather than a small-sample
            # Gaussian approximation. Average ranks handle tied magnitudes.
            null = [float(np.dot(signs,ranks)) for signs in itertools.product([-1,1],repeat=len(ranks))]
            p = sum(abs(v)>=abs(observed)-1e-10 for v in null)/len(null)
            rows.append(dict(stress=stress,comparison='full_racs vs '+other,n=len(diff),
                             delta=float(diff.mean()),ci_low=float(np.quantile(boots,.025)),
                             ci_high=float(np.quantile(boots,.975)),p=p,
                             rank_biserial=observed/sum(ranks) if len(ranks) else 0.,
                             wins=int((diff>1e-12).sum()),ties=int((abs(diff)<=1e-12).sum()),
                             losses=int((diff< -1e-12).sum())))
    evidence = pd.DataFrame(rows)
    # One Holm family includes every displayed stress/comparator test.
    ordered = evidence.sort_values('p').index
    adjusted = np.maximum.accumulate([(len(evidence)-i)*evidence.loc[idx,'p'] for i,idx in enumerate(ordered)])
    evidence.loc[ordered,'p_holm'] = np.minimum(1,adjusted)
    return evidence


def figures(frame, sensitivity, output):
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    grouped = frame.groupby(['stress','instance','policy'])[METRICS].mean().reset_index()
    fig,axes = plt.subplots(1,2,figsize=(7.1,2.55),layout='constrained')
    labels = {'single_margin':'Single margin','full_earliest':'Full earliest',
              'full_margin':'Full margin','full_congestion':'Full congestion',
              'full_racs':'RACS','full_learned':'Full learned','full_random':'Full random'}
    for policy,label in labels.items():
        means,lows,highs = [],[],[]
        for stress in STRESSES:
            vals = grouped.query('policy==@policy and stress==@stress').completion.to_numpy()
            rng=np.random.default_rng(2026)
            boots=rng.choice(vals,(3000,len(vals)),replace=True).mean(axis=1)
            means.append(vals.mean());lo,hi=np.quantile(boots,[.025,.975]);lows.append(lo);highs.append(hi)
        axes[0].plot(range(5),means,marker='o',markersize=3,label=label)
        # Uncertainty is shown separately in CSV; error bars only on RACS to
        # avoid an unreadable overlay of six broad nine-instance intervals.
        if policy=='full_racs': axes[0].errorbar(range(5),means,yerr=[np.array(means)-lows,np.array(highs)-means],fmt='none',capsize=2,color='black',alpha=.5)
    axes[0].set_xticks(range(5),['Base','Tight','BW','Volume','Joint'])
    axes[0].set_ylabel('Completed fraction');axes[0].grid(alpha=.2)
    axes[0].set_title('(a) Matched scheduling options')
    base=frame.query('stress=="base" and policy=="full_racs" and ~instance.str.contains("Congested|Tiny")',engine='python').sort_values('satellites')
    axes[1].plot(base.satellites,base.runtime_ms,'o-',color='#176d81')
    axes[1].set_xlabel('Satellites (tasks increase too)');axes[1].set_ylabel('Scheduling time (ms)')
    axes[1].set_title('(b) Measured RACS runtime');axes[1].grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    fig.savefig(output/'matched_results.pdf');fig.savefig(output/'matched_results.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(4,2.3),layout='constrained')
    for term,rows in sensitivity.groupby('term'):
        avg=rows.groupby('factor').completion.mean()
        ax.plot(avg.index,avg.values,marker='o',label=term)
    ax.set_xlabel('Coefficient multiplier');ax.set_ylabel('Completed fraction (joint stress)')
    ax.legend(ncol=3,fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.3));ax.grid(alpha=.2)
    fig.savefig(output/'weight_sensitivity.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('results/reviewer_revision'))
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    paths=sorted(args.dataset_dir.glob('*.json'))
    if not paths: raise ValueError('No benchmark JSON instances found.')
    base=[m.load_instance(p) for p in paths]
    # Fleet sizes come from input data, never from a fixed satellite-count cap.
    if len({i['satellites'] for i in base}) < 2:
        raise ValueError('Held-out-size learning requires at least two fleet sizes.')
    audit=[]
    for p,i in zip(paths,base):
        audit.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                          satellites=i['satellites'],tasks=len(i['tasks']),
                          observations=len(i['raw']['windows']['obs']),
                          excluded_short_observations=i['discarded_obs'],config=i['raw']['config']))
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print('Building independent training labels on base instances...',flush=True)
    tick=time.perf_counter();training={i['name']:learning_examples(i) for i in base}
    label_seconds=time.perf_counter()-tick
    models={};learning_log=[]
    for size in sorted({i['satellites'] for i in base}):
        X,y=[],[];train_names=[];test_names=[]
        for i in base:
            if i['satellites']==size: test_names.append(i['name']);continue
            xs,ys=training[i['name']];X.extend(xs);y.extend(ys);train_names.append(i['name'])
        model=RandomForestRegressor(n_estimators=64,max_depth=5,min_samples_leaf=5,random_state=2026,n_jobs=1)
        tick=time.perf_counter();model.fit(X,y);fit_seconds=time.perf_counter()-tick;models[size]=model
        learning_log.append(dict(heldout_satellites=size,training_instances=train_names,
                                 evaluation_instances=test_names,training_rows=len(y),
                                 fit_seconds=fit_seconds,all_label_generation_seconds=label_seconds))
    (out/'training_protocol.json').write_text(json.dumps(learning_log,indent=2))
    rows=[];schedule_log=out/'schedules.jsonl'
    with schedule_log.open('w') as handle:
        for stress in STRESSES:
            for p in paths:
                instance=m.load_instance(p,stress)
                for policy,single in [('margin',True)]+[(v,False) for v in m.SELECTORS]+[('learned',False)]:
                    for seed in (range(10) if policy=='random' else [0]):
                        row,plans=m.evaluate(instance,policy,single,seed,model=models[instance['satellites']])
                        rows.append(row)
                        handle.write(json.dumps(dict(instance=instance['name'],stress=stress,
                                                     policy=row['policy'],seed=seed,plans=[asdict(x) for x in plans]))+'\n')
                print(stress,instance['name'],'complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'instance_results.csv',index=False)
    grouped=frame.groupby(['stress','instance','policy'])[METRICS].mean().reset_index()
    summary=grouped.groupby(['stress','policy'])[METRICS].mean().reset_index()
    summary.to_csv(out/'comparison_summary.csv',index=False)
    paired_evidence(frame).to_csv(out/'paired_statistics.csv',index=False)
    sensitivity=[]
    print('Running one-at-a-time and joint sensitivity...',flush=True)
    for p in paths:
        instance=m.load_instance(p,'combined')
        for term in list(m.DEFAULT_WEIGHTS)+['joint']:
            for factor in [.5,.75,1.,1.25,1.5]:
                w=dict(m.DEFAULT_WEIGHTS)
                for key in w:
                    if term=='joint' or key==term: w[key]*=factor
                row,_=m.evaluate(instance,weights=w)
                row.update(term=term,factor=factor);sensitivity.append(row)
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(out/'weight_sensitivity.csv',index=False)
    # Positive common scaling must leave the ordering of scores unchanged.
    joint=sensitivity.query('term=="joint"')
    assert all(g.completed.nunique()==1 for _,g in joint.groupby('instance'))
    figures(frame,sensitivity,out/'figures')
    import scipy,sklearn
    manifest=dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),
                  numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                  weights=m.DEFAULT_WEIGHTS,random_seeds=list(range(10)),
                  validation='Every saved schedule passed independent constraint checks.',
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(m.__file__)]})
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.query('stress=="combined"').to_string(index=False),flush=True)


if __name__=='__main__': main()
