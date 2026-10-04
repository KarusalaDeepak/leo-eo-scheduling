"""Turn held-out calibration evidence into manuscript assets and a plain report."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    out=ROOT/'results/weight_calibration'
    frame=pd.read_csv(out/'heldout_results.csv')
    folds=json.loads((out/'fold_weights.json').read_text())
    stats=pd.read_csv(out/'paired_group_statistics.csv')
    summary=pd.read_csv(out/'summary.csv')
    ablation=pd.read_csv(out/'term_ablations.csv')
    keys=list(folds[0]['weights'])
    lines=[]
    # Combine identical displayed vectors, not the independently held-out folds.
    displayed={}
    for fold in folds:
        vector=tuple(f"{fold['weights'][k]:.3f}" for k in keys)
        displayed.setdefault(vector, []).append(str(fold['heldout_satellites']))
    for vector,sizes in displayed.items():
        label=', '.join(sizes)
        if len(sizes)>1:
            label=r'\mbox{'+label+'}'  # Keep grouped fleet sizes on one line.
        lines.append(label+' & '+' & '.join(vector)+r' \\')
    (ROOT/'revision/calibration_rows.tex').write_text('\\newcommand{\\CalibrationRows}{%\n'+'\n'.join(lines)+'%\n}\n')
    additions=[]
    for policy,label in [('equal','RACS equal'),('calibrated','RACS calibrated')]:
        base=summary[(summary.stress=='base') & (summary.policy==policy)].iloc[0]
        joint=summary[(summary.stress=='combined') & (summary.policy==policy)].iloc[0]
        additions.append(f'{label} & {base.completion:.3f} & {joint.completion:.3f} & {joint.reward:.3f} & {joint.congestion:.3f} & {joint.fairness:.3f} & {joint.runtime_ms:.1f} '+r'\\')
    (ROOT/'revision/calibrated_comparison_rows.tex').write_text('\\newcommand{\\CalibratedComparisonRows}{%\n'+'\n'.join(additions)+'%\n}\n')
    # Group bootstrap reflects the split unit, not 45 independent deployments.
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    order=['default','equal','calibrated','earliest','margin']
    for ax,metric,title in zip(axes,['reward','completion'],['Completed priority','Completed fraction']):
        groups=frame.groupby(['satellites','policy'])[metric].mean().unstack()
        means=[];lower=[];upper=[]
        for name in order:
            values=groups[name].to_numpy();mean=values.mean()
            boot=np.random.default_rng(20261003).choice(values,(10000,len(values)),replace=True).mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975]);means.append(mean);lower.append(mean-lo);upper.append(hi-mean)
        ax.bar(range(len(order)),means,color=['#8b8b8b','#dc9d39','#197c80','#4e6ca0','#806da5'])
        ax.errorbar(range(len(order)),means,yerr=[lower,upper],fmt='none',ecolor='black',capsize=3)
        ax.set_xticks(range(len(order)),['Default','Equal','Calibrated','Earliest','Margin'],rotation=25,ha='right')
        ax.set_ylabel(title);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'heldout_calibration.pdf');fig.savefig(out/'heldout_calibration.png',dpi=220);plt.close(fig)
    baseline=frame.groupby('policy')[['completion','reward']].mean()
    removed=ablation.groupby('removed_term')[['completion','reward']].mean()
    for metric in ['completion','reward']:
        removed['delta_'+metric]=removed[metric]-baseline.loc['calibrated',metric]
    removed.to_csv(out/'ablation_summary.csv')
    # Check deterministic metrics against the prior version. Timing is not invariant.
    historical=pd.read_csv(ROOT/'results/reviewer_revision/instance_results.csv')
    checks=[]
    for new,old in [('default','full_racs'),('earliest','full_earliest'),('margin','full_margin')]:
        a=frame[frame.policy==new].set_index(['instance','stress'])
        b=historical[(historical.policy==old)&(historical.seed==0)].set_index(['instance','stress'])
        for metric in ['completion','reward','congestion','fairness']:
            assert np.allclose(a[metric].sort_index(),b[metric].sort_index(),equal_nan=True)
        checks.append(new+': unchanged non-timing metrics')
    report='''WEIGHT CALIBRATION: POST-REVIEW EVIDENCE

Protocol: 33 candidates (normalized inherited weights, equal weights, 31
Dirichlet(1) draws with seed 20261003). Four leave-one-fleet-size-out groups.
All normal/congested/stress variants of the held-out size are excluded from
selection. Optimize mean completed-priority fraction with equal development
group weighting. Exact ties prefer the earlier candidate, with default first.
No held-out performance feedback changes the search budget or selection rule.

This is a reanalysis of a previously inspected benchmark, not an untouched
external test. Folds share development data and are not independent replications.
Group-bootstrap intervals and signed-rank tests are exploratory, with only four
groups. Holm covers eight comparisons (two metrics, four comparators).
Internal margin mixture/scales are fixed, not calibrated by this experiment.

1980 schedules evaluated: 1485 search, 225 held-out/comparator, 270 drop-one.
All undergo independent constraint validation. Search scores for each group are
cached for computation, but selection explicitly excludes the outer test group.

SELECTED WEIGHTS\n'''+json.dumps(folds,indent=2)
    report+='\n\nCOMBINED STRESS (instance means)\n'+summary[summary.stress=='combined'].to_string(index=False)
    report+='\n\nPAIRED GROUP EVIDENCE (all stress conditions averaged within fleet group)\n'+stats.to_string(index=False)
    report+='\n\nDROP-ONE DIAGNOSTICS (not additional tuning)\n'+removed.to_string()
    report+='\n\nREGRESSION CHECKS\n'+'\n'.join(checks)+'\n'
    (out/'calibration_report.txt').write_text(report)
    print(report)


if __name__=='__main__':main()
