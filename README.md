# RACS: Reliability-Aware Congestion-Stressed EO Scheduling

## Current four-page reviewer revision (2026-10-01)

Weight-calibration follow-up (2026-10-03): `results/weight_calibration/` contains
the pre-run search protocol, held-out-size weights, search scores, comparator
results, term ablations, exploratory group statistics, and a plain-text report.
The 1,980 additional schedules all passed validation. Calibrated and equal
weights match on completion/reward; no adjusted comparison establishes
superiority. Original default coefficients are retained, not silently replaced.
The evaluation runner accepts any number of valid JSON files with at least two
fleet sizes for held-out learning. There is no 40-satellite scheduler cap.
Tested fleet sizes remain disclosed; unseen orbital data is not invented.

### Weight selection and relationship to the second paper

The six score coefficients are empirical hyperparameters, not physical constants.
The normalized search removes redundant common positive scaling; it does not
normalize feature magnitudes. Internal margin scales and the 0.55/0.45 mixture
remain fixed assumptions. Each fold selects on other fleet groups' mean priority
reward, with equal group weighting and deterministic candidate-ID tie-breaking.
The selector rejects missing development conditions and nonfinite development
rewards. Held-out rewards never participate in selection. Fold-specific vectors
are not a universally validated deployment vector. Sensitivity and ablations
test robustness and necessity; neither is assumed from the experiment's existence.

The related manuscript is *Delay-Aware Disagreement Diagnostics and Evidence-Gated
5G-LEO Multi-Access Path Selection* (code:
https://github.com/KarusalaDeepak/open-leo-latency-routing). Its separation of
calibration and evaluation motivates the evidence discipline here, not an import
of its algorithm. RACS does not use its RTT/CVaR gate, reactive fallback, or
independent-group non-inferiority bounds. These EO folds cannot inherit those
guarantees. No second-paper dataset or numerical result is reused as EO evidence.

Manuscripts, reviewer correspondence, and generated paper assets are maintained locally and are not distributed in the current repository tree. Historical experiment scripts remain for provenance; use the matched evaluation commands below for the current results.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 scripts/run_matched_evaluation.py --dataset-dir /path/to/dataset
python3 scripts/check_restricted_plan_optimum.py --dataset-dir /path/to/dataset
python3 scripts/run_weight_calibration.py --dataset-dir /path/to/dataset
python3 scripts/build_revision_assets.py
python3 scripts/build_weight_calibration_report.py
```

The revised engine is `scripts/matched_scheduler.py`. It enforces supplied
observation bounds, exact observation duration, satellite and ground-station
exclusivity, ISL reservation at both endpoints, payload causality, and storage
capacity. Six bounded construction rules form a shared plan pool. RACS now
selects the highest-scoring plan. Full earliest, margin, congestion, random, and
learned selectors share these capabilities. The learned ranker holds out all
instances and stress variants of each satellite count. This is batch scheduling.

Outputs in `results/reviewer_revision/` include 720 validated schedules across
45 conditions, 315 sensitivity runs, exact signed-rank tests with Holm correction,
and 18 restricted-plan MILP checks. Random seeds are averaged within instance.

Combined-stress completion: RACS 0.60148, single margin 0.52037, and matched
earliest/margin 0.60556. RACS is **not** best on completion. No completion test
passes Holm correction. Lower payload/reserved-capacity occupancy is a
headroom/utilization trade-off, not measured congestion reduction.

Orbital-generation provenance is incomplete. Short or zero-length observation
intervals are excluded; the previous parser's implicit extension is not used.
Scalability is measured only through 40 satellites. These synthetic contacts do not establish operational or orbital-model validity.


## Dataset and local outputs

Obtain the EO scheduling JSON instances from the original IEEE DataPort provider; raw data are not redistributed. Pass their directory with --dataset-dir. Generated results/ and revision/ assets stay local.

## Interpretation

The search includes normalized default weights, equal weights, and 31 draws from Dirichlet(1,1,1,1,1,1), using seed 20261003. Each six-component vector sums to one. The seed provides reproducibility, not evidence of optimality.

Downlink-capacity occupancy is payload volume divided by reserved satellite-to-ground capacity, including unused reserved capacity. It is not storage occupancy or measured network congestion. Runs with no completed requests have undefined occupancy and are excluded from its average.

Calibration p-values are paired Wilcoxon results across four satellite-count groups. Raw p=0.5 is unadjusted; Holm-adjusted p=1 accounts for eight comparisons. Neither establishes a difference at the 0.05 threshold, and neither proves equivalence.

## Internal margin-scale sensitivity

Run `python3 scripts/run_margin_scale_sensitivity.py --dataset-dir /path/to/dataset`.
This exploratory 3-by-3 grid varies the time scale (900, 1800, 3600 seconds)
and capacity scale (1, 2, 4 GB), keeping the six default weights and the
0.55/0.45 mixture unchanged. Half/double perturbations probe local sensitivity;
they do not justify the original scales or select a new default. Every schedule
undergoes feasibility validation. CSV outputs and the pre-run protocol are saved
under `results/margin_scale_sensitivity/`. Stress variants are not independent
deployments. The original 1800-second/2-GB setting remains the default.
