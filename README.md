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

The current paper is `revision/main.tex`. Compile from `revision/` so the generated
table inputs and vector figures resolve. Reviewer correspondence is retained
locally and is not distributed. The submitted manuscript and original
experiment scripts are retained for provenance; their results are **not** the
corrected revision results described here.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 scripts/run_matched_evaluation.py --dataset-dir /path/to/dataset
python3 scripts/check_restricted_plan_optimum.py --dataset-dir /path/to/dataset
python3 scripts/run_weight_calibration.py --dataset-dir /path/to/dataset
python3 scripts/build_revision_assets.py
python3 scripts/build_weight_calibration_report.py
cd revision
mkdir -p output/pdf
tectonic --keep-logs --outdir output/pdf main.tex
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
Scalability is measured only through 40 satellites. See the manuscript for limits.

## Submitted version (historical documentation below)

The following descriptions and numerical claims document the six-page submitted
version. Use the commands and results above for the corrected paper.

This repository contains the code and paper-facing outputs for:

**RACS: Reliability-Aware Congestion-Stressed Scheduling for Earth Observation in LEO Satellite Networks**

The code evaluates reliability-aware Earth-observation (EO) task scheduling in low-Earth-orbit (LEO) satellite networks. The main scheduler, **RACS**, treats a meta-task as completed only when all atomic observations are captured and the generated data are delivered to a ground station before the deadline. The implementation explicitly models:

- observation-window feasibility,
- satellite-to-ground (S2G) downlink capacity,
- inter-satellite-link (ISL) relay fallback,
- deadline urgency,
- meta-task priority,
- downlink congestion,
- reliability margin,
- uncertainty,
- multi-satellite atomic-task splitting, and
- split downlink across multiple delivery windows.

## Dataset

The experiments use the following IEEE DataPort dataset:

**Congestion-stressed LEO satellite network dataset for Earth observation task scheduling and routing**, IEEE DataPort, 2026.

Dataset URL:

<https://ieee-dataport.org/documents/congestion-stressed-leo-satellite-network-dataset-earth-observation-task-scheduling-and>

The dataset is not redistributed in this repository. Download it from IEEE DataPort and place the JSON instance files under:

```text
data/raw/dataset/
```

Expected instance files include normal and congested LEO EO scheduling cases, for example:

```text
S05_Small.json
S05_Small_Congested.json
S10_Medium.json
S10_Medium_Congested.json
S20_Large.json
S20_Large_Congested.json
S40_Extra.json
S40_Extra_Congested.json
S05_Tiny_CmaxTest.json
```

You may also keep the dataset elsewhere and pass its path using `--dataset-dir`.

## Repository Structure

```text
.
├── figures/                         # Paper figures copied from experiment outputs
├── main.tex                         # Manuscript source
├── requirements.txt                 # Python dependencies
├── scripts/
│   ├── run_eo_scheduling_experiments.py
│   └── run_additional_analysis.py
└── results/                         # Generated locally; ignored by Git
```

## Installation

Create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Main Scheduling Experiments

If the dataset is placed at `data/raw/dataset/`, run:

```bash
python scripts/run_eo_scheduling_experiments.py
```

If the dataset is stored elsewhere:

```bash
python scripts/run_eo_scheduling_experiments.py \
  --dataset-dir /path/to/dataset \
  --output-dir results/eo_scheduling
```

This generates:

```text
results/eo_scheduling/dataset_summary.csv
results/eo_scheduling/policy_summary.csv
results/eo_scheduling/comparison_summary.csv
results/eo_scheduling/policy_rankings.csv
results/eo_scheduling/task_assignment_diagnostics.csv
results/eo_scheduling/stress_policy_summary.csv
results/eo_scheduling/stress_comparison_summary.csv
results/eo_scheduling/stress_policy_rankings.csv
results/eo_scheduling/stress_task_assignment_diagnostics.csv
results/eo_scheduling/experiment_report.md
results/eo_scheduling/figures/*.png
```

## Run Additional Reviewer-Facing Analyses

The additional analysis script produces the restricted candidate-plan MILP sanity check, runtime scaling summary, and score-weight sensitivity results:

```bash
python scripts/run_additional_analysis.py \
  --dataset-dir data/raw/dataset \
  --output-dir results/eo_scheduling
```

Main outputs:

```text
results/eo_scheduling/candidate_plan_milp_summary.csv
results/eo_scheduling/runtime_scaling_summary.csv
results/eo_scheduling/score_weight_sensitivity_summary.csv
results/eo_scheduling/figures/runtime_scaling_by_satellites.png
results/eo_scheduling/figures/score_weight_sensitivity.png
```

## Policies Evaluated

The main experiment compares:

- `earliest_deadline_first`
- `highest_priority_first`
- `shortest_task_first`
- `congestion_unaware`
- `reliability_only`
- `proposed_reliability_congestion_aware`
- `proposed_multi_satellite_relay_aware` / RACS

RACS differs from single-satellite baselines because it can assign atomic observations of the same EO meta-task to different visible satellites, split generated data across multiple downlink windows, and use one-hop relay fallback when direct S2G delivery is infeasible.

## Paper-Relevant Results

The current reproduced results support the following claims:

- In normal instances, RACS improves average completion rate from `0.900` to `0.958` compared with reliability-only scheduling.
- In congested instances, RACS improves average completion rate from `0.713` to `0.743` and reduces mean congestion ratio from `0.323` to `0.272`.
- Under combined deadline, bandwidth, and data-volume stress, RACS improves completion rate from `0.626` to `0.806` and reduces deadline miss rate from `0.374` to `0.194`.
- The restricted candidate-plan MILP is reported only as a sanity check on small cases, not as an optimal benchmark or optimality-gap certificate.
- Score-weight sensitivity shows that perturbing congestion, uncertainty, and relay penalties keeps completion stable under combined stress.

## Citation Text for the Dataset

Use this form in the paper or artifact description:

```bibtex
@misc{dataport_eo_2026,
  author       = {{IEEE DataPort}},
  title        = {{Congestion-stressed LEO satellite network dataset for Earth observation task scheduling and routing}},
  year         = {2026},
  howpublished = {\url{https://ieee-dataport.org/documents/congestion-stressed-leo-satellite-network-dataset-earth-observation-task-scheduling-and}}
}
```

## Notes

- The dataset is benchmark/synthetic rather than a complete operator trace; results should be interpreted as benchmarked congestion-stressed LEO EO scheduling results.
- The one-hop relay model is a bounded online design choice, not a claim that deeper relay routing is unimportant.
- Atomic-task splitting is appropriate for tiled imaging, separable sensing points, or sub-area EO workloads.
