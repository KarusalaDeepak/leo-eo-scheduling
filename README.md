# RACS: Reliability-Aware Congestion-Stressed EO Scheduling

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
