#!/usr/bin/env python3
"""Additional reviewer-facing analyses for EO scheduling experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_eo_scheduling_experiments as exp  # noqa: E402


def load_instances(data_dir: Path) -> list[dict]:
    return [exp.load_instance(path) for path in sorted(data_dir.glob("*.json"))]


def plans_overlap(left: exp.TaskPlan, right: exp.TaskPlan) -> bool:
    left_rows = defaultdict(list)
    right_rows = defaultdict(list)
    for sat_id, interval in left.resource_reservations:
        left_rows[sat_id].append(interval)
    for sat_id, interval in right.resource_reservations:
        right_rows[sat_id].append(interval)
    for sat_id in set(left_rows) & set(right_rows):
        for a in left_rows[sat_id]:
            for b in right_rows[sat_id]:
                if exp.intervals_overlap(a, b):
                    return True
    return False


def build_candidate_plans(instance: dict) -> list[exp.TaskPlan]:
    plans: list[exp.TaskPlan] = []
    empty_reservations: dict[str, list[exp.Interval]] = {}
    for task in instance["tasks"]:
        for sat_id in instance["satellites"]:
            plan = exp.build_candidate_plan(
                instance,
                task,
                sat_id,
                "proposed_reliability_congestion_aware",
                empty_reservations,
            )
            if plan is not None:
                plans.append(plan)
        split_plan = exp.build_multi_satellite_plan(instance, task, empty_reservations)
        if split_plan is not None:
            plans.append(split_plan)
    return plans


def solve_candidate_plan_milp(instance: dict) -> dict:
    plans = build_candidate_plans(instance)
    if not plans:
        return {
            "instance": instance["name"],
            "policy": "candidate_plan_milp",
            "meta_tasks": len(instance["tasks"]),
            "completed_tasks": 0,
            "completion_rate": 0.0,
            "normalized_priority_reward": 0.0,
            "deadline_miss_rate": 1.0,
            "mean_congestion_ratio": 0.0,
        }

    objective = -np.array([plan.priority + 0.001 * plan.reliability_score for plan in plans], dtype=float)
    constraints = []
    lower = []
    upper = []

    by_task: dict[str, list[int]] = defaultdict(list)
    for idx, plan in enumerate(plans):
        by_task[plan.task_id].append(idx)
    for indices in by_task.values():
        row = np.zeros(len(plans))
        row[indices] = 1.0
        constraints.append(row)
        lower.append(0.0)
        upper.append(1.0)

    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            if plans[i].task_id == plans[j].task_id:
                continue
            if plans_overlap(plans[i], plans[j]):
                row = np.zeros(len(plans))
                row[i] = 1.0
                row[j] = 1.0
                constraints.append(row)
                lower.append(0.0)
                upper.append(1.0)

    linear_constraint = LinearConstraint(np.vstack(constraints), np.array(lower), np.array(upper))
    result = milp(
        c=objective,
        integrality=np.ones(len(plans)),
        bounds=Bounds(np.zeros(len(plans)), np.ones(len(plans))),
        constraints=linear_constraint,
        options={"time_limit": 30},
    )
    selected = []
    if result.success and result.x is not None:
        selected = [plan for plan, value in zip(plans, result.x) if value >= 0.5]

    total_priority = sum(int(task["priority"]) for task in instance["tasks"])
    completed_priority = sum(plan.priority for plan in selected)
    return {
        "instance": instance["name"],
        "policy": "candidate_plan_milp",
        "meta_tasks": len(instance["tasks"]),
        "candidate_plans": len(plans),
        "completed_tasks": len(selected),
        "completion_rate": len(selected) / max(1, len(instance["tasks"])),
        "normalized_priority_reward": completed_priority / max(1, total_priority),
        "deadline_miss_rate": 1.0 - len(selected) / max(1, len(instance["tasks"])),
        "mean_congestion_ratio": float(np.mean([plan.congestion_ratio for plan in selected])) if selected else 0.0,
        "mean_reliability_score": float(np.mean([plan.reliability_score for plan in selected])) if selected else 0.0,
    }


def runtime_scaling(instances: list[dict]) -> pd.DataFrame:
    rows = []
    for instance in instances:
        for policy in ["reliability_only", "proposed_multi_satellite_relay_aware"]:
            durations = []
            for _ in range(10):
                start = time.perf_counter()
                exp.evaluate_policy(instance, policy)
                durations.append((time.perf_counter() - start) * 1000.0)
            rows.append(
                {
                    "instance": instance["name"],
                    "satellites": len(instance["satellites"]),
                    "meta_tasks": len(instance["tasks"]),
                    "policy": policy,
                    "runtime_ms": float(np.mean(durations)),
                }
            )
    return pd.DataFrame(rows)


def plot_runtime_scaling(frame: pd.DataFrame, output_path: Path) -> None:
    labels = {
        "reliability_only": "Reliability-only",
        "proposed_multi_satellite_relay_aware": "RACS",
    }
    normal = frame[~frame["instance"].str.contains("Congested|Tiny", regex=True)].copy()
    normal = normal.sort_values(["satellites", "policy"])
    plt.figure(figsize=(5.8, 3.6))
    for policy, rows in normal.groupby("policy"):
        plt.plot(
            rows["satellites"],
            rows["runtime_ms"],
            marker="o",
            linewidth=2,
            label=labels.get(policy, policy),
        )
        for _, row in rows.iterrows():
            plt.annotate(
                f"{int(row['meta_tasks'])} tasks",
                (row["satellites"], row["runtime_ms"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    plt.xlabel("Satellites in normal instances")
    plt.ylabel("Runtime per instance (ms)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def score_plan(plan: exp.TaskPlan, task: dict, instance: dict, weights: dict[str, float]) -> float:
    start = instance["start"]
    arrival_s = exp.seconds_from_start(task["arrival_time"], start)
    deadline_s = exp.seconds_from_start(task["deadline"], start)
    urgency = 1.0 / max(1.0, (deadline_s - arrival_s) / 3600.0)
    delay_h = max(0.0, (plan.completion_time - arrival_s) / 3600.0)
    split_gain = min(1.0, len(set(plan.sat_id.split("+"))) / max(1, len(task["atomic_tasks"])))
    return (
        weights["priority"] * (int(task["priority"]) / 10.0)
        + weights["reliability"] * plan.reliability_score
        + weights["urgency"] * urgency
        + weights["split"] * split_gain
        - weights["congestion"] * plan.congestion_ratio
        - weights["uncertainty"] * plan.uncertainty_score
        - weights["relay"] * plan.relay_count
        - weights["delay"] * delay_h
    )


def evaluate_score_guided(instance: dict, weights: dict[str, float]) -> dict:
    reservations: dict[str, list[exp.Interval]] = {}
    remaining = list(instance["tasks"])
    completed: list[exp.TaskPlan] = []

    while remaining:
        candidates = []
        for task in remaining:
            plan = exp.build_multi_satellite_plan(instance, task, reservations)
            if plan is not None:
                candidates.append((score_plan(plan, task, instance, weights), task, plan))
        if not candidates:
            break
        _, task, plan = max(candidates, key=lambda item: item[0])
        exp.reserve_plan(plan, reservations)
        completed.append(plan)
        remaining = [item for item in remaining if item["id"] != task["id"]]

    total_tasks = len(instance["tasks"])
    total_priority = sum(int(task["priority"]) for task in instance["tasks"])
    completed_priority = sum(plan.priority for plan in completed)
    return {
        "completed_tasks": len(completed),
        "completion_rate": len(completed) / max(1, total_tasks),
        "normalized_priority_reward": completed_priority / max(1, total_priority),
        "deadline_miss_rate": 1.0 - len(completed) / max(1, total_tasks),
        "mean_congestion_ratio": float(np.mean([plan.congestion_ratio for plan in completed])) if completed else 0.0,
    }


def score_weight_sensitivity(instances: list[dict]) -> pd.DataFrame:
    base_weights = {
        "priority": 4.0,
        "reliability": 2.8,
        "urgency": 1.2,
        "split": 0.4,
        "congestion": 2.2,
        "uncertainty": 1.6,
        "relay": 0.15,
        "delay": 0.15,
    }
    variants = [("default", "all", 1.0, base_weights)]
    for key in ["congestion", "uncertainty", "relay"]:
        for factor in [0.5, 1.5]:
            weights = dict(base_weights)
            weights[key] = base_weights[key] * factor
            variants.append((f"{key}_{factor:.1f}x", key, factor, weights))

    stressed = [
        exp.prepare_instance(exp.transform_instance_data(instance["raw"], "combined_stress"))
        for instance in instances
    ]
    rows = []
    for variant_name, varied_weight, factor, weights in variants:
        summaries = [evaluate_score_guided(instance, weights) for instance in stressed]
        rows.append(
            {
                "variant": variant_name,
                "varied_weight": varied_weight,
                "factor": factor,
                "completion_rate": float(np.mean([item["completion_rate"] for item in summaries])),
                "normalized_priority_reward": float(
                    np.mean([item["normalized_priority_reward"] for item in summaries])
                ),
                "deadline_miss_rate": float(np.mean([item["deadline_miss_rate"] for item in summaries])),
                "mean_congestion_ratio": float(np.mean([item["mean_congestion_ratio"] for item in summaries])),
            }
        )
    return pd.DataFrame(rows)


def plot_score_sensitivity(frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = frame.copy()
    order = ["default", "congestion_0.5x", "congestion_1.5x", "uncertainty_0.5x", "uncertainty_1.5x", "relay_0.5x", "relay_1.5x"]
    labels = ["Default", "Cong. 0.5x", "Cong. 1.5x", "Uncert. 0.5x", "Uncert. 1.5x", "Relay 0.5x", "Relay 1.5x"]
    plot_frame = plot_frame.set_index("variant").loc[order]
    x = np.arange(len(plot_frame))
    width = 0.36
    plt.figure(figsize=(6.6, 3.6))
    plt.bar(x - width / 2, plot_frame["completion_rate"], width, label="Completion")
    plt.bar(x + width / 2, plot_frame["deadline_miss_rate"], width, label="Miss")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Rate under combined stress")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/raw/dataset")
    parser.add_argument("--output-dir", default="results/eo_scheduling")
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir)
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    instances = load_instances(data_dir)
    small_instances = [item for item in instances if len(item["satellites"]) <= 5]
    milp_rows = [solve_candidate_plan_milp(instance) for instance in small_instances]
    pd.DataFrame(milp_rows).to_csv(output_dir / "candidate_plan_milp_summary.csv", index=False)
    runtime_frame = runtime_scaling(instances)
    runtime_frame.to_csv(output_dir / "runtime_scaling_summary.csv", index=False)
    sensitivity_frame = score_weight_sensitivity(instances)
    sensitivity_frame.to_csv(output_dir / "score_weight_sensitivity_summary.csv", index=False)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_runtime_scaling(runtime_frame, figure_dir / "runtime_scaling_by_satellites.png")
    plot_score_sensitivity(sensitivity_frame, figure_dir / "score_weight_sensitivity.png")
    print(f"milp_csv={output_dir / 'candidate_plan_milp_summary.csv'}")
    print(f"runtime_csv={output_dir / 'runtime_scaling_summary.csv'}")
    print(f"sensitivity_csv={output_dir / 'score_weight_sensitivity_summary.csv'}")
    print(f"runtime_plot={output_dir / 'figures' / 'runtime_scaling_by_satellites.png'}")
    print(f"sensitivity_plot={output_dir / 'figures' / 'score_weight_sensitivity.png'}")


if __name__ == "__main__":
    main()
