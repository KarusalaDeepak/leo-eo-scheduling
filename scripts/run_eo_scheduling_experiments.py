#!/usr/bin/env python3
"""Run reliability-aware EO task-scheduling experiments on LEO JSON instances.

The scheduler treats each meta-task as complete only when all its atomic
observations are assigned to one satellite and the generated data is downlinked
before the task deadline. This keeps the decision model explicit and reproducible
for paper-facing comparisons.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


POLICIES = [
    "earliest_deadline_first",
    "highest_priority_first",
    "shortest_task_first",
    "congestion_unaware",
    "reliability_only",
    "proposed_reliability_congestion_aware",
    "proposed_multi_satellite_relay_aware",
]


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    sat_id: str
    observation_intervals: tuple[Interval, ...]
    downlink_interval: Interval
    ground_station_id: str
    completion_time: int
    total_duration_s: float
    total_volume_gb: float
    priority: int
    slack_s: float
    downlink_capacity_gb: float
    congestion_ratio: float
    reliability_score: float
    uncertainty_score: float
    score: float
    visible_satellite_count: int = 0
    feasible_satellite_count: int = 0
    assignment_mode: str = "single_satellite"
    relay_count: int = 0
    downlink_window_count: int = 1
    resource_reservations: tuple[tuple[str, Interval], ...] = ()


def parse_time(value: str) -> pd.Timestamp:
    return pd.Timestamp(value.replace("Z", "+00:00"))


def seconds_from_start(value: str, start: pd.Timestamp) -> int:
    return int((parse_time(value) - start).total_seconds())


def intervals_overlap(left: Interval, right: Interval) -> bool:
    return left.start < right.end and right.start < left.end


def is_available(interval: Interval, reservations: list[Interval]) -> bool:
    return all(not intervals_overlap(interval, reserved) for reserved in reservations)


def capacity_gb(window: dict[str, Any]) -> float:
    duration_s = max(0, int((parse_time(window["end"]) - parse_time(window["start"])).total_seconds()))
    return float(window["bandwidth_gbps"]) * duration_s / 8.0


def load_instance(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return prepare_instance(data)


def prepare_instance(data: dict[str, Any]) -> dict[str, Any]:
    start = parse_time(data["config"]["start_utc"])

    atomic_to_task: dict[str, dict[str, Any]] = {}
    for task in data["tasks"]:
        for atomic in task["atomic_tasks"]:
            atomic_to_task[atomic["id"]] = task

    obs_by_sat_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for window in data["windows"]["obs"]:
        atomic = atomic_to_task.get(window["atomic_task_id"])
        if atomic is None:
            continue
        atomic_detail = next(item for item in atomic["atomic_tasks"] if item["id"] == window["atomic_task_id"])
        start_s = seconds_from_start(window["start"], start)
        parsed_end_s = seconds_from_start(window["end"], start)
        required_end_s = start_s + int(atomic_detail["duration"])
        end_s = max(parsed_end_s, required_end_s)
        obs_by_sat_task.setdefault((window["sat_id"], window["atomic_task_id"]), []).append(
            {
                "sat_id": window["sat_id"],
                "atomic_task_id": window["atomic_task_id"],
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": int(atomic_detail["duration"]),
            }
        )

    for rows in obs_by_sat_task.values():
        rows.sort(key=lambda item: (item["start_s"], item["end_s"]))

    s2g_by_sat: dict[str, list[dict[str, Any]]] = {}
    for window in data["windows"]["s2g"]:
        enriched = dict(window)
        enriched["start_s"] = seconds_from_start(window["start"], start)
        enriched["end_s"] = seconds_from_start(window["end"], start)
        enriched["capacity_gb"] = capacity_gb(window)
        s2g_by_sat.setdefault(window["sat_id"], []).append(enriched)

    for rows in s2g_by_sat.values():
        rows.sort(key=lambda item: (item["start_s"], item["end_s"]))

    isl_by_sat: dict[str, list[dict[str, Any]]] = {}
    for window in data["windows"]["isl"]:
        enriched = dict(window)
        enriched["start_s"] = seconds_from_start(window["start"], start)
        enriched["end_s"] = seconds_from_start(window["end"], start)
        enriched["capacity_gb"] = float(window.get("capacity_gbps", 0.0)) * max(
            0, enriched["end_s"] - enriched["start_s"]
        ) / 8.0
        isl_by_sat.setdefault(window["s1"], []).append(enriched)
        isl_by_sat.setdefault(window["s2"], []).append(enriched)

    return {
        "name": data["instance_name"],
        "raw": data,
        "start": start,
        "satellites": [item["id"] for item in data["satellites"]],
        "tasks": data["tasks"],
        "obs_by_sat_task": obs_by_sat_task,
        "s2g_by_sat": s2g_by_sat,
        "isl_by_sat": isl_by_sat,
    }


def task_total_duration(task: dict[str, Any]) -> float:
    return float(sum(item["duration"] for item in task["atomic_tasks"]))


def task_total_volume(task: dict[str, Any]) -> float:
    return float(sum(item["vol_gb"] for item in task["atomic_tasks"]))


def order_tasks(tasks: list[dict[str, Any]], policy: str, start: pd.Timestamp) -> list[dict[str, Any]]:
    def arrival(task: dict[str, Any]) -> int:
        return seconds_from_start(task["arrival_time"], start)

    def deadline(task: dict[str, Any]) -> int:
        return seconds_from_start(task["deadline"], start)

    if policy == "highest_priority_first":
        key = lambda task: (-int(task["priority"]), deadline(task), arrival(task))
    elif policy == "shortest_task_first":
        key = lambda task: (task_total_duration(task) + 60.0 * task_total_volume(task), deadline(task))
    elif policy == "proposed_reliability_congestion_aware":
        key = lambda task: (deadline(task) - arrival(task), -int(task["priority"]), arrival(task))
    elif policy == "proposed_multi_satellite_relay_aware":
        key = lambda task: (deadline(task) - arrival(task), -int(task["priority"]), task_total_volume(task))
    else:
        key = lambda task: (deadline(task), -int(task["priority"]), arrival(task))
    return sorted(tasks, key=key)


def sigmoid(value: float) -> float:
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def build_candidate_plan(
    instance: dict[str, Any],
    task: dict[str, Any],
    sat_id: str,
    policy: str,
    reservations: dict[str, list[Interval]],
) -> TaskPlan | None:
    start = instance["start"]
    arrival_s = seconds_from_start(task["arrival_time"], start)
    deadline_s = seconds_from_start(task["deadline"], start)
    local_reservations = list(reservations.get(sat_id, []))
    observation_intervals: list[Interval] = []

    ready_s = arrival_s
    for atomic in sorted(task["atomic_tasks"], key=lambda item: item["duration"], reverse=True):
        options = instance["obs_by_sat_task"].get((sat_id, atomic["id"]), [])
        chosen_interval: Interval | None = None
        for option in options:
            interval = Interval(max(option["start_s"], arrival_s), option["end_s"])
            if interval.start < arrival_s or interval.end > deadline_s:
                continue
            if interval.start < ready_s and policy in {"congestion_unaware", "earliest_deadline_first"}:
                continue
            if is_available(interval, local_reservations):
                chosen_interval = interval
                break
        if chosen_interval is None:
            return None
        local_reservations.append(chosen_interval)
        observation_intervals.append(chosen_interval)
        ready_s = max(ready_s, chosen_interval.end)

    total_volume_gb = task_total_volume(task)
    downlink_options = instance["s2g_by_sat"].get(sat_id, [])
    chosen_downlink: dict[str, Any] | None = None
    for option in downlink_options:
        interval = Interval(option["start_s"], option["end_s"])
        if interval.start < ready_s or interval.end > deadline_s:
            continue
        if option["capacity_gb"] + 1e-9 < total_volume_gb:
            continue
        if not is_available(interval, local_reservations):
            continue
        chosen_downlink = option
        break

    if chosen_downlink is None:
        return None

    downlink_interval = Interval(chosen_downlink["start_s"], chosen_downlink["end_s"])
    completion_s = downlink_interval.end
    slack_s = max(0.0, float(deadline_s - completion_s))
    capacity_margin_gb = float(chosen_downlink["capacity_gb"] - total_volume_gb)
    congestion_ratio = total_volume_gb / max(float(chosen_downlink["capacity_gb"]), 1e-9)
    reliability_score = 0.55 * sigmoid(slack_s / 1800.0) + 0.45 * sigmoid(capacity_margin_gb / 2.0)
    uncertainty_score = min(1.0, congestion_ratio) * (1.0 - reliability_score)
    priority = int(task["priority"])
    delay_s = max(0.0, float(completion_s - arrival_s))

    if policy == "reliability_only":
        score = 1000.0 * reliability_score + 0.001 * slack_s - 0.01 * delay_s
    elif policy == "congestion_unaware":
        score = -delay_s
    elif policy == "proposed_reliability_congestion_aware":
        normalized_priority = priority / 10.0
        urgency = 1.0 / max(1.0, (deadline_s - arrival_s) / 3600.0)
        score = (
            4.0 * normalized_priority
            + 2.5 * reliability_score
            + 1.2 * urgency
            - 2.0 * congestion_ratio
            - 1.5 * uncertainty_score
            - 0.15 * delay_s / 3600.0
        )
    elif policy == "highest_priority_first":
        score = -delay_s + 60.0 * priority
    elif policy == "shortest_task_first":
        score = -delay_s - task_total_duration(task)
    else:
        score = -completion_s

    return TaskPlan(
        task_id=task["id"],
        sat_id=sat_id,
        observation_intervals=tuple(observation_intervals),
        downlink_interval=downlink_interval,
        ground_station_id=str(chosen_downlink["gs_id"]),
        completion_time=completion_s,
        total_duration_s=task_total_duration(task),
        total_volume_gb=total_volume_gb,
        priority=priority,
        slack_s=slack_s,
        downlink_capacity_gb=float(chosen_downlink["capacity_gb"]),
        congestion_ratio=congestion_ratio,
        reliability_score=reliability_score,
        uncertainty_score=uncertainty_score,
        score=score,
        resource_reservations=tuple((sat_id, interval) for interval in observation_intervals + [downlink_interval]),
    )


def other_isl_endpoint(window: dict[str, Any], sat_id: str) -> str | None:
    if window["s1"] == sat_id:
        return str(window["s2"])
    if window["s2"] == sat_id:
        return str(window["s1"])
    return None


def reserve_interval_copy(reservations: dict[str, list[Interval]]) -> dict[str, list[Interval]]:
    return {key: list(value) for key, value in reservations.items()}


def choose_downlink_segments(
    instance: dict[str, Any],
    sat_id: str,
    volume_gb: float,
    ready_s: int,
    deadline_s: int,
    local_reservations: dict[str, list[Interval]],
) -> tuple[list[tuple[str, str, Interval, float]], float, int] | None:
    """Choose one or more downlink windows, optionally through one ISL relay."""

    def direct_segments(
        downlink_sat_id: str,
        earliest_s: int,
        required_volume_gb: float,
    ) -> tuple[list[tuple[str, str, Interval, float]], float] | None:
        remaining = required_volume_gb
        chosen: list[tuple[str, str, Interval, float]] = []
        total_capacity = 0.0
        for option in instance["s2g_by_sat"].get(downlink_sat_id, []):
            interval = Interval(option["start_s"], option["end_s"])
            if interval.start < earliest_s or interval.end > deadline_s:
                continue
            if not is_available(interval, local_reservations.get(downlink_sat_id, [])):
                continue
            capacity = float(option["capacity_gb"])
            if capacity <= 0:
                continue
            chosen.append((downlink_sat_id, str(option["gs_id"]), interval, capacity))
            total_capacity += capacity
            remaining -= capacity
            if remaining <= 1e-9:
                return chosen, total_capacity
        return None

    direct = direct_segments(sat_id, ready_s, volume_gb)
    if direct is not None:
        return direct[0], direct[1], 0

    for isl in instance["isl_by_sat"].get(sat_id, []):
        relay_sat = other_isl_endpoint(isl, sat_id)
        if relay_sat is None:
            continue
        transfer_start = max(ready_s, int(isl["start_s"]))
        if transfer_start >= deadline_s:
            continue
        transfer_rate = float(isl.get("capacity_gbps", 0.0))
        if transfer_rate <= 0:
            continue
        transfer_duration_s = math.ceil(volume_gb * 8.0 / transfer_rate)
        transfer_end = transfer_start + transfer_duration_s
        if transfer_end > int(isl["end_s"]) or transfer_end > deadline_s:
            continue
        relayed = direct_segments(relay_sat, transfer_end, volume_gb)
        if relayed is not None:
            return relayed[0], relayed[1], 1
    return None


def build_multi_satellite_plan(
    instance: dict[str, Any],
    task: dict[str, Any],
    reservations: dict[str, list[Interval]],
) -> TaskPlan | None:
    """Build a split atomic-task schedule using all visible satellites."""

    start = instance["start"]
    arrival_s = seconds_from_start(task["arrival_time"], start)
    deadline_s = seconds_from_start(task["deadline"], start)
    local_reservations = reserve_interval_copy(reservations)
    observation_intervals: list[Interval] = []
    assigned_sats: list[str] = []
    volume_by_sat: dict[str, float] = {}
    feasible_satellite_choices = 0

    for atomic in sorted(task["atomic_tasks"], key=lambda item: item["duration"], reverse=True):
        atomic_candidates: list[tuple[float, str, Interval]] = []
        for sat_id in instance["satellites"]:
            for option in instance["obs_by_sat_task"].get((sat_id, atomic["id"]), []):
                interval = Interval(max(option["start_s"], arrival_s), option["end_s"])
                if interval.start < arrival_s or interval.end > deadline_s:
                    continue
                if not is_available(interval, local_reservations.get(sat_id, [])):
                    continue
                atomic_candidates.append((interval.end + 0.01 * len(local_reservations.get(sat_id, [])), sat_id, interval))
        if not atomic_candidates:
            return None
        feasible_satellite_choices += len({item[1] for item in atomic_candidates})
        _, chosen_sat, chosen_interval = min(atomic_candidates, key=lambda item: item[0])
        local_reservations.setdefault(chosen_sat, []).append(chosen_interval)
        local_reservations[chosen_sat].sort(key=lambda interval: (interval.start, interval.end))
        observation_intervals.append(chosen_interval)
        assigned_sats.append(chosen_sat)
        volume_by_sat[chosen_sat] = volume_by_sat.get(chosen_sat, 0.0) + float(atomic["vol_gb"])

    ready_s = max(interval.end for interval in observation_intervals)
    downlink_segments: list[tuple[str, str, Interval, float]] = []
    resource_reservations: list[tuple[str, Interval]] = [
        (sat_id, interval) for sat_id, interval in zip(assigned_sats, observation_intervals)
    ]
    total_downlink_capacity_gb = 0.0
    relay_count = 0
    for sat_id, volume_gb in sorted(volume_by_sat.items()):
        chosen = choose_downlink_segments(instance, sat_id, volume_gb, ready_s, deadline_s, local_reservations)
        if chosen is None:
            return None
        segments, segment_capacity, segment_relay_count = chosen
        relay_count += segment_relay_count
        total_downlink_capacity_gb += segment_capacity
        for downlink_sat, _, interval, _ in segments:
            local_reservations.setdefault(downlink_sat, []).append(interval)
            local_reservations[downlink_sat].sort(key=lambda item: (item.start, item.end))
            resource_reservations.append((downlink_sat, interval))
        downlink_segments.extend(segments)

    if not downlink_segments:
        return None

    completion_s = max(segment[2].end for segment in downlink_segments)
    total_volume_gb = task_total_volume(task)
    slack_s = max(0.0, float(deadline_s - completion_s))
    capacity_margin_gb = total_downlink_capacity_gb - total_volume_gb
    congestion_ratio = total_volume_gb / max(total_downlink_capacity_gb, 1e-9)
    reliability_score = 0.55 * sigmoid(slack_s / 1800.0) + 0.45 * sigmoid(capacity_margin_gb / 2.0)
    uncertainty_score = min(1.0, congestion_ratio) * (1.0 - reliability_score)
    priority = int(task["priority"])
    delay_s = max(0.0, float(completion_s - arrival_s))
    normalized_priority = priority / 10.0
    urgency = 1.0 / max(1.0, (deadline_s - arrival_s) / 3600.0)
    split_gain = min(1.0, len(set(assigned_sats)) / max(1, len(task["atomic_tasks"])))
    relay_penalty = 0.15 * relay_count
    score = (
        4.0 * normalized_priority
        + 2.8 * reliability_score
        + 1.2 * urgency
        + 0.4 * split_gain
        - 2.2 * congestion_ratio
        - 1.6 * uncertainty_score
        - relay_penalty
        - 0.15 * delay_s / 3600.0
    )
    return TaskPlan(
        task_id=task["id"],
        sat_id="+".join(sorted(set(assigned_sats))),
        observation_intervals=tuple(observation_intervals),
        downlink_interval=Interval(min(segment[2].start for segment in downlink_segments), completion_s),
        ground_station_id="+".join(sorted(set(segment[1] for segment in downlink_segments))),
        completion_time=completion_s,
        total_duration_s=task_total_duration(task),
        total_volume_gb=total_volume_gb,
        priority=priority,
        slack_s=slack_s,
        downlink_capacity_gb=total_downlink_capacity_gb,
        congestion_ratio=congestion_ratio,
        reliability_score=reliability_score,
        uncertainty_score=uncertainty_score,
        score=score,
        visible_satellite_count=count_visible_satellites(instance, task),
        feasible_satellite_count=feasible_satellite_choices,
        assignment_mode="multi_satellite_relay",
        relay_count=relay_count,
        downlink_window_count=len(downlink_segments),
        resource_reservations=tuple(resource_reservations),
    )


def choose_plan(
    instance: dict[str, Any],
    task: dict[str, Any],
    policy: str,
    reservations: dict[str, list[Interval]],
) -> tuple[TaskPlan | None, int]:
    visible_count = count_visible_satellites(instance, task)
    candidates = [
        plan
        for sat_id in instance["satellites"]
        if (plan := build_candidate_plan(instance, task, sat_id, policy, reservations)) is not None
    ]
    if not candidates:
        return None, visible_count
    feasible_count = len(candidates)
    chosen = max(candidates, key=lambda plan: (plan.score, plan.reliability_score, -plan.completion_time))
    return (
        TaskPlan(
            **{
                **chosen.__dict__,
                "visible_satellite_count": visible_count,
                "feasible_satellite_count": feasible_count,
            }
        ),
        visible_count,
    )


def count_visible_satellites(instance: dict[str, Any], task: dict[str, Any]) -> int:
    """Count satellites that have observation windows for every atomic task."""

    visible_count = 0
    for sat_id in instance["satellites"]:
        if all(instance["obs_by_sat_task"].get((sat_id, atomic["id"])) for atomic in task["atomic_tasks"]):
            visible_count += 1
    return visible_count


def reserve_plan(plan: TaskPlan, reservations: dict[str, list[Interval]]) -> None:
    if plan.resource_reservations:
        for sat_id, interval in plan.resource_reservations:
            rows = reservations.setdefault(sat_id, [])
            rows.append(interval)
            rows.sort(key=lambda item: (item.start, item.end))
        return
    rows = reservations.setdefault(plan.sat_id, [])
    rows.extend(plan.observation_intervals)
    rows.append(plan.downlink_interval)
    rows.sort(key=lambda interval: (interval.start, interval.end))


def priority_class(priority: int) -> str:
    if priority >= 7:
        return "high"
    if priority >= 4:
        return "medium"
    return "low"


def jain_index(values: list[float]) -> float:
    active = [value for value in values if not math.isnan(value)]
    if not active:
        return 0.0
    denominator = len(active) * sum(value * value for value in active)
    if denominator == 0:
        return 0.0
    return (sum(active) ** 2) / denominator


def evaluate_policy(instance: dict[str, Any], policy: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reservations: dict[str, list[Interval]] = {}
    completed: list[TaskPlan] = []
    rejected: list[dict[str, Any]] = []
    task_diagnostics: list[dict[str, Any]] = []

    ordered_tasks = order_tasks(instance["tasks"], policy, instance["start"])
    for task in ordered_tasks:
        if policy == "proposed_multi_satellite_relay_aware":
            visible_count = count_visible_satellites(instance, task)
            plan = build_multi_satellite_plan(instance, task, reservations)
        else:
            plan, visible_count = choose_plan(instance, task, policy, reservations)
        if plan is None:
            rejected.append(task)
            task_diagnostics.append(
                {
                    "instance": instance["name"],
                    "policy": policy,
                    "task_id": task["id"],
                    "status": "rejected",
                    "selected_sat_id": "",
                    "visible_satellite_count": visible_count,
                    "feasible_satellite_count": 0,
                    "assignment_mode": "",
                    "relay_count": 0,
                    "downlink_window_count": 0,
                    "priority": int(task["priority"]),
                    "atomic_task_count": len(task["atomic_tasks"]),
                    "total_duration_s": task_total_duration(task),
                    "total_volume_gb": task_total_volume(task),
                }
            )
            continue
        reserve_plan(plan, reservations)
        completed.append(plan)
        task_diagnostics.append(
            {
                "instance": instance["name"],
                "policy": policy,
                "task_id": task["id"],
                "status": "completed",
                "selected_sat_id": plan.sat_id,
                "visible_satellite_count": plan.visible_satellite_count,
                "feasible_satellite_count": plan.feasible_satellite_count,
                "assignment_mode": plan.assignment_mode,
                "relay_count": plan.relay_count,
                "downlink_window_count": plan.downlink_window_count,
                "priority": plan.priority,
                "atomic_task_count": len(task["atomic_tasks"]),
                "total_duration_s": plan.total_duration_s,
                "total_volume_gb": plan.total_volume_gb,
            }
        )

    total_tasks = len(instance["tasks"])
    total_priority = sum(int(task["priority"]) for task in instance["tasks"])
    completed_priority = sum(plan.priority for plan in completed)
    classes = ["low", "medium", "high"]
    class_totals = {name: 0 for name in classes}
    class_completed = {name: 0 for name in classes}
    for task in instance["tasks"]:
        class_totals[priority_class(int(task["priority"]))] += 1
    for plan in completed:
        class_completed[priority_class(plan.priority)] += 1
    class_rates = [
        class_completed[name] / class_totals[name] if class_totals[name] else float("nan")
        for name in classes
    ]

    detail_rows = [
        {
            "instance": instance["name"],
            "policy": policy,
            "task_id": plan.task_id,
            "sat_id": plan.sat_id,
            "ground_station_id": plan.ground_station_id,
            "priority": plan.priority,
            "completion_time_s": plan.completion_time,
            "slack_min": plan.slack_s / 60.0,
            "total_volume_gb": plan.total_volume_gb,
            "congestion_ratio": plan.congestion_ratio,
            "reliability_score": plan.reliability_score,
            "uncertainty_score": plan.uncertainty_score,
            "visible_satellite_count": plan.visible_satellite_count,
            "feasible_satellite_count": plan.feasible_satellite_count,
            "assignment_mode": plan.assignment_mode,
            "relay_count": plan.relay_count,
            "downlink_window_count": plan.downlink_window_count,
        }
        for plan in completed
    ]

    summary = {
        "instance": instance["name"],
        "policy": policy,
        "satellites": len(instance["satellites"]),
        "meta_tasks": total_tasks,
        "completed_tasks": len(completed),
        "rejected_tasks": len(rejected),
        "completion_rate": len(completed) / total_tasks if total_tasks else 0.0,
        "priority_reward": completed_priority,
        "normalized_priority_reward": completed_priority / total_priority if total_priority else 0.0,
        "deadline_miss_rate": len(rejected) / total_tasks if total_tasks else 0.0,
        "mean_completion_delay_min": mean(
            [
                (plan.completion_time - seconds_from_start(next(t for t in instance["tasks"] if t["id"] == plan.task_id)["arrival_time"], instance["start"])) / 60.0
                for plan in completed
            ]
        )
        if completed
        else float("nan"),
        "mean_slack_min": mean([plan.slack_s / 60.0 for plan in completed]) if completed else float("nan"),
        "mean_congestion_ratio": mean([plan.congestion_ratio for plan in completed]) if completed else float("nan"),
        "mean_reliability_score": mean([plan.reliability_score for plan in completed]) if completed else float("nan"),
        "mean_uncertainty_score": mean([plan.uncertainty_score for plan in completed]) if completed else float("nan"),
        "mean_visible_satellites": mean([row["visible_satellite_count"] for row in task_diagnostics])
        if task_diagnostics
        else float("nan"),
        "mean_feasible_satellites": mean([row["feasible_satellite_count"] for row in task_diagnostics])
        if task_diagnostics
        else float("nan"),
        "mean_relay_count": mean([plan.relay_count for plan in completed]) if completed else float("nan"),
        "mean_downlink_window_count": mean([plan.downlink_window_count for plan in completed]) if completed else float("nan"),
        "jain_priority_class_completion": jain_index(class_rates),
        "low_priority_completion": class_rates[0],
        "medium_priority_completion": class_rates[1],
        "high_priority_completion": class_rates[2],
    }
    return summary, detail_rows, task_diagnostics


def scenario_group(instance_name: str) -> str:
    if "Congested" in instance_name or "Cmax" in instance_name:
        return "congested"
    return "normal"


def format_timestamp(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat().replace("+00:00", "Z")


def transform_instance_data(data: dict[str, Any], stress_scenario: str) -> dict[str, Any]:
    """Create deterministic controlled-stress variants for robustness testing."""

    transformed = copy.deepcopy(data)
    if stress_scenario == "tight_deadline":
        deadline_factor = 0.70
        volume_factor = 1.0
        bandwidth_factor = 1.0
    elif stress_scenario == "reduced_bandwidth":
        deadline_factor = 1.0
        volume_factor = 1.0
        bandwidth_factor = 0.65
    elif stress_scenario == "high_volume":
        deadline_factor = 1.0
        volume_factor = 1.35
        bandwidth_factor = 1.0
    elif stress_scenario == "combined_stress":
        deadline_factor = 0.65
        volume_factor = 1.35
        bandwidth_factor = 0.60
    else:
        raise ValueError(f"unsupported stress scenario: {stress_scenario}")

    transformed["instance_name"] = f"{data['instance_name']}__{stress_scenario}"
    for task in transformed["tasks"]:
        arrival = parse_time(task["arrival_time"])
        deadline = parse_time(task["deadline"])
        duration = max(60.0, (deadline - arrival).total_seconds() * deadline_factor)
        task["deadline"] = format_timestamp(arrival + pd.Timedelta(seconds=duration))
        for atomic in task["atomic_tasks"]:
            atomic["vol_gb"] = round(float(atomic["vol_gb"]) * volume_factor, 4)

    for window in transformed["windows"]["s2g"]:
        window["bandwidth_gbps"] = round(float(window["bandwidth_gbps"]) * bandwidth_factor, 4)

    return transformed


def evaluate_instances(instances: list[dict[str, Any]], stress_scenario: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for instance in instances:
        for policy in POLICIES:
            summary, rows, task_rows = evaluate_policy(instance, policy)
            summary["scenario_group"] = scenario_group(instance["name"])
            summary["stress_scenario"] = stress_scenario
            summaries.append(summary)
            for row in rows:
                row["stress_scenario"] = stress_scenario
            for row in task_rows:
                row["stress_scenario"] = stress_scenario
            details.extend(rows)
            diagnostics.extend(task_rows)
    return pd.DataFrame(summaries), pd.DataFrame(details), pd.DataFrame(diagnostics)


def run_experiments(dataset_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict[str, Any]] = []
    base_instances: list[dict[str, Any]] = []

    for path in sorted(dataset_dir.glob("*.json")):
        instance = load_instance(path)
        base_instances.append(instance)
        raw = instance["raw"]
        dataset_rows.append(
            {
                "instance": instance["name"],
                "scenario_group": scenario_group(instance["name"]),
                "satellites": len(raw["satellites"]),
                "ground_stations": len(raw["ground_stations"]),
                "meta_tasks": len(raw["tasks"]),
                "atomic_tasks": sum(len(task["atomic_tasks"]) for task in raw["tasks"]),
                "s2g_windows": len(raw["windows"]["s2g"]),
                "obs_windows": len(raw["windows"]["obs"]),
                "isl_windows": len(raw["windows"]["isl"]),
            }
        )
    dataset_summary = pd.DataFrame(dataset_rows)
    policy_summary, decision_details, task_diagnostics_frame = evaluate_instances(base_instances, "base")

    stress_summaries: list[pd.DataFrame] = []
    stress_details: list[pd.DataFrame] = []
    stress_diagnostics: list[pd.DataFrame] = []
    for stress_scenario in ["tight_deadline", "reduced_bandwidth", "high_volume", "combined_stress"]:
        stress_instances = [
            prepare_instance(transform_instance_data(instance["raw"], stress_scenario)) for instance in base_instances
        ]
        stress_summary, stress_detail, stress_diag = evaluate_instances(stress_instances, stress_scenario)
        stress_summaries.append(stress_summary)
        stress_details.append(stress_detail)
        stress_diagnostics.append(stress_diag)
    stress_policy_summary = pd.concat(stress_summaries, ignore_index=True)
    stress_task_diagnostics = pd.concat(stress_diagnostics, ignore_index=True)

    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    policy_summary.to_csv(output_dir / "policy_summary.csv", index=False)
    decision_details.to_csv(output_dir / "scheduled_task_details.csv", index=False)
    task_diagnostics_frame.to_csv(output_dir / "task_assignment_diagnostics.csv", index=False)
    stress_policy_summary.to_csv(output_dir / "stress_policy_summary.csv", index=False)
    stress_task_diagnostics.to_csv(output_dir / "stress_task_assignment_diagnostics.csv", index=False)
    comparison_summary = build_comparison_summary(policy_summary)
    comparison_summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    stress_comparison_summary = build_stress_comparison_summary(stress_policy_summary)
    stress_comparison_summary.to_csv(output_dir / "stress_comparison_summary.csv", index=False)
    build_policy_rankings(policy_summary).to_csv(output_dir / "policy_rankings.csv", index=False)
    build_stress_rankings(stress_policy_summary).to_csv(output_dir / "stress_policy_rankings.csv", index=False)
    write_markdown_report(output_dir / "experiment_report.md", dataset_summary, policy_summary)
    plot_results(policy_summary, figure_dir)
    plot_stress_results(stress_policy_summary, figure_dir)
    collect_paper_outputs(output_dir)


def build_comparison_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    proposed_policy = "proposed_multi_satellite_relay_aware"
    grouped = (
        policy_summary.groupby(["scenario_group", "policy"], as_index=False)
        .agg(
            completion_rate=("completion_rate", "mean"),
            normalized_priority_reward=("normalized_priority_reward", "mean"),
            deadline_miss_rate=("deadline_miss_rate", "mean"),
            mean_congestion_ratio=("mean_congestion_ratio", "mean"),
            mean_reliability_score=("mean_reliability_score", "mean"),
        )
    )
    proposed = grouped[grouped["policy"] == proposed_policy].set_index("scenario_group")
    rows: list[dict[str, Any]] = []
    for _, baseline in grouped[grouped["policy"] != proposed_policy].iterrows():
        group = baseline["scenario_group"]
        if group not in proposed.index:
            continue
        prop = proposed.loc[group]
        rows.append(
            {
                "scenario_group": group,
                "baseline_policy": baseline["policy"],
                "completion_rate_delta": prop["completion_rate"] - baseline["completion_rate"],
                "priority_reward_delta": prop["normalized_priority_reward"] - baseline["normalized_priority_reward"],
                "deadline_miss_delta": prop["deadline_miss_rate"] - baseline["deadline_miss_rate"],
                "congestion_ratio_delta": prop["mean_congestion_ratio"] - baseline["mean_congestion_ratio"],
                "reliability_score_delta": prop["mean_reliability_score"] - baseline["mean_reliability_score"],
            }
        )
    return pd.DataFrame(rows)


def build_policy_rankings(policy_summary: pd.DataFrame) -> pd.DataFrame:
    rank_frame = policy_summary.copy()
    rank_frame["rank_score"] = (
        3.0 * rank_frame["normalized_priority_reward"]
        + 2.0 * rank_frame["completion_rate"]
        - 1.5 * rank_frame["deadline_miss_rate"]
        - 0.8 * rank_frame["mean_congestion_ratio"]
        + 0.5 * rank_frame["mean_reliability_score"]
    )
    rank_frame["rank"] = rank_frame.groupby("instance")["rank_score"].rank(method="min", ascending=False).astype(int)
    instance_rankings = rank_frame[
        [
            "instance",
            "scenario_group",
            "policy",
            "rank",
            "rank_score",
            "completion_rate",
            "normalized_priority_reward",
            "deadline_miss_rate",
            "mean_congestion_ratio",
            "mean_reliability_score",
            "mean_visible_satellites",
            "mean_feasible_satellites",
        ]
    ].assign(ranking_scope="instance")

    grouped = (
        rank_frame.groupby(["scenario_group", "policy"], as_index=False)
        .agg(
            rank_score=("rank_score", "mean"),
            completion_rate=("completion_rate", "mean"),
            normalized_priority_reward=("normalized_priority_reward", "mean"),
            deadline_miss_rate=("deadline_miss_rate", "mean"),
            mean_congestion_ratio=("mean_congestion_ratio", "mean"),
            mean_reliability_score=("mean_reliability_score", "mean"),
            mean_visible_satellites=("mean_visible_satellites", "mean"),
            mean_feasible_satellites=("mean_feasible_satellites", "mean"),
        )
    )
    grouped["rank"] = grouped.groupby("scenario_group")["rank_score"].rank(method="min", ascending=False).astype(int)
    grouped["instance"] = ""
    grouped = grouped[
        [
            "instance",
            "scenario_group",
            "policy",
            "rank",
            "rank_score",
            "completion_rate",
            "normalized_priority_reward",
            "deadline_miss_rate",
            "mean_congestion_ratio",
            "mean_reliability_score",
            "mean_visible_satellites",
            "mean_feasible_satellites",
        ]
    ].assign(ranking_scope="scenario_group")
    return pd.concat([instance_rankings, grouped], ignore_index=True)


def build_stress_rankings(stress_policy_summary: pd.DataFrame) -> pd.DataFrame:
    rank_frame = stress_policy_summary.copy()
    rank_frame["rank_score"] = (
        3.0 * rank_frame["normalized_priority_reward"]
        + 2.0 * rank_frame["completion_rate"]
        - 1.5 * rank_frame["deadline_miss_rate"]
        - 0.8 * rank_frame["mean_congestion_ratio"]
        + 0.5 * rank_frame["mean_reliability_score"]
    )
    grouped = (
        rank_frame.groupby(["stress_scenario", "policy"], as_index=False)
        .agg(
            rank_score=("rank_score", "mean"),
            completion_rate=("completion_rate", "mean"),
            normalized_priority_reward=("normalized_priority_reward", "mean"),
            deadline_miss_rate=("deadline_miss_rate", "mean"),
            mean_congestion_ratio=("mean_congestion_ratio", "mean"),
            mean_reliability_score=("mean_reliability_score", "mean"),
        )
        .sort_values(["stress_scenario", "rank_score"], ascending=[True, False])
    )
    grouped["rank"] = grouped.groupby("stress_scenario")["rank_score"].rank(method="min", ascending=False).astype(int)
    return grouped


def build_stress_comparison_summary(stress_policy_summary: pd.DataFrame) -> pd.DataFrame:
    proposed_policy = "proposed_multi_satellite_relay_aware"
    grouped = (
        stress_policy_summary.groupby(["stress_scenario", "policy"], as_index=False)
        .agg(
            completion_rate=("completion_rate", "mean"),
            normalized_priority_reward=("normalized_priority_reward", "mean"),
            deadline_miss_rate=("deadline_miss_rate", "mean"),
            mean_congestion_ratio=("mean_congestion_ratio", "mean"),
            mean_reliability_score=("mean_reliability_score", "mean"),
        )
    )
    proposed = grouped[grouped["policy"] == proposed_policy].set_index("stress_scenario")
    rows: list[dict[str, Any]] = []
    for _, baseline in grouped[grouped["policy"] != proposed_policy].iterrows():
        scenario = baseline["stress_scenario"]
        if scenario not in proposed.index:
            continue
        prop = proposed.loc[scenario]
        rows.append(
            {
                "stress_scenario": scenario,
                "baseline_policy": baseline["policy"],
                "completion_rate_delta": prop["completion_rate"] - baseline["completion_rate"],
                "priority_reward_delta": prop["normalized_priority_reward"] - baseline["normalized_priority_reward"],
                "deadline_miss_delta": prop["deadline_miss_rate"] - baseline["deadline_miss_rate"],
                "congestion_ratio_delta": prop["mean_congestion_ratio"] - baseline["mean_congestion_ratio"],
                "reliability_score_delta": prop["mean_reliability_score"] - baseline["mean_reliability_score"],
            }
        )
    return pd.DataFrame(rows)


def short_policy_name(policy: str) -> str:
    return {
        "earliest_deadline_first": "EDF",
        "highest_priority_first": "Priority",
        "shortest_task_first": "Shortest",
        "congestion_unaware": "No congestion",
        "reliability_only": "Reliability",
        "proposed_reliability_congestion_aware": "Proposed",
        "proposed_multi_satellite_relay_aware": "Proposed+split",
    }[policy]


def write_markdown_report(path: Path, dataset_summary: pd.DataFrame, policy_summary: pd.DataFrame) -> None:
    def to_markdown(frame: pd.DataFrame) -> str:
        text_frame = frame.copy().fillna("").astype(str)
        headers = list(text_frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in text_frame.values.tolist():
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    best = (
        policy_summary.sort_values(
            ["instance", "normalized_priority_reward", "completion_rate", "mean_reliability_score"],
            ascending=[True, False, False, False],
        )
        .groupby("instance", as_index=False)
        .head(1)
    )
    grouped = (
        policy_summary.groupby(["scenario_group", "policy"], as_index=False)
        .agg(
            completion_rate=("completion_rate", "mean"),
            normalized_priority_reward=("normalized_priority_reward", "mean"),
            deadline_miss_rate=("deadline_miss_rate", "mean"),
            mean_congestion_ratio=("mean_congestion_ratio", "mean"),
            mean_reliability_score=("mean_reliability_score", "mean"),
            fairness=("jain_priority_class_completion", "mean"),
        )
        .sort_values(["scenario_group", "normalized_priority_reward"], ascending=[True, False])
    )
    comparison = build_comparison_summary(policy_summary)
    lines = [
        "# EO Scheduling Experiment Report",
        "",
        "## Dataset Summary",
        "",
        to_markdown(dataset_summary),
        "",
        "## Best Policy Per Instance",
        "",
        to_markdown(
            best[
                [
                    "instance",
                    "policy",
                    "completion_rate",
                    "normalized_priority_reward",
                    "deadline_miss_rate",
                    "mean_congestion_ratio",
                    "mean_reliability_score",
                ]
            ]
        ),
        "",
        "## Average Results By Scenario Group",
        "",
        to_markdown(grouped),
        "",
        "## Proposed Method Delta Against Baselines",
        "",
        "Positive completion/reward/reliability deltas are better. Negative deadline-miss and congestion-ratio deltas are better.",
        "",
        to_markdown(comparison),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(policy_summary: pd.DataFrame, figure_dir: Path) -> None:
    if plt is None:
        (figure_dir / "PLOTS_SKIPPED.txt").write_text(
            "matplotlib is not installed in this Python environment. CSV and Markdown results were generated.\n",
            encoding="utf-8",
        )
        return
    grouped = (
        policy_summary.groupby(["scenario_group", "policy"], as_index=False)
        .agg(
            completion_rate=("completion_rate", "mean"),
            normalized_priority_reward=("normalized_priority_reward", "mean"),
            deadline_miss_rate=("deadline_miss_rate", "mean"),
            mean_congestion_ratio=("mean_congestion_ratio", "mean"),
            mean_reliability_score=("mean_reliability_score", "mean"),
            jain_priority_class_completion=("jain_priority_class_completion", "mean"),
        )
    )
    grouped["policy_label"] = grouped["policy"].map(short_policy_name)

    for metric, ylabel, filename in [
        ("completion_rate", "Completion rate", "completion_rate_by_policy.png"),
        ("normalized_priority_reward", "Normalized priority reward", "priority_reward_by_policy.png"),
        ("deadline_miss_rate", "Deadline miss rate", "deadline_miss_by_policy.png"),
        ("mean_congestion_ratio", "Mean congestion ratio", "congestion_by_policy.png"),
        ("mean_reliability_score", "Mean reliability score", "reliability_by_policy.png"),
        ("jain_priority_class_completion", "Priority-class fairness", "fairness_by_policy.png"),
    ]:
        pivot = grouped.pivot(index="policy_label", columns="scenario_group", values=metric)
        pivot = pivot.reindex([short_policy_name(policy) for policy in POLICIES])
        ax = pivot.plot(kind="bar", figsize=(8.8, 3.8), rot=25)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="Scenario")
        plt.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        plt.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for group_name, group in grouped.groupby("scenario_group"):
        ax.scatter(
            group["mean_congestion_ratio"],
            group["normalized_priority_reward"],
            s=80,
            label=group_name,
        )
    ax.set_xlabel("Mean congestion ratio lower is better")
    ax.set_ylabel("Normalized priority reward higher is better")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="Scenario")
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(figure_dir / "reward_congestion_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_stress_results(stress_policy_summary: pd.DataFrame, figure_dir: Path) -> None:
    if plt is None:
        return
    selected_policies = [
        "earliest_deadline_first",
        "highest_priority_first",
        "reliability_only",
        "proposed_reliability_congestion_aware",
        "proposed_multi_satellite_relay_aware",
    ]
    frame = stress_policy_summary[stress_policy_summary["policy"].isin(selected_policies)].copy()
    frame["policy_label"] = frame["policy"].map(short_policy_name)
    grouped = (
        frame.groupby(["stress_scenario", "policy", "policy_label"], as_index=False)
        .agg(
            completion_rate=("completion_rate", "mean"),
            normalized_priority_reward=("normalized_priority_reward", "mean"),
            deadline_miss_rate=("deadline_miss_rate", "mean"),
            mean_congestion_ratio=("mean_congestion_ratio", "mean"),
        )
    )
    stress_order = ["tight_deadline", "reduced_bandwidth", "high_volume", "combined_stress"]
    stress_labels = {
        "tight_deadline": "Tight deadline",
        "reduced_bandwidth": "Reduced bandwidth",
        "high_volume": "High volume",
        "combined_stress": "Combined",
    }
    for metric, ylabel, filename in [
        ("completion_rate", "Completion rate", "stress_completion_rate.png"),
        ("normalized_priority_reward", "Normalized priority reward", "stress_priority_reward.png"),
        ("deadline_miss_rate", "Deadline miss rate", "stress_deadline_miss.png"),
        ("mean_congestion_ratio", "Mean congestion ratio", "stress_congestion_ratio.png"),
    ]:
        pivot = grouped.pivot(index="stress_scenario", columns="policy_label", values=metric)
        pivot = pivot.reindex(stress_order)
        pivot.index = [stress_labels[item] for item in pivot.index]
        ax = pivot.plot(kind="bar", figsize=(9.2, 4.0), rot=20)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="Policy")
        plt.tight_layout(rect=(0.0, 0.0, 0.77, 1.0))
        plt.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()


def collect_paper_outputs(output_dir: Path) -> None:
    paper_dir = output_dir / "paper_outputs"
    paper_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "completion_rate_by_policy.png",
        "priority_reward_by_policy.png",
        "deadline_miss_by_policy.png",
        "congestion_by_policy.png",
        "reliability_by_policy.png",
        "fairness_by_policy.png",
        "reward_congestion_tradeoff.png",
        "stress_completion_rate.png",
        "stress_priority_reward.png",
        "stress_deadline_miss.png",
        "stress_congestion_ratio.png",
    ]:
        source = output_dir / "figures" / filename
        if source.exists():
            (paper_dir / filename).write_bytes(source.read_bytes())
    for filename in [
        "dataset_summary.csv",
        "policy_summary.csv",
        "comparison_summary.csv",
        "policy_rankings.csv",
        "task_assignment_diagnostics.csv",
        "stress_policy_summary.csv",
        "stress_comparison_summary.csv",
        "stress_policy_rankings.csv",
        "stress_task_assignment_diagnostics.csv",
        "experiment_report.md",
    ]:
        source = output_dir / filename
        if source.exists():
            (paper_dir / filename).write_bytes(source.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/raw/dataset")
    parser.add_argument("--output-dir", default="results/eo_scheduling")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    run_experiments(dataset_dir, output_dir)
    print(f"results_written={output_dir}")
    print(f"summary_csv={output_dir / 'policy_summary.csv'}")
    print(f"report={output_dir / 'experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
