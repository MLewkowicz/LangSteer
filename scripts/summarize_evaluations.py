"""Aggregate evaluation JSONs (one per condition) into a wide CSV.

One row per task. Columns: task, then per-condition (rate %, n_success/n,
avg steps to success). Matches the fields printed by
run_evaluation.print_summary.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_RUNS = [
    ("P0_baseline_steered", "outputs/evaluation/baseline_steered_P0/baseline_steered.json"),
    ("P1_baseline_steered", "outputs/evaluation/baseline_steered_P1/baseline_steered.json"),
    ("P0_action_object",    "outputs/evaluation/P0_action_object/langsteer_primitive_object.json"),
    ("P1_action_object",    "outputs/evaluation/P1_action_object/langsteer_primitive_object.json"),
    ("P2_action_object",    "outputs/evaluation/P2_action_object/langsteer_primitive_object.json"),
    ("P3_action_object",    "outputs/evaluation/P3_action_object/langsteer_primitive_object.json"),
    ("P4_action_object",    "outputs/evaluation/P4_2026-05-23_22-55-54/_langsteer_primitive_object_vlm.json"),
    ("P0_baseline",         "outputs/evaluation/P0_baseline/baseline.json"),
    ("P1_baseline",         "outputs/evaluation/P1_baseline/baseline.json"),
    ("P2_baseline",         "outputs/evaluation/P2_baseline/baseline.json"),
    ("P3_baseline",         "outputs/evaluation/P3_baseline/baseline.json"),
    ("P4_baseline",         "outputs/evaluation/P4_baseline/baseline.json"),
]


def summarize(path: Path):
    """Return dict[task_name -> {n_success, n_episodes, rate_pct, avg_steps}]."""
    data = json.loads(path.read_text())
    out = {}
    total_success = 0
    total_eps = 0
    for task_name, task_data in data["tasks"].items():
        eps = task_data["episodes"]
        n = len(eps)
        n_success = sum(1 for e in eps if e["success"])
        succ_steps = [e["steps"] for e in eps if e["success"]]
        avg_steps = float(np.mean(succ_steps)) if succ_steps else float("nan")
        out[task_name] = {
            "n_success": n_success,
            "n_episodes": n,
            "rate_pct": 100.0 * n_success / n if n else 0.0,
            "avg_steps": avg_steps,
        }
        total_success += n_success
        total_eps += n
    out["OVERALL"] = {
        "n_success": total_success,
        "n_episodes": total_eps,
        "rate_pct": 100.0 * total_success / total_eps if total_eps else 0.0,
        "avg_steps": float("nan"),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Repo root (paths in DEFAULT_RUNS are resolved against this).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <repo-root>/outputs/evaluation/summary.csv).",
    )
    args = parser.parse_args()

    root = Path(args.repo_root)
    out_path = Path(args.output) if args.output else root / "outputs/evaluation/summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-condition summaries + the canonical task ordering (first run that has it).
    per_condition = {}
    task_order = []
    for condition, rel in DEFAULT_RUNS:
        path = root / rel
        if not path.exists():
            print(f"[skip] missing: {path}")
            continue
        summary = summarize(path)
        per_condition[condition] = summary
        for t in summary:
            if t != "OVERALL" and t not in task_order:
                task_order.append(t)
        print(
            f"[ok] {condition:<20} {summary['OVERALL']['n_success']:>3}/"
            f"{summary['OVERALL']['n_episodes']:<4} {summary['OVERALL']['rate_pct']:>6.1f}%"
        )

    conditions = list(per_condition.keys())
    header = ["task"]
    for c in conditions:
        header += [f"{c}__rate_pct", f"{c}__n", f"{c}__avg_steps"]

    def row_for(task):
        row = {"task": task}
        for c in conditions:
            s = per_condition[c].get(task)
            if s is None:
                row[f"{c}__rate_pct"] = ""
                row[f"{c}__n"] = ""
                row[f"{c}__avg_steps"] = ""
            else:
                row[f"{c}__rate_pct"] = f"{s['rate_pct']:.1f}"
                # Wrap "n/total" as an Excel string formula so it isn't auto-coerced to a date.
                row[f"{c}__n"] = f'="{s["n_success"]}/{s["n_episodes"]}"'
                row[f"{c}__avg_steps"] = (
                    "" if np.isnan(s["avg_steps"]) else f"{s['avg_steps']:.1f}"
                )
        return row

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for task in task_order:
            writer.writerow(row_for(task))
        writer.writerow(row_for("OVERALL"))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
