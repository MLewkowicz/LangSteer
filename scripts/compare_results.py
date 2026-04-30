"""Aggregate and compare benchmark results across multiple evaluation conditions.

Reads LangSteer's per-condition JSON files and prints a per-task success-rate
table with one column per condition.

Usage:
    uv run python scripts/compare_results.py \\
        outputs/evaluation/run/3d_diffuser_actor.json \\
        outputs/evaluation/run/langsteer.json \\
        outputs/evaluation/run/vls.json

    # CSV output (redirect to file):
    uv run python scripts/compare_results.py ... --format csv > results.csv

    # Markdown table (for README / PR comments):
    uv run python scripts/compare_results.py ... --format markdown
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_evaluation import load_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare benchmark results across conditions")
    p.add_argument("json_paths", nargs="+", type=Path, help="Result JSON files (one per condition)")
    p.add_argument("--format", choices=["text", "csv", "markdown"], default="text",
                   help="Output format (default: text)")
    p.add_argument("--sort-by", type=str, default=None,
                   help="Sort tasks by success rate of this condition name (substring match)")
    return p.parse_args()


def _success_rate(episodes: List[dict]) -> tuple[int, int, float]:
    n = len(episodes)
    k = sum(1 for e in episodes if e.get("success", False))
    rate = k / n if n else 0.0
    return k, n, rate


def compare_results(
    json_paths: List[Path],
    output_format: str = "text",
    sort_by: str = None,
) -> None:
    # Load all result files
    conditions: List[Dict] = []
    for path in json_paths:
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        data = load_results(path)
        if not data:
            print(f"WARNING: {path} is empty, skipping", file=sys.stderr)
            continue
        conditions.append(data)

    if not conditions:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    cond_names = [c.get("condition_name", f"Condition {i+1}") for i, c in enumerate(conditions)]

    # Collect all task names across all conditions (preserve order from first file)
    all_tasks: List[str] = []
    seen = set()
    for c in conditions:
        for t in c.get("tasks", {}):
            if t not in seen:
                all_tasks.append(t)
                seen.add(t)

    # Build per-task rate table: {task: {cond_name: (k, n, rate)}}
    table: Dict[str, Dict[str, tuple]] = {}
    for task in all_tasks:
        table[task] = {}
        for cond_name, cond in zip(cond_names, conditions):
            eps = cond.get("tasks", {}).get(task, {}).get("episodes", [])
            table[task][cond_name] = _success_rate(eps)

    # Optional sort
    if sort_by:
        target = next((n for n in cond_names if sort_by.lower() in n.lower()), None)
        if target:
            all_tasks = sorted(all_tasks, key=lambda t: table[t][target][2], reverse=True)
        else:
            print(f"WARNING: --sort-by '{sort_by}' matched no condition", file=sys.stderr)

    # Compute per-condition totals
    totals: Dict[str, tuple] = {}
    for cond_name in cond_names:
        k_tot = sum(table[t][cond_name][0] for t in all_tasks)
        n_tot = sum(table[t][cond_name][1] for t in all_tasks)
        rate_tot = k_tot / n_tot if n_tot else 0.0
        totals[cond_name] = (k_tot, n_tot, rate_tot)

    # --- Render ---
    if output_format == "csv":
        _render_csv(all_tasks, cond_names, table, totals)
    elif output_format == "markdown":
        _render_markdown(all_tasks, cond_names, table, totals)
    else:
        _render_text(all_tasks, cond_names, table, totals)


def _cell(k, n, rate) -> str:
    return f"{k}/{n} ({rate:.0%})" if n else "-"


def _render_text(all_tasks, cond_names, table, totals):
    col_w = max(18, *(len(n) + 2 for n in cond_names))
    task_w = max(35, *(len(t) + 2 for t in all_tasks))
    total_w = task_w + col_w * len(cond_names)

    header = f"{'Task':<{task_w}}" + "".join(f"{n:^{col_w}}" for n in cond_names)
    print("=" * total_w)
    print(header)
    print("-" * total_w)
    for task in all_tasks:
        row = f"{task:<{task_w}}"
        for n in cond_names:
            row += f"{_cell(*table[task][n]):^{col_w}}"
        print(row)
    print("-" * total_w)
    row = f"{'OVERALL':<{task_w}}"
    for n in cond_names:
        row += f"{_cell(*totals[n]):^{col_w}}"
    print(row)
    print("=" * total_w)


def _render_csv(all_tasks, cond_names, table, totals):
    import csv
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["task"] + cond_names)
    for task in all_tasks:
        writer.writerow([task] + [f"{table[task][n][2]:.4f}" for n in cond_names])
    writer.writerow(["OVERALL"] + [f"{totals[n][2]:.4f}" for n in cond_names])
    print(buf.getvalue(), end="")


def _render_markdown(all_tasks, cond_names, table, totals):
    header = "| Task | " + " | ".join(cond_names) + " |"
    sep = "|------|" + "|".join(["------"] * len(cond_names)) + "|"
    print(header)
    print(sep)
    for task in all_tasks:
        cells = " | ".join(_cell(*table[task][n]) for n in cond_names)
        print(f"| {task} | {cells} |")
    overall = " | ".join(_cell(*totals[n]) for n in cond_names)
    print(f"| **OVERALL** | {overall} |")


def main():
    args = parse_args()
    compare_results(args.json_paths, output_format=args.format, sort_by=args.sort_by)


if __name__ == "__main__":
    main()
