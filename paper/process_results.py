"""Process evaluation JSON files and print a LaTeX-ready results table."""

import json
import argparse
from pathlib import Path

# Task groupings matching the paper table columns.
# Each entry: (display_name, [task_keys])
GROUPS = [
    ("rotate block", [
        "rotate_red_block_right", "rotate_red_block_left",
        "rotate_blue_block_right", "rotate_blue_block_left",
        "rotate_pink_block_right", "rotate_pink_block_left",
    ]),
    ("push block", [
        "push_red_block_right", "push_red_block_left",
        "push_blue_block_right", "push_blue_block_left",
        "push_pink_block_right", "push_pink_block_left",
    ]),
    ("move slider", [
        "move_slider_left", "move_slider_right",
    ]),
    ("open drawer", [
        "open_drawer",
    ]),
    ("close drawer", [
        "close_drawer",
    ]),
    ("lift from table", [
        "lift_red_block_table", "lift_blue_block_table", "lift_pink_block_table",
    ]),
    ("lift from slider", [
        "lift_red_block_slider", "lift_blue_block_slider", "lift_pink_block_slider",
    ]),
    ("lift from drawer", [
        "lift_red_block_drawer", "lift_blue_block_drawer", "lift_pink_block_drawer",
    ]),
    ("place in slider", [
        "place_in_slider",
    ]),
    ("place in drawer", [
        "place_in_drawer",
    ]),
    ("push into drawer", [
        "push_into_drawer",
    ]),
    ("stack block", [
        "stack_block",
    ]),
    ("unstack block", [
        "unstack_block",
    ]),
    ("turn on lightbulb", [
        "turn_on_lightbulb",
    ]),
    ("turn off lightbulb", [
        "turn_off_lightbulb",
    ]),
    ("turn on led", [
        "turn_on_led",
    ]),
    ("turn off led", [
        "turn_off_led",
    ]),
]


def task_stats(episodes: list[dict]) -> tuple[float, float | None]:
    """Return (success_rate_pct, mean_steps_on_success_or_None)."""
    successes = [e for e in episodes if e["success"]]
    sr = 100.0 * len(successes) / len(episodes)
    steps = sum(e["steps"] for e in successes) / len(successes) if successes else None
    return sr, steps


def group_stats(tasks: dict, task_keys: list[str]) -> tuple[float | None, float | None, int]:
    """Return (mean_sr, mean_steps, n_tasks_found)."""
    srs, stepss = [], []
    for key in task_keys:
        if key in tasks:
            sr, steps = task_stats(tasks[key]["episodes"])
            srs.append(sr)
            if steps is not None:
                stepss.append(steps)
    if not srs:
        return None, None, 0
    return (
        sum(srs) / len(srs),
        sum(stepss) / len(stepss) if stepss else None,
        len(srs),
    )


def process_file(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    condition = data.get("condition_name", path.stem)
    tasks = data["tasks"]

    all_srs, all_steps = [], []
    for eps in tasks.values():
        sr, steps = task_stats(eps["episodes"])
        all_srs.append(sr)
        if steps is not None:
            all_steps.append(steps)

    overall_sr = sum(all_srs) / len(all_srs)
    overall_steps = sum(all_steps) / len(all_steps) if all_steps else None

    group_results = {}
    for name, keys in GROUPS:
        sr, steps, n = group_stats(tasks, keys)
        group_results[name] = (sr, steps, n)

    return {
        "condition": condition,
        "overall_sr": overall_sr,
        "overall_steps": overall_steps,
        "groups": group_results,
    }


def fmt_sr(val: float | None) -> str:
    return f"{val:.1f}" if val is not None else "---"


def fmt_steps(val: float | None) -> str:
    return f"{val:.1f}" if val is not None else "---"


def print_table(results: list[dict]) -> None:
    group_names = [g[0] for g in GROUPS]
    counts = {}
    if results:
        for name, _ in GROUPS:
            _, _, n = results[0]["groups"][name]
            counts[name] = n

    headers = " | ".join(f"{name}({counts.get(name,0)}) SR / Steps" for name in group_names)
    print(f"{'Condition':<35} | {'Avg SR':>7} | {'Avg Steps':>9} | {headers}")
    print("-" * 140)
    for r in results:
        group_vals = " | ".join(
            f"{fmt_sr(r['groups'][n][0]):>7} / {fmt_steps(r['groups'][n][1]):>6}"
            for n in group_names
        )
        print(
            f"{r['condition']:<35} | {fmt_sr(r['overall_sr']):>7} | "
            f"{fmt_steps(r['overall_steps']):>9} | {group_vals}"
        )


def print_latex_rows(results: list[dict]) -> None:
    group_names = [g[0] for g in GROUPS]
    print("\n% LaTeX rows (paste into tabular):")
    for r in results:
        cells = []
        for n in group_names:
            sr, steps, _ = r["groups"][n]
            cells.append(fmt_sr(sr))
            cells.append(fmt_steps(steps))
        group_str = " & ".join(cells)
        print(
            f"{r['condition']}\n"
            f"  & {fmt_sr(r['overall_sr'])} & {fmt_steps(r['overall_steps'])} & {group_str} \\\\"
        )


def main():
    parser = argparse.ArgumentParser(description="Process LangSteer evaluation JSONs.")
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more langsteer_primitive.json result files.",
    )
    args = parser.parse_args()

    results = [process_file(p) for p in args.files]

    print_table(results)
    print_latex_rows(results)


if __name__ == "__main__":
    main()
