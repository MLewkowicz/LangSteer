"""Task 7 Phase 0 — VLM scene-grounding model A/B audit.

Captures 8 annotated overhead frames from the eval task list (mix of static,
cavity, light, and P4 perturbation cases), runs the `scene_grounding` LMP
prompt against both gpt-4o and gpt-5.4-mini, scores each output dict against
hand-labeled ground truth, and writes `docs/refactor/task7_phase0_audit.json`.

Phase 0 is read-only (no inference-side changes). The output JSON drives the
model-choice decision for Phases 2-5.

Usage:
    uv run python scripts/audit_scene_grounding.py [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.calvin import CalvinEnvironment
from voxposer.calvin_interface import CalvinLMPInterface
from voxposer.llm_cache import DiskCache
from voxposer.lmp import LLMBackend
from voxposer.scene_image import render_annotated_overhead

logger = logging.getLogger(__name__)


# 8 audit scenes: mix of static-table, cavity, light, and P4 perturbation
# cases. Each entry carries the task name, the instruction (canonical or P4),
# and the hand-labeled ground-truth grounding dict.
AUDIT_SCENES: list[dict[str, Any]] = [
    # Ground truths derive from CALVIN's `TASK_INITIAL_CONDITIONS` in
    # `envs/calvin_utils/task_configs.py`. All audit tasks reset to a
    # canonical initial state where blocks rest on the table regardless of
    # task name (e.g. `lift_blue_block_drawer` starts with the blue block
    # ON the table but the drawer OPEN). Fixture states vary by task.
    # --- Canonical (P0) cases ---
    {
        "scene_id": "lift_red_block_table_canonical",
        "task": "lift_red_block_table",
        "instruction": "pick up the red block",
        "ground_truth": {
            "blocks_visible": {"red_block": "table", "blue_block": "table", "pink_block": "table"},
            "fixtures_state": {"drawer": "closed", "slider": "right", "lightbulb": "off", "led": "off"},
            "ambiguous_resolutions": {},
        },
    },
    {
        "scene_id": "place_in_slider_canonical",
        "task": "place_in_slider",
        "instruction": "store the block inside the sliding cabinet",
        "ground_truth": {
            "blocks_visible": {"red_block": "table", "blue_block": "table", "pink_block": "table"},
            "fixtures_state": {"drawer": "closed", "slider": "right", "lightbulb": "off", "led": "off"},
            "ambiguous_resolutions": {"the block": "red_block"},
        },
    },
    {
        "scene_id": "lift_blue_block_drawer_canonical",
        "task": "lift_blue_block_drawer",
        "instruction": "grasp the blue block lying in the drawer",
        "ground_truth": {
            "blocks_visible": {"red_block": "table", "blue_block": "table", "pink_block": "table"},
            "fixtures_state": {"drawer": "open", "slider": "right", "lightbulb": "off", "led": "off"},
            "ambiguous_resolutions": {},
        },
    },
    {
        "scene_id": "turn_on_lightbulb_canonical",
        "task": "turn_on_lightbulb",
        "instruction": "switch the lightbulb on",
        "ground_truth": {
            "blocks_visible": {"red_block": "table", "blue_block": "table", "pink_block": "table"},
            "fixtures_state": {"drawer": "closed", "slider": "right", "lightbulb": "off", "led": "off"},
            "ambiguous_resolutions": {},
        },
    },
    {
        "scene_id": "turn_on_led_canonical",
        "task": "turn_on_led",
        "instruction": "push the button to turn on the led",
        "ground_truth": {
            "blocks_visible": {"red_block": "table", "blue_block": "table", "pink_block": "table"},
            "fixtures_state": {"drawer": "closed", "slider": "right", "lightbulb": "off", "led": "off"},
            "ambiguous_resolutions": {},
        },
    },
    # --- P4 perturbation cases ---
    {
        "scene_id": "lift_red_block_table_P4",
        "task": "lift_red_block_table",
        "instruction": "Lift a block from the table.",
        "ground_truth": {
            "blocks_visible": {"red_block": "table", "blue_block": "table", "pink_block": "table"},
            "fixtures_state": {"drawer": "closed", "slider": "right", "lightbulb": "off", "led": "off"},
            "ambiguous_resolutions": {"a block": "red_block"},
        },
    },
    {
        "scene_id": "turn_on_lightbulb_P4",
        "task": "turn_on_lightbulb",
        "instruction": "Turn on the light.",
        "ground_truth": {
            "blocks_visible": {"red_block": "table", "blue_block": "table", "pink_block": "table"},
            "fixtures_state": {"drawer": "closed", "slider": "right", "lightbulb": "off", "led": "off"},
            "ambiguous_resolutions": {"the light": "lightbulb"},
        },
    },
    {
        "scene_id": "lift_blue_block_drawer_P4",
        "task": "lift_blue_block_drawer",
        "instruction": "Pick up the block from the drawer.",
        "ground_truth": {
            # P4 instruction implies the block is in the drawer; the actual
            # initial scene has it on the table. The VLM should report the
            # blocks' actual locations (all on table) and still resolve the
            # ambiguous phrase to blue_block (the canonical task's color).
            "blocks_visible": {"red_block": "table", "blue_block": "table", "pink_block": "table"},
            "fixtures_state": {"drawer": "open", "slider": "right", "lightbulb": "off", "led": "off"},
            "ambiguous_resolutions": {"the block": "blue_block", "the block from the drawer": "blue_block"},
        },
    },
]

MODELS_TO_AUDIT = ["gpt-4o", "gpt-5.4-mini"]

# OpenAI token-pricing snapshot (per million tokens) for cost estimation.
# Source: openai.com/pricing as of 2026-05-19.
TOKEN_PRICES = {
    "gpt-4o":        {"input": 2.50, "output": 10.00},   # $/1M tokens
    "gpt-5.4-mini":  {"input": 0.15, "output": 0.60},    # cheaper text-only
}


def setup_env(dataset_path: str) -> tuple[CalvinEnvironment, CalvinLMPInterface]:
    """Build a CalvinEnvironment + LMP interface for scene capture."""
    env_cfg = {
        "task": "open_drawer",  # any default; overridden per-scene
        "dataset_path": dataset_path,
        "split": "validation",
        "lang_ann_path": f"{dataset_path}/validation/lang_annotations/auto_lang_ann.npy",
        "use_gui": False,
        "provide_pcd_images": False,
        "num_points": 2048,
        "use_task_initial_condition": True,
        "randomize_initial_condition": False,
        "done_on_success": False,
        "max_steps": 60,
    }
    env = CalvinEnvironment(env_cfg)
    lmp_cfg = {
        "map_size": 100,
        "workspace_bounds_min": [-0.35, -0.60, 0.30],
        "workspace_bounds_max": [0.35, 0.15, 0.85],
    }
    lmp_iface = CalvinLMPInterface(lmp_cfg)
    return env, lmp_iface


def capture_scene(
    env: CalvinEnvironment, lmp_iface: CalvinLMPInterface, task: str,
    width: int = 600, height: int = 600, fov: float = 20.0,
) -> tuple[bytes, list]:
    """Reset env to task's initial condition; return (annotated JPEG bytes, detections)."""
    env.set_task(task)
    env.reset()

    state = env.get_scene_state()
    lmp_iface.update_state(
        state["robot_obs"], state["scene_obs"],
        fixture_positions=state.get("fixture_positions"),
        block_aabbs=state.get("block_aabbs"),
    )
    detections = lmp_iface.get_all_detections()

    rgb = env.render_high_res_static(width, height, fov=fov)
    view_mat, proj_mat = env.get_static_camera_matrices(width, height, fov=fov)

    jpeg = render_annotated_overhead(rgb, detections, view_mat, proj_mat)
    return jpeg, detections


def load_prompt() -> str:
    return Path(__file__).parent.parent.joinpath(
        "voxposer/prompts/calvin/scene_grounding_prompt.txt"
    ).read_text()


def run_grounding(
    backend: LLMBackend, prompt_template: str, instruction: str,
    image_bytes: bytes,
) -> tuple[str, dict | None, str | None]:
    """Call the model with the grounding prompt + image. Returns (raw_text, parsed_dict, parse_error)."""
    prompt = prompt_template + f" instruction = {instruction!r}, annotated scene.\n"
    raw = backend.generate(prompt, stop=["# Query:"], image_bytes=image_bytes)

    # Try to eval the `ret_val = {...}` assignment safely.
    parsed: dict | None = None
    err: str | None = None
    try:
        loc: dict = {}
        # The model is expected to emit `ret_val = {...}`. Strip stop sequence
        # leftovers and `python` fences (the backend already strips ``` but
        # may leave commentary).
        body = raw.strip()
        if not body.startswith("ret_val"):
            # Pull the ret_val line out of any commentary.
            for line in body.splitlines():
                if line.strip().startswith("ret_val"):
                    body = "\n".join(
                        line for line in body.splitlines() if line.strip()
                    )
                    break
        exec(body, {"__builtins__": {}}, loc)
        parsed = loc.get("ret_val")
        if not isinstance(parsed, dict):
            err = f"ret_val was {type(parsed).__name__}, expected dict"
            parsed = None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return raw, parsed, err


def score_grounding(predicted: dict | None, ground_truth: dict) -> dict[str, Any]:
    """Per-field accuracy. Each top-level key contributes a sub-score."""
    if predicted is None:
        return {
            "blocks_visible_correct": 0, "blocks_visible_total": len(ground_truth["blocks_visible"]),
            "fixtures_state_correct": 0, "fixtures_state_total": len(ground_truth["fixtures_state"]),
            "ambig_correct": 0, "ambig_total": len(ground_truth["ambiguous_resolutions"]),
            "schema_valid": False,
        }

    blocks_correct = sum(
        1 for k, v in ground_truth["blocks_visible"].items()
        if predicted.get("blocks_visible", {}).get(k) == v
    )
    fixtures_correct = sum(
        1 for k, v in ground_truth["fixtures_state"].items()
        if predicted.get("fixtures_state", {}).get(k) == v
    )
    # Ambig: count it correct if ANY emitted resolution for the same phrase
    # matches the GT value. Empty GT + empty prediction → trivially correct.
    ambig_gt = ground_truth["ambiguous_resolutions"]
    if not ambig_gt:
        ambig_correct = 1 if not predicted.get("ambiguous_resolutions", {}) else 0
        ambig_total = 1
    else:
        ambig_pred = predicted.get("ambiguous_resolutions", {})
        # Soft match: count correct if the EXPECTED value appears anywhere in
        # the predicted resolutions (handles phrase-wording variance).
        ambig_correct = 0
        for phrase, expected in ambig_gt.items():
            if expected in ambig_pred.values():
                ambig_correct += 1
            elif phrase in ambig_pred and ambig_pred[phrase] == expected:
                ambig_correct += 1
        ambig_total = len(ambig_gt)

    return {
        "blocks_visible_correct": blocks_correct,
        "blocks_visible_total": len(ground_truth["blocks_visible"]),
        "fixtures_state_correct": fixtures_correct,
        "fixtures_state_total": len(ground_truth["fixtures_state"]),
        "ambig_correct": ambig_correct,
        "ambig_total": ambig_total,
        "schema_valid": True,
    }


def aggregate(per_scene_scores: list[dict]) -> dict[str, Any]:
    """Roll up per-scene scores → aggregate accuracy + per-category breakdown."""
    fields = ["blocks_visible", "fixtures_state", "ambig"]
    agg = {}
    total_correct = 0
    total_total = 0
    for f in fields:
        c = sum(s[f"{f}_correct"] for s in per_scene_scores)
        t = sum(s[f"{f}_total"] for s in per_scene_scores)
        agg[f"{f}_accuracy"] = (c / t) if t else 1.0
        total_correct += c
        total_total += t
    agg["aggregate_accuracy"] = (total_correct / total_total) if total_total else 1.0
    agg["schema_valid_count"] = sum(1 for s in per_scene_scores if s["schema_valid"])
    agg["schema_valid_total"] = len(per_scene_scores)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=str, default="/tmp/task7_phase0_audit",
        help="Disk cache for LLM responses (re-runs reuse cached vision calls).",
    )
    parser.add_argument(
        "--output", type=str,
        default="docs/refactor/task7_phase0_audit.json",
        help="Output JSON path (relative to repo root).",
    )
    parser.add_argument(
        "--dataset-path", type=str,
        default="/home/mlewkowicz/calvin/dataset/task_D_D",
        help="CALVIN dataset path for env construction.",
    )
    parser.add_argument(
        "--save-images-to", type=str, default=None,
        help="Optional dir to save the annotated JPEGs for visual inspection.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).parent.parent
    output_path = repo_root / args.output

    prompt_template = load_prompt()
    cache = DiskCache(cache_dir=args.cache_dir)

    # Build env + interface ONCE; reset per scene below.
    env, lmp_iface = setup_env(args.dataset_path)

    # Capture all 8 scene images up front so each model run uses the same
    # frames (and so any env-side errors surface before we burn API calls).
    scenes_data = []
    for scene in AUDIT_SCENES:
        logger.info(f"Capturing scene: {scene['scene_id']} (task={scene['task']})")
        try:
            jpeg, detections = capture_scene(env, lmp_iface, scene["task"])
            if args.save_images_to:
                img_dir = Path(args.save_images_to)
                img_dir.mkdir(parents=True, exist_ok=True)
                (img_dir / f"{scene['scene_id']}.jpg").write_bytes(jpeg)
            scenes_data.append({
                "scene": scene,
                "jpeg": jpeg,
                "n_detections": len(detections),
            })
        except Exception as e:
            logger.error(f"Scene capture failed for {scene['scene_id']}: {e}")
            scenes_data.append({"scene": scene, "jpeg": None, "error": str(e)})

    env.close()

    # Run each model on each scene.
    per_model_results: dict[str, dict] = {}
    for model in MODELS_TO_AUDIT:
        logger.info(f"\n{'='*60}\nRunning model: {model}\n{'='*60}")
        backend = LLMBackend(
            provider="openai",
            model=model,
            temperature=0.0,
            max_tokens=512,
            cache=cache,
        )

        scene_results: list[dict] = []
        for entry in scenes_data:
            scene = entry["scene"]
            scene_id = scene["scene_id"]
            if entry.get("jpeg") is None:
                scene_results.append({
                    "scene_id": scene_id, "error": entry.get("error", "capture failed"),
                    "predicted": None, "score": score_grounding(None, scene["ground_truth"]),
                })
                continue

            t0 = time.time()
            try:
                raw, parsed, parse_err = run_grounding(
                    backend, prompt_template, scene["instruction"], entry["jpeg"]
                )
            except Exception as e:
                logger.error(f"  {scene_id}: API failed — {e}")
                scene_results.append({
                    "scene_id": scene_id, "api_error": str(e),
                    "predicted": None, "score": score_grounding(None, scene["ground_truth"]),
                })
                continue
            dt = time.time() - t0
            score = score_grounding(parsed, scene["ground_truth"])
            logger.info(
                f"  {scene_id}: schema_valid={score['schema_valid']} "
                f"blocks={score['blocks_visible_correct']}/{score['blocks_visible_total']} "
                f"fixtures={score['fixtures_state_correct']}/{score['fixtures_state_total']} "
                f"ambig={score['ambig_correct']}/{score['ambig_total']} "
                f"({dt:.1f}s)"
            )
            scene_results.append({
                "scene_id": scene_id,
                "instruction": scene["instruction"],
                "ground_truth": scene["ground_truth"],
                "raw_response": raw,
                "predicted": parsed,
                "parse_error": parse_err,
                "score": score,
                "wall_clock_s": dt,
            })

        agg = aggregate([r["score"] for r in scene_results])
        per_model_results[model] = {"per_scene": scene_results, "aggregate": agg}
        logger.info(
            f"\n  MODEL {model} aggregate: {agg['aggregate_accuracy']:.1%} "
            f"(blocks={agg['blocks_visible_accuracy']:.1%}, "
            f"fixtures={agg['fixtures_state_accuracy']:.1%}, "
            f"ambig={agg['ambig_accuracy']:.1%}, "
            f"schema_valid={agg['schema_valid_count']}/{agg['schema_valid_total']})"
        )

    # Final summary
    audit_output = {
        "task7_phase0_audit": {
            "models": MODELS_TO_AUDIT,
            "n_scenes": len(AUDIT_SCENES),
            "scenes": [
                {k: v for k, v in s.items() if k != "ground_truth"}
                for s in AUDIT_SCENES
            ],
            "results_by_model": per_model_results,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(audit_output, f, indent=2, default=str)
    logger.info(f"\nAudit written to {output_path}")

    # Decision summary
    print("\n" + "=" * 60)
    print("PHASE 0 AUDIT SUMMARY")
    print("=" * 60)
    for model, res in per_model_results.items():
        agg = res["aggregate"]
        print(f"  {model:20s} agg={agg['aggregate_accuracy']:.1%}  "
              f"blocks={agg['blocks_visible_accuracy']:.1%}  "
              f"fixtures={agg['fixtures_state_accuracy']:.1%}  "
              f"ambig={agg['ambig_accuracy']:.1%}  "
              f"schema={agg['schema_valid_count']}/{agg['schema_valid_total']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
