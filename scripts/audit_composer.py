"""Phase 0 audit — catalog composer emissions across the 28 CALVIN tasks.

Standalone measurement tool (not production code). For each of the 28 tasks in
`conf/evaluation/task_order.json`, run the composer LMP across one or more
instruction sources (canonical from `auto_lang_ann.npy`, perturbed P1–P4 from
`perturbed_language_annotations.json`, or both), capture the raw LLM output +
the result of `steering.stage_spec.parse_composer_stages`, classify each
stage, and optionally classify the whole emission against the task's expected
(primitive, object) sequence from `action_primitive_object_annotations.json`.

The ground-truth annotations file is loaded ONLY by this audit script
post-emission — NEVER by the composer at inference time. The negative
leakage grep at Phase 3a/3b acceptance verifies that no production module
imports it.

Usage:
    # Canonical only (matches 3a's audit behavior):
    uv run python scripts/audit_composer.py \
        --tasks conf/evaluation/task_order.json \
        --out docs/refactor/task3b_baseline_emissions_canonical.json \
        --cache-dir /tmp/task3b_audit_cache_gpt5 \
        --llm-model gpt-5-chat-latest

    # Canonical + P1-P4 perturbations with ground-truth classification (3b §2.1):
    uv run python scripts/audit_composer.py \
        --tasks conf/evaluation/task_order.json \
        --out docs/refactor/task3b_composer_perturbation.json \
        --cache-dir /tmp/task3b_audit_cache_gpt5 \
        --llm-model gpt-5-chat-latest \
        --instruction-source all \
        --perturbed-ann-path perturbed_language_annotations.json \
        --ground-truth action_primitive_object_annotations.json \
        --use-linter
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

# Project root on sys.path so `voxposer.*` / `steering.*` import like the runners do.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.calvin_utils.language_ann import (  # noqa: E402
    get_instruction_for_task,
    load_language_annotations,
)
from steering.stage_spec import (  # noqa: E402
    OBJECT_VOCAB,
    PRIMITIVE_VOCAB,
    VALID_STAGE_MODES,
)
from voxposer.lmp import (  # noqa: E402
    VocabValidationError,
    compose_with_repair,
    set_lmp_objects,
    setup_lmp,
)

logger = logging.getLogger(__name__)

PERTURBATION_AXES = ("P1", "P2", "P3", "P4")


def _classify_stage(raw: Any, idx: int) -> dict:
    """Classify one raw composer tuple — mirrors `stage_spec._parse_one`
    but returns a diagnostic record instead of a StageSpec.
    """
    record: dict = {
        "idx": idx,
        "raw_repr": repr(raw)[:200],
        "len": None,
        "mode": None,
        "primitive": None,
        "object": None,
        "has_rot_target": False,
        "dropped": False,
        "dropped_reason": None,
    }

    if not isinstance(raw, (tuple, list)):
        record["dropped"] = True
        record["dropped_reason"] = f"expected tuple/list, got {type(raw).__name__}"
        return record

    record["len"] = len(raw)
    n = len(raw)

    def _check_primitive(p: Any) -> tuple[bool, str | None]:
        if isinstance(p, str) and p in PRIMITIVE_VOCAB:
            return True, None
        return False, f"invalid primitive {p!r}"

    def _check_object(o: Any) -> tuple[bool, str | None]:
        if isinstance(o, str) and o in OBJECT_VOCAB:
            return True, None
        return False, f"invalid object {o!r}"

    def _mode_or_default(m: Any) -> str:
        if isinstance(m, str) and m in VALID_STAGE_MODES:
            return m
        return "static"  # default for malformed; parser still accepts

    if n == 2:
        record["mode"] = "static"
        return record
    if n == 3:
        record["mode"] = _mode_or_default(raw[2])
        return record
    if n == 4:
        record["mode"] = _mode_or_default(raw[2])
        ok, reason = _check_primitive(raw[3])
        if not ok:
            record["dropped"] = True
            record["dropped_reason"] = reason
        record["primitive"] = raw[3] if isinstance(raw[3], str) else None
        return record
    if n == 5:
        if isinstance(raw[2], str) and raw[2] in VALID_STAGE_MODES:
            record["mode"] = _mode_or_default(raw[2])
            record["primitive"] = raw[3] if isinstance(raw[3], str) else None
            record["object"] = raw[4] if isinstance(raw[4], str) else None
            okp, rp = _check_primitive(raw[3])
            oko, ro = _check_object(raw[4])
            if not okp or not oko:
                record["dropped"] = True
                record["dropped_reason"] = "; ".join(r for r in (rp, ro) if r)
            return record
        record["has_rot_target"] = raw[2] is not None
        record["mode"] = _mode_or_default(raw[3])
        record["primitive"] = raw[4] if isinstance(raw[4], str) else None
        okp, rp = _check_primitive(raw[4])
        if not okp:
            record["dropped"] = True
            record["dropped_reason"] = rp
        return record
    if n == 6:
        record["has_rot_target"] = raw[2] is not None
        record["mode"] = _mode_or_default(raw[3])
        record["primitive"] = raw[4] if isinstance(raw[4], str) else None
        record["object"] = raw[5] if isinstance(raw[5], str) else None
        okp, rp = _check_primitive(raw[4])
        oko, ro = _check_object(raw[5])
        if not okp or not oko:
            record["dropped"] = True
            record["dropped_reason"] = "; ".join(r for r in (rp, ro) if r)
        return record

    record["dropped"] = True
    record["dropped_reason"] = f"expected 2- to 6-tuple, got len={n}"
    return record


def _classify_final_status(entry: dict, gt: Optional[dict]) -> str:
    """Bucket the whole emission into one of the 5 §2.1 final-status buckets.

    `gt` is the ground-truth annotation for this task (or None for canonical
    audits where ground-truth isn't loaded).
    """
    if entry["composer_error"]:
        if entry["composer_error"].startswith("VocabValidationError"):
            return "invalid_after_retries"
        return "structurally_broken"

    stages = entry["stages"]
    # If parser dropped any stage, the emission would have been rejected pre-linter.
    if any(s["dropped"] for s in stages):
        # Linter would have caught these; if it ran (use_linter=True) and we got
        # here, something else happened. Bucket as `structurally_broken`.
        return "structurally_broken"

    if gt is None:
        # No ground truth — best we can say is "valid"; don't try to bucket finer.
        return "valid"

    gt_primitives = set(gt.get("primitives", []))
    gt_objects = set(gt.get("objects", []))
    emitted_prims = [s["primitive"] for s in stages if s["primitive"] is not None]
    emitted_objs = [s["object"] for s in stages if s["object"] is not None]

    bad_prims = [p for p in emitted_prims if p not in gt_primitives]
    bad_objs = [o for o in emitted_objs if o not in gt_objects]

    if bad_prims and bad_objs:
        # Both wrong — surface as primitive wrong since that's more diagnostic.
        return "valid_wrong_primitive"
    if bad_prims:
        return "valid_wrong_primitive"
    if bad_objs:
        return "valid_wrong_object"
    return "valid_correct"


def audit_one(
    lmps: dict,
    lmp_interface: Any,
    task_name: str,
    instruction: str,
    *,
    variant: str,
    use_linter: bool = False,
    ground_truth_entry: Optional[dict] = None,
) -> dict:
    """Run composer once on `instruction` and classify each emitted stage."""
    # Fresh per-task object context; matches setup_episode behavior.
    set_lmp_objects(lmps, lmp_interface.get_object_names())

    entry: dict = {
        "task": task_name,
        "variant": variant,
        "instruction": instruction,
        "composer_error": None,
        "raw_output_repr": None,
        "num_stages_raw": 0,
        "num_stages_kept": 0,
        "stages": [],
        "used_linter": use_linter,
        "final_status": None,
    }

    try:
        if use_linter:
            result = compose_with_repair(lmps["composer"], instruction)
        else:
            result = lmps["composer"](instruction)
    except VocabValidationError as e:
        entry["composer_error"] = f"VocabValidationError: {e}"
        entry["final_status"] = _classify_final_status(entry, ground_truth_entry)
        return entry
    except Exception as e:
        entry["composer_error"] = f"{type(e).__name__}: {e}"
        entry["composer_traceback"] = traceback.format_exc(limit=3)
        entry["final_status"] = _classify_final_status(entry, ground_truth_entry)
        return entry

    if isinstance(result, tuple):
        raw_stages: list = [result]
    elif isinstance(result, list):
        raw_stages = result
    else:
        entry["raw_output_repr"] = repr(result)[:200]
        entry["composer_error"] = (
            f"unexpected composer result type: {type(result).__name__}"
        )
        entry["final_status"] = _classify_final_status(entry, ground_truth_entry)
        return entry

    entry["raw_output_repr"] = repr(result)[:400]
    entry["num_stages_raw"] = len(raw_stages)
    entry["stages"] = [_classify_stage(s, i) for i, s in enumerate(raw_stages)]
    entry["num_stages_kept"] = sum(1 for s in entry["stages"] if not s["dropped"])
    entry["final_status"] = _classify_final_status(entry, ground_truth_entry)
    return entry


def _resolve_variants(
    args, task: str, annotations: Any, perturbed: Optional[dict],
) -> list[tuple[str, str]]:
    """Build the list of (variant, instruction) pairs to audit for `task`."""
    pairs: list[tuple[str, str]] = []
    if args.instruction_source in ("canonical", "all"):
        pairs.append(("canonical", get_instruction_for_task(task, annotations)))
    if args.instruction_source in ("perturbed", "all"):
        if perturbed is None:
            return pairs
        task_perturbed = perturbed.get(task) or {}
        for axis in PERTURBATION_AXES:
            if axis in task_perturbed:
                pairs.append((axis, task_perturbed[axis]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True,
                        help="Path to task_order.json")
    parser.add_argument("--out", required=True, help="Path to baseline JSON")
    parser.add_argument("--cache-dir", required=True,
                        help="LLM disk cache directory (fresh per audit)")
    parser.add_argument(
        "--lang-ann-path",
        default="/home/mlewkowicz/calvin/dataset/task_D_D/validation/"
                "lang_annotations/auto_lang_ann.npy",
        help="CALVIN canonical language annotations .npy",
    )
    parser.add_argument(
        "--perturbed-ann-path",
        default="perturbed_language_annotations.json",
        help="P1–P4 perturbed annotations JSON. Used when --instruction-source "
             "is 'perturbed' or 'all'.",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to action_primitive_object_annotations.json for "
             "final-status classification (valid_correct vs wrong_object/primitive). "
             "Audit-script-only — never read by the composer at inference time.",
    )
    parser.add_argument(
        "--instruction-source", default="canonical",
        choices=("canonical", "perturbed", "all"),
        help="Which instruction set to audit. 'canonical' = auto_lang_ann.npy "
             "default. 'perturbed' = P1–P4 only. 'all' = canonical + P1–P4.",
    )
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=512)
    parser.add_argument("--map-size", type=int, default=100)
    parser.add_argument(
        "--use-linter", action="store_true",
        help="Route composer calls through compose_with_repair (vocab linter "
             "+ re-prompt loop). Default off — measures raw composer behavior.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # Tasks
    tasks_payload = json.loads(Path(args.tasks).read_text())
    task_names = tasks_payload["tasks"]
    logger.info(f"Auditing {len(task_names)} tasks (source={args.instruction_source})")

    # Instructions
    annotations = load_language_annotations(args.lang_ann_path)

    perturbed: Optional[dict] = None
    if args.instruction_source in ("perturbed", "all"):
        perturbed = json.loads(Path(args.perturbed_ann_path).read_text())
        logger.info(f"Loaded perturbed annotations for {len(perturbed)} tasks")

    ground_truth_full: Optional[dict] = None
    if args.ground_truth:
        gt_payload = json.loads(Path(args.ground_truth).read_text())
        ground_truth_full = gt_payload.get("task_annotation_schema", gt_payload)
        logger.info(f"Loaded ground truth for {len(ground_truth_full)} tasks")

    config = {
        "map_size": args.map_size,
        "workspace_bounds_min": [-0.35, -0.60, 0.30],
        "workspace_bounds_max": [0.35, 0.15, 0.85],
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "llm_temperature": args.llm_temperature,
        "llm_max_tokens": args.llm_max_tokens,
        "cache_dir": args.cache_dir,
        "load_cache": True,
    }
    lmps, lmp_interface = setup_lmp(config)

    out_payload: dict = {
        "meta": {
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "cache_dir": args.cache_dir,
            "instruction_source": args.instruction_source,
            "use_linter": args.use_linter,
            "ground_truth_used": args.ground_truth is not None,
        },
        "per_task_composer_emissions": [],
    }

    for task in task_names:
        pairs = _resolve_variants(args, task, annotations, perturbed)
        gt = ground_truth_full.get(task) if ground_truth_full else None
        for variant, instruction in pairs:
            entry = audit_one(
                lmps, lmp_interface, task, instruction,
                variant=variant, use_linter=args.use_linter,
                ground_truth_entry=gt,
            )
            logger.info(
                f"[{task}/{variant}] inst={instruction!r} "
                f"stages={entry['num_stages_kept']}/{entry['num_stages_raw']} "
                f"status={entry['final_status']} err={entry['composer_error']}"
            )
            for s in entry["stages"]:
                if s["dropped"]:
                    logger.warning(
                        f"  drop stage {s['idx']}: {s['dropped_reason']} "
                        f"(prim={s['primitive']!r} obj={s['object']!r})"
                    )
            out_payload["per_task_composer_emissions"].append(entry)

    # Summary tallies.
    emissions = out_payload["per_task_composer_emissions"]
    total_stages = sum(e["num_stages_raw"] for e in emissions)
    total_dropped = sum(
        sum(1 for s in e["stages"] if s["dropped"]) for e in emissions
    )
    final_status_counts: dict[str, int] = {}
    per_variant_status_counts: dict[str, dict[str, int]] = {}
    for e in emissions:
        fs = e["final_status"] or "unclassified"
        final_status_counts[fs] = final_status_counts.get(fs, 0) + 1
        per_variant_status_counts.setdefault(e["variant"], {})[fs] = (
            per_variant_status_counts.setdefault(e["variant"], {}).get(fs, 0) + 1
        )

    out_payload["summary"] = {
        "audits_total": len(emissions),
        "tasks_total": len(task_names),
        "variants_per_task_avg": (
            len(emissions) / len(task_names) if task_names else 0
        ),
        "stages_total": total_stages,
        "stages_dropped": total_dropped,
        "final_status_counts": final_status_counts,
        "per_variant_status_counts": per_variant_status_counts,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))
    logger.info(f"Audit written to {out_path}")
    logger.info(f"Summary: {out_payload['summary']}")


if __name__ == "__main__":
    main()
