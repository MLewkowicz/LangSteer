"""Generate primitive+object language annotations from CALVIN task annotations.

Reads auto_lang_ann.npy + action_primitive_object_annotations.json, then for each
annotated episode emits labels of the form "<primitive> <object>" (e.g. "grasp drawer_handle",
"pull drawer", "rotate blue_block").

  - single-stage task: emit one label spanning (start, end)
  - multi-stage task: find frame where the target scene_obs observable starts
    changing; split into stage-0 + stage-1 labels

Output mirrors auto_lang_ann.npy format so existing loaders work unchanged.

Example:
    uv run python scripts/preprocess_primitive_object_annotations.py \\
        --dataset_path /home/mlewkowicz/calvin/dataset/task_D_D \\
        --split training \\
        --primitive_schema action_primitive_object_annotations.json \\
        --output_path /tmp/primitive_object_lang_ann.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("preprocess_primitive_object_annotations")


def _load_scene_obs_sequence(
    split_dir: Path, start_id: int, end_id: int
) -> tuple[np.ndarray, list[int]]:
    """Load scene_obs for frames [start_id, end_id] inclusive."""
    frames = []
    obs = []
    for ep_id in range(start_id, end_id + 1):
        ep_path = split_dir / f"episode_{ep_id:07d}.npz"
        if not ep_path.is_file():
            continue
        with np.load(ep_path) as data:
            obs.append(np.asarray(data["scene_obs"], dtype=np.float64))
        frames.append(ep_id)
    if not obs:
        return np.empty((0, 24)), []
    return np.stack(obs, axis=0), frames


def find_transition_frame(
    scene_obs_seq: np.ndarray,
    observable: dict,
) -> Optional[int]:
    """Return the index within scene_obs_seq where target motion first exceeds threshold."""
    if scene_obs_seq.shape[0] < 2:
        return None

    indices = observable["indices"]
    direction = observable.get("direction", "any")
    threshold = float(observable.get("threshold", 0.005))

    initial = scene_obs_seq[0, indices]
    series = scene_obs_seq[:, indices]
    delta = series - initial

    if direction == "increasing":
        qualifies = delta >= threshold
    elif direction == "decreasing":
        qualifies = -delta >= threshold
    elif direction == "any":
        qualifies = np.abs(delta) >= threshold
    else:
        raise ValueError(f"Unknown direction '{direction}'")

    per_frame = qualifies.any(axis=1)
    hits = np.where(per_frame)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def build_primitive_object_annotations(
    ann: dict,
    schema: dict,
    split_dir: Path,
) -> tuple[dict, dict]:
    """Convert task annotations -> primitive+object annotations.

    Labels have the form "<primitive> <object>" (e.g. "grasp blue_block").
    Returns (output_dict_in_auto_lang_ann_format, stats_dict).
    """
    task_schema = schema["task_annotation_schema"]

    out_ann: list[str] = []
    out_task: list[str] = []          # primitive+object labels
    out_primitive: list[str] = []     # bare primitive (for downstream filtering)
    out_object: list[str] = []        # bare object (for downstream filtering)
    out_indx: list[tuple[int, int]] = []
    out_parent_task: list[str] = []

    in_tasks = list(ann["language"]["task"])
    in_indx = list(ann["info"]["indx"])

    stats = Counter()
    fallback_log: list[dict] = []
    unknown_tasks: set[str] = set()

    for i, (task, (start_id, end_id)) in enumerate(
        tqdm(list(zip(in_tasks, in_indx)), desc="Annotating primitive+object")
    ):
        task = str(task)
        if task not in task_schema:
            unknown_tasks.add(task)
            stats["unknown_task"] += 1
            continue

        entry = task_schema[task]
        primitives: list[str] = entry["primitives"]
        objects: list[str] = entry["objects"]
        start_id, end_id = int(start_id), int(end_id)

        if len(primitives) != len(objects):
            logger.error(
                "Task %s: primitives and objects lists have different lengths; skipping", task
            )
            stats["schema_mismatch"] += 1
            continue

        def _label(p: str, o: str) -> str:
            return f"{p} {o}"

        if len(primitives) == 1:
            lbl = _label(primitives[0], objects[0])
            out_ann.append(lbl)
            out_task.append(lbl)
            out_primitive.append(primitives[0])
            out_object.append(objects[0])
            out_indx.append((start_id, end_id))
            out_parent_task.append(task)
            stats[f"single:{lbl}"] += 1
            continue

        # Multi-stage: find transition frame.
        observable = entry.get("transition_observable")
        if observable is None:
            logger.error(
                "Task %s has %d primitives but no transition_observable; skipping",
                task, len(primitives),
            )
            stats["multi_missing_schema"] += 1
            continue

        scene_obs_seq, loaded_frames = _load_scene_obs_sequence(split_dir, start_id, end_id)
        if scene_obs_seq.shape[0] == 0:
            logger.warning(
                "No episode files found for range %d-%d (task=%s); skipping", start_id, end_id, task
            )
            stats["missing_episodes"] += 1
            continue

        t = find_transition_frame(scene_obs_seq, observable)
        fallback_reason = None
        if t is None:
            fallback_reason = "no_motion_detected"
        elif t == 0:
            fallback_reason = "motion_at_frame_0"
        if fallback_reason is not None:
            t = max(1, len(loaded_frames) // 2)
            fallback_log.append({
                "ann_index": i,
                "task": task,
                "range": (start_id, end_id),
                "reason": fallback_reason,
            })
            stats["fallback_midpoint"] += 1

        transition_frame = loaded_frames[t]
        if transition_frame <= start_id:
            transition_frame = start_id + 1
        if transition_frame > end_id:
            transition_frame = end_id

        lbl0 = _label(primitives[0], objects[0])
        lbl1 = _label(primitives[1], objects[1])

        out_ann.append(lbl0)
        out_task.append(lbl0)
        out_primitive.append(primitives[0])
        out_object.append(objects[0])
        out_indx.append((start_id, transition_frame - 1))
        out_parent_task.append(task)

        out_ann.append(lbl1)
        out_task.append(lbl1)
        out_primitive.append(primitives[1])
        out_object.append(objects[1])
        out_indx.append((transition_frame, end_id))
        out_parent_task.append(task)

        stats[f"multi:{task}->{lbl0}+{lbl1}"] += 1

    if unknown_tasks:
        logger.warning(
            "Skipped %d annotations with unknown tasks: %s",
            stats["unknown_task"], sorted(unknown_tasks),
        )
    if fallback_log:
        logger.warning("%d annotations fell back to midpoint split", len(fallback_log))

    emb = np.zeros((len(out_ann), 1, 384), dtype=np.float32)

    output = {
        "language": {
            "ann": out_ann,
            "task": out_task,
            "emb": emb,
        },
        "info": {
            "indx": out_indx,
            "episodes": [],
            "parent_task": out_parent_task,
            "primitive": out_primitive,
            "object": out_object,
        },
    }
    return output, {"counts": dict(stats), "fallbacks": fallback_log}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="CALVIN dataset root (contains training/ and validation/)")
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument("--primitive_schema", type=Path,
                        default=Path(__file__).resolve().parent.parent / "action_primitive_object_annotations.json")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    split_dir = args.dataset_path / args.split
    ann_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.is_file():
        logger.error("Input annotations not found: %s", ann_path)
        return 2

    with open(args.primitive_schema) as f:
        schema = json.load(f)

    ann = np.load(ann_path, allow_pickle=True).item()
    logger.info("Loaded %d source annotations from %s", len(ann["language"]["ann"]), ann_path)

    output, stats = build_primitive_object_annotations(ann, schema, split_dir)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, output, allow_pickle=True)
    logger.info(
        "Wrote %d primitive+object annotations to %s",
        len(output["language"]["ann"]), args.output_path,
    )

    counts = stats["counts"]
    total_out = len(output["language"]["ann"])
    label_counts = Counter(output["language"]["task"])
    primitive_counts = Counter(output["info"]["primitive"])
    object_counts = Counter(output["info"]["object"])
    logger.info("Summary:")
    logger.info("  total output entries: %d", total_out)
    logger.info("  label distribution: %s", dict(label_counts))
    logger.info("  primitive distribution: %s", dict(primitive_counts))
    logger.info("  object distribution: %s", dict(object_counts))
    logger.info("  per-kind counts: %s", counts)
    if stats["fallbacks"]:
        logger.info(
            "  fallbacks logged: %d (first 5: %s)",
            len(stats["fallbacks"]), stats["fallbacks"][:5],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
