"""Stage specifications and composer-output parsing for VoxPoser steering.

This module is the pure-data / pure-parsing layer between the LLM composer
(which emits raw stage tuples) and `StageManager` (which evaluates them). It
holds:

  * the canonical primitive / object / articulated-task vocabularies,
  * the `StageSpec` dataclass that carries one stage's lazy value-map fns,
    primitive/object names, optional rotation target, and cached arrays,
  * `parse_composer_stages` — turns the composer's tuple-or-list output into
    a validated list of `StageSpec`s, dropping malformed entries with a
    warning rather than crashing the rollout,
  * `normalize_rot_target` — resolves a composer rotation target (callable,
    quaternion, 3x3 matrix, ortho-6D row, or flat 3x3) into the canonical
    (6,) ortho-6D form expected by the rotation guidance branch.

Public surface (re-exported from `voxposer_steering` for backwards compatibility
with anything that used to import the private vocab dicts):
    PRIMITIVE_VOCAB, OBJECT_VOCAB, ARTICULATED_TARGET_TASKS, VALID_STAGE_MODES,
    StageSpec, parse_composer_stages, normalize_rot_target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

from policies.diffuser_actor_components.rotation_utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    quaternion_to_matrix,
)

logger = logging.getLogger(__name__)

# Canonical primitive vocabulary (must match PrimitiveEmbedding training order).
# `rotate` is only meaningful in 6-tuple form (always paired with rot_target).
PRIMITIVE_VOCAB: dict[str, int] = {
    "grasp": 0,
    "push": 1,
    "pull": 2,
    "place": 3,
    "rotate": 4,
}

# Canonical object vocabulary — ALPHABETICAL ORDER. Must match
# trainer.OBJECT_VOCAB (and the ordering used to train the
# primitive_object_ABCD checkpoint). Do not reorder without retraining.
OBJECT_VOCAB: dict[str, int] = {
    "block": 0,
    "blue_block": 1,
    "drawer_handle": 2,
    "led_button": 3,
    "lightbulb_switch": 4,
    "pink_block": 5,
    "red_block": 6,
    "slider_handle": 7,
}

VALID_STAGE_MODES: set[str] = {"static", "track"}

# CALVIN tasks whose manipulation target is an articulated joint (slider,
# drawer, switch lever, button) rather than a movable block. The grasp-
# completion gate is disabled for these — the gripper doesn't actually close
# *around* the target, it contacts a handle, so the (min, max, stable) width
# criterion isn't a meaningful "grasp succeeded" signal here. Tasks that
# touch articulated objects but still grasp a block first (e.g. place_in_drawer,
# stack_block) are deliberately NOT in this set.
ARTICULATED_TARGET_TASKS: frozenset[str] = frozenset(
    {
        "open_drawer",
        "close_drawer",
        "move_slider_left",
        "move_slider_right",
        "turn_on_lightbulb",
        "turn_off_lightbulb",
        "turn_on_led",
        "turn_off_led",
        "push_into_drawer",
    }
)


@dataclass
class StageSpec:
    """A single composer-emitted stage with optional refresh-mode metadata.

    `mode='static'` evaluates aff_fn/avoid_fn once at first activation and
    pins the resulting voxel arrays + target centroid for the rest of the
    stage's lifetime — refresh_costmap skips re-eval. Use this for affordances
    that compute a fixed offset from an object's pose (e.g. "15cm to the left
    of the slider handle"); re-evaluating with live state causes the target
    to chase the moving object.

    `mode='track'` re-evaluates each refresh tick, letting the affordance
    follow a moving object. Use for affordances anchored *at* an object.

    `primitive` is the motion primitive the base policy should be conditioned
    on for this stage. Constrained to the 4-symbol vocabulary
    {grasp, push, pull, place}; any other value is rejected at parse time.
    None is allowed only when the policy isn't primitive-id-conditioned.
    """

    aff_fn: Optional[Callable]
    avoid_fn: Optional[Callable]
    mode: str = "static"
    primitive: Optional[str] = None
    # Object name from OBJECT_VOCAB. Forwarded to policy.set_object() when the
    # policy is built with use_object_id=True. None when the policy is
    # primitive-only (or CLIP-conditioned).
    object: Optional[str] = None
    # Optional rotation guidance target. Accepts a callable (resolved at
    # activation), a (3,3) matrix, a (6,) ortho-6D row, a (9,) flattened
    # matrix, or a (4,) wxyz quaternion. Normalized to canonical (6,) by
    # `normalize_rot_target`. None disables rotation guidance for the stage.
    rot_target: Any = None
    cached_affordance: Optional[np.ndarray] = field(default=None, repr=False)
    cached_avoidance: Optional[np.ndarray] = field(default=None, repr=False)
    cached_target: Optional[np.ndarray] = field(default=None, repr=False)
    cached_rotation: Optional[np.ndarray] = field(default=None, repr=False)


def _parse_one(raw: Any, idx: int, default_mode: str) -> Optional[StageSpec]:
    """Normalize a single composer stage tuple into a StageSpec, or None.

    Accepts:
      - (aff_fn, avoid_fn)                                                  → no primitive
      - (aff_fn, avoid_fn, 'static'|'track')                                → explicit mode, no primitive
      - (aff_fn, avoid_fn, 'static'|'track', primitive)                     → primitive
      - (aff_fn, avoid_fn, 'static'|'track', primitive, object)             → primitive + object
      - (aff_fn, avoid_fn, rot_target, 'static'|'track', primitive)         → primitive + rotation
      - (aff_fn, avoid_fn, rot_target, 'static'|'track', primitive, object) → primitive + object + rotation

    `primitive`, when present, MUST be in PRIMITIVE_VOCAB.
    `object`, when present, MUST be in OBJECT_VOCAB.
    Invalid values cause the stage to be dropped — we don't want the LLM
    inventing tokens the policy can't decode.
    """
    if not isinstance(raw, (tuple, list)):
        logger.warning(f"Stage {idx}: expected tuple/list, got {type(raw)}")
        return None

    def _check_primitive(p: Any) -> bool:
        if not (isinstance(p, str) and p in PRIMITIVE_VOCAB):
            logger.error(
                f"Stage {idx}: invalid primitive {p!r}; must be one "
                f"of {sorted(PRIMITIVE_VOCAB)}. Dropping stage."
            )
            return False
        return True

    def _check_object(o: Any) -> bool:
        if not (isinstance(o, str) and o in OBJECT_VOCAB):
            logger.error(
                f"Stage {idx}: invalid object {o!r}; must be one "
                f"of {sorted(OBJECT_VOCAB)}. Dropping stage."
            )
            return False
        return True

    def _normalize_mode(m: Any) -> str:
        if isinstance(m, str) and m in VALID_STAGE_MODES:
            return m
        logger.warning(
            f"Stage {idx}: invalid mode {m!r}, falling back to default '{default_mode}'"
        )
        return default_mode

    if len(raw) == 2:
        return StageSpec(raw[0], raw[1], mode=default_mode)
    if len(raw) == 3:
        return StageSpec(raw[0], raw[1], mode=_normalize_mode(raw[2]))
    if len(raw) == 4:
        mode, primitive = _normalize_mode(raw[2]), raw[3]
        if not _check_primitive(primitive):
            return None
        return StageSpec(raw[0], raw[1], mode=mode, primitive=primitive)
    if len(raw) == 5:
        # Two valid 5-tuple shapes:
        #   (aff, avoid, mode, primitive, object)         — primitive+object, no rotation
        #   (aff, avoid, rot_target, mode, primitive)     — primitive + rotation
        # Disambiguate on the type of raw[2]: a string mode vs anything else (rot target).
        if isinstance(raw[2], str) and raw[2] in VALID_STAGE_MODES:
            mode, primitive, obj = _normalize_mode(raw[2]), raw[3], raw[4]
            if not _check_primitive(primitive) or not _check_object(obj):
                return None
            return StageSpec(raw[0], raw[1], mode=mode, primitive=primitive, object=obj)
        rot_target, mode, primitive = raw[2], _normalize_mode(raw[3]), raw[4]
        if not _check_primitive(primitive):
            return None
        return StageSpec(
            raw[0],
            raw[1],
            mode=mode,
            primitive=primitive,
            rot_target=rot_target,
        )
    if len(raw) == 6:
        # (aff, avoid, rot_target, mode, primitive, object)
        rot_target, mode = raw[2], _normalize_mode(raw[3])
        primitive, obj = raw[4], raw[5]
        if not _check_primitive(primitive) or not _check_object(obj):
            return None
        return StageSpec(
            raw[0],
            raw[1],
            mode=mode,
            primitive=primitive,
            object=obj,
            rot_target=rot_target,
        )
    logger.warning(f"Stage {idx}: expected 2- to 6-tuple, got len={len(raw)}")
    return None


def parse_composer_stages(
    raw_result: Any,
    *,
    default_mode: str = "static",
) -> list[StageSpec]:
    """Turn composer output into a validated list of StageSpec instances.

    Accepts either a single tuple (one-stage shorthand) or a list of tuples
    (multi-stage). Malformed entries are dropped with a warning so a single
    bad LLM emission doesn't abort the whole rollout. Returns `[]` when the
    top-level result type isn't list/tuple.
    """
    if isinstance(raw_result, list):
        raw_stages = raw_result
    elif isinstance(raw_result, tuple):
        raw_stages = [raw_result]
    else:
        logger.warning(f"Unexpected composer result type: {type(raw_result)}")
        return []

    parsed = [_parse_one(s, i, default_mode) for i, s in enumerate(raw_stages)]
    return [s for s in parsed if s is not None]


def normalize_rot_target(value: Any, *, idx: int) -> Optional[np.ndarray]:
    """Resolve a composer rot_target into a canonical (6,) ortho-6D row.

    Accepts:
      - callable: invoked with no args; result is fed back through this
        method so a callable returning a 3x3 (or any other shape below)
        works transparently.
      - (3,3) rotation matrix: Gram-Schmidt-orthonormalized via the
        existing ortho6d roundtrip, then sliced to 6D. Off-manifold inputs
        (LLM emits a non-orthonormal matrix) are silently corrected.
      - (6,) ortho-6D row: assumed valid; passed through.
      - (9,) flattened 3x3: reshaped, then treated as a 3x3.
      - (4,) wxyz quaternion: converted via quaternion_to_matrix.

    Returns None on failure (and logs a warning); the stage proceeds with
    rotation guidance disabled rather than crashing the rollout.
    """
    if value is None:
        return None
    try:
        if callable(value):
            value = value()
        arr = np.asarray(value, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Stage {idx}: rot_target eval failed: {e}")
        return None

    if arr.shape == (6,):
        return arr.astype(np.float32)
    if arr.shape == (9,):
        arr = arr.reshape(3, 3)
    if arr.shape == (4,):
        quat_t = torch.from_numpy(arr).float().unsqueeze(0)
        mat = quaternion_to_matrix(quat_t).squeeze(0).numpy()
        arr = mat.astype(np.float32)
    if arr.shape == (3, 3):
        mat_t = torch.from_numpy(arr).float().unsqueeze(0)
        ortho_t = compute_rotation_matrix_from_ortho6d(
            get_ortho6d_from_rotation_matrix(mat_t)
        )
        return (
            get_ortho6d_from_rotation_matrix(ortho_t)
            .squeeze(0)
            .numpy()
            .astype(np.float32)
        )
    logger.warning(
        f"Stage {idx}: rot_target has unexpected shape {arr.shape}; "
        f"expected (3,3), (6,), (9,), or (4,)"
    )
    return None
