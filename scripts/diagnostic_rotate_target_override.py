"""DIAGNOSTIC ONLY — not for the main repo.

Forces the rotation target on `rotate_*_*` CALVIN tasks to a known-correct
value, bypassing the composer's emitted `rot_target`. Pairs with Phase A
(setting `steering.guidance_strength_rot` to a non-zero value) to isolate
"composer picks the wrong rotation target" from "rotation steering math is
broken" — see `~/.claude/plans/there-is-an-issue-keen-floyd.md` for the
full plan.

What it overrides:
  When a stage with `primitive == 'rotate'` activates on a `rotate_*_left`
  or `rotate_*_right` task, we replace whatever 6D the composer emitted
  with `R_z(±90°) @ R_current_ee`, where R_current_ee is the EE rotation
  at the *moment of rotate-stage activation* (post-grasp). This is the
  semantically correct target the composer is supposed to emit but evaluates
  at composer-call time (start of episode) instead — see Finding 4 in the
  plan.

How it works:
  Monkey-patches `StageManager.setup_episode` to record `task_name`, then
  monkey-patches `StageManager._activate_stage` to override
  `_current_stage_target_rotation` after the original logic runs. No edits
  to any module under steering/, voxposer/, or policies/.

Usage:
  uv run python scripts/diagnostic_rotate_target_override.py \\
      evaluation=langsteer_primitive_object \\
      steering.guidance_strength_rot=1.0

  Restrict to the 6 rotate tasks via a Hydra task override or by editing
  langsteer_primitive_object.yaml temporarily.

To disable the override without removing the script, set the env var
LANGSTEER_DIAGNOSTIC_ROT_OVERRIDE=0 — useful for an A/B run from the same
script.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from policies.diffuser_actor_components.rotation_utils import (
    get_ortho6d_from_rotation_matrix,
)
from steering import stage_manager as _sm_module

logger = logging.getLogger("diagnostic_rot_override")


def _ee_rotation_matrix_from_robot_obs(robot_obs: np.ndarray) -> np.ndarray:
    """Same Euler XYZ → matrix convention as voxposer/calvin_interface.py:617-624."""
    rx, ry, rz = robot_obs[3:6]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _world_z_rotation_matrix(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _world_x_rotation_matrix(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _world_y_rotation_matrix(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _axis_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    axis = axis.lower()
    if axis == "x":
        return _world_x_rotation_matrix(angle_deg)
    if axis == "y":
        return _world_y_rotation_matrix(angle_deg)
    if axis == "z":
        return _world_z_rotation_matrix(angle_deg)
    raise ValueError(f"ROT_AXIS must be x|y|z, got {axis!r}")


def _compose_mode_left_or_right(R_delta: np.ndarray, R_ee: np.ndarray, mode: str) -> np.ndarray:
    """`mode` = 'world' → R_delta @ R_ee (world-frame rotation, what the
    composer prompt does today). `mode` = 'local' → R_ee @ R_delta (EE-local
    frame rotation, what you'd want if the EE axes are not aligned with world).
    """
    mode = mode.lower()
    if mode == "local":
        return R_ee @ R_delta
    return R_delta @ R_ee  # default: world


def _matrix_to_ortho6d(R: np.ndarray) -> np.ndarray:
    R_t = torch.from_numpy(R).float().unsqueeze(0)  # (1, 3, 3)
    return get_ortho6d_from_rotation_matrix(R_t).squeeze(0).numpy().astype(np.float32)


def _sign_from_task_name(task_name: str) -> int | None:
    """+1 for left, -1 for right, None when the task isn't a rotate."""
    if not task_name:
        return None
    if not task_name.startswith("rotate_"):
        return None
    if task_name.endswith("_left"):
        return +1
    if task_name.endswith("_right"):
        return -1
    return None


def _install_monkey_patches() -> None:
    if os.environ.get("LANGSTEER_DIAGNOSTIC_ROT_OVERRIDE", "1") == "0":
        logger.warning("LANGSTEER_DIAGNOSTIC_ROT_OVERRIDE=0 — patches NOT installed")
        return

    StageManager = _sm_module.StageManager
    orig_setup_episode = StageManager.setup_episode
    orig_activate_stage = StageManager._activate_stage

    def patched_setup_episode(self, task_name, *args, **kwargs):  # type: ignore[no-redef]
        self._diagnostic_task_name = task_name  # captured for _activate_stage
        return orig_setup_episode(self, task_name, *args, **kwargs)

    def patched_activate_stage(self, idx, is_refresh=False):  # type: ignore[no-redef]
        orig_activate_stage(self, idx, is_refresh=is_refresh)

        task_name = getattr(self, "_diagnostic_task_name", None)
        sign = _sign_from_task_name(task_name)
        if sign is None:
            return  # not a rotate_* task

        if idx >= len(self._stages):
            return
        spec = self._stages[idx]
        if spec.primitive != "rotate":
            return
        if self._robot_obs is None:
            logger.warning(
                "rotate stage activated but robot_obs is None — leaving "
                "composer's rot_target in place"
            )
            return

        R_ee = _ee_rotation_matrix_from_robot_obs(self._robot_obs)
        axis = os.environ.get("ROT_AXIS", "z")
        mode = os.environ.get("ROT_COMPOSE_MODE", "world")  # world | local
        R_delta = _axis_rotation_matrix(axis, 90.0 * sign)
        R_target = _compose_mode_left_or_right(R_delta, R_ee, mode)
        rot6d = _matrix_to_ortho6d(R_target)

        if spec.mode == "static":
            spec.cached_rotation = rot6d
        self._current_stage_target_rotation = (
            torch.from_numpy(rot6d).float().to(self._device)
        )

        logger.info(
            "[diagnostic_rot_override] task=%s stage=%d primitive=rotate → "
            "forced R_%s(%+d°) compose=%s @ R_current_ee (ee Euler XYZ=%s)",
            task_name,
            idx,
            axis,
            int(90 * sign),
            mode,
            np.round(self._robot_obs[3:6], 3).tolist(),
        )

    StageManager.setup_episode = patched_setup_episode  # type: ignore[assignment]
    StageManager._activate_stage = patched_activate_stage  # type: ignore[assignment]

    logger.warning(
        "[diagnostic_rot_override] StageManager patched — rotate_* tasks "
        "will use forced ±90° world-Z targets, NOT the composer's emission"
    )


_install_monkey_patches()


if __name__ == "__main__":
    # `scripts/` isn't a package (not in pyproject's `tool.setuptools.packages.find`),
    # so we can't `from scripts.run_evaluation import main`. Use runpy to exec
    # the sibling file as __main__ with our monkey-patches already installed.
    import runpy
    from pathlib import Path

    run_eval_path = str(Path(__file__).resolve().parent / "run_evaluation.py")
    runpy.run_path(run_eval_path, run_name="__main__")
