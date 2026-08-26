"""DIAGNOSTIC ONLY — not for the main repo.

Tests the mode-commitment hypothesis: if the (primitive=rotate, object=X)
policy is direction-blind, it should commit to a single rotation direction
regardless of whether the task is `rotate_*_left` or `rotate_*_right`.

What it does:
  Monkey-patches `DiffuserActorBasePolicy._convert_action` so that each policy
  call appends `(task_name, episode_step, ee_euler_at_first_waypoint)` to
  `/tmp/rot_pred_log.jsonl`. The first waypoint is the most-imminent action
  the policy is about to execute — its euler-Z (yaw) tells us the rotation
  direction the policy committed to.

Usage:
  uv run python scripts/diagnostic_log_rot_predictions.py \\
      --evaluation _rot_diag_off \\
      --num-episodes 3 --max-steps 360 --tries-per-episode 1 \\
      --seed 42 --output-dir /tmp/rot_diag_out_log

Analyze afterwards by tailing /tmp/rot_pred_log.jsonl and computing the
per-episode yaw drift `euler_z[-1] - euler_z[0]`. If LEFT and RIGHT tasks
show the same sign of drift → mode commitment confirmed.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from policies.diffuser_actor_base import DiffuserActorBasePolicy

logger = logging.getLogger("diagnostic_rot_pred")

LOG_PATH = os.environ.get("ROT_PRED_LOG", "/tmp/rot_pred_log.jsonl")


def _install_monkey_patch() -> None:
    orig_convert_action = DiffuserActorBasePolicy._convert_action
    # The base policy doesn't know the task name; we sniff it from the env at
    # patch-call time via a side channel. The runner sets env._task_name when
    # it calls env.set_task(...), so we attach the env to the policy lazily.
    # Easier alternative: write everything keyed only by the rollout's
    # episode_step + the policy's call count; the user reads
    # /tmp/rot_pred_log.jsonl alongside the eval's stdout to correlate.

    call_counter = {"n": 0}

    def patched_convert_action(self, trajectory):
        action_np = orig_convert_action(self, trajectory)
        # action_np shape: (L, 7) = [pos(3), euler_xyz(3), gripper(1)]
        try:
            waypoint0 = action_np[0] if action_np.ndim == 2 else action_np[0, 0]
            ee_euler = waypoint0[3:6].tolist()
            ee_pos = waypoint0[:3].tolist()
            entry = {
                "call": call_counter["n"],
                "pos0": ee_pos,
                "euler0": ee_euler,
            }
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("rot-pred log failed: %s", e)
        call_counter["n"] += 1
        return action_np

    DiffuserActorBasePolicy._convert_action = patched_convert_action
    logger.warning(
        "[diagnostic_rot_pred] DiffuserActorBasePolicy._convert_action patched; "
        "logging to %s",
        LOG_PATH,
    )


# Truncate the log file at startup so each run starts clean.
if os.environ.get("ROT_PRED_NO_TRUNCATE", "0") != "1":
    try:
        open(LOG_PATH, "w").close()
    except Exception:
        pass

_install_monkey_patch()


if __name__ == "__main__":
    import runpy
    from pathlib import Path

    run_eval_path = str(Path(__file__).resolve().parent / "run_evaluation.py")
    runpy.run_path(run_eval_path, run_name="__main__")
