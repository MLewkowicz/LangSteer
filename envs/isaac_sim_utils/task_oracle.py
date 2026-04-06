"""Task success detection for Isaac Sim environment.

Checks success conditions using ground-truth object poses from the
simulator scene graph.  Mirrors CALVIN's Tasks oracle but uses
Isaac Sim's USD scene API instead of PyBullet link states.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class IsaacSimTaskOracle:
    """Evaluate task success from ground-truth sim state.

    Each task's ``success`` config specifies a check type:
      - ``object_lifted``: object's Z position exceeds ``min_height``
      - ``object_at_target``: object is within ``threshold`` of target position
      - ``object_in_container``: object is inside container AABB (XY) and above
        container bottom (Z with margin)
    """

    def __init__(self) -> None:
        self._checkers = {
            "object_lifted": self._check_lifted,
            "object_at_target": self._check_at_target,
            "object_in_container": self._check_in_container,
        }

    def check_success(
        self,
        success_cfg: Dict[str, Any],
        object_poses: Dict[str, np.ndarray],
    ) -> bool:
        """Check whether the task success condition is met.

        Args:
            success_cfg: The ``success`` dict from the task config.
            object_poses: Mapping of object name → (3,) world position,
                populated from the scene graph each step.

        Returns:
            True if the success condition is satisfied.
        """
        check_type = success_cfg.get("type")
        checker = self._checkers.get(check_type)
        if checker is None:
            logger.warning(f"Unknown success check type: {check_type}")
            return False
        return checker(success_cfg, object_poses)

    # ------------------------------------------------------------------
    # Individual checkers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_lifted(cfg: Dict, poses: Dict[str, np.ndarray]) -> bool:
        obj = cfg["object"]
        if obj not in poses:
            return False
        return float(poses[obj][2]) >= cfg["min_height"]

    @staticmethod
    def _check_at_target(cfg: Dict, poses: Dict[str, np.ndarray]) -> bool:
        obj = cfg["object"]
        target = cfg["target"]
        if obj not in poses or target not in poses:
            return False

        obj_pos = poses[obj].copy()
        target_pos = poses[target].copy()

        # Optional height offset (e.g. stacking)
        height_offset = cfg.get("height_offset", 0.0)
        if height_offset > 0:
            target_pos[2] += height_offset

        dist = np.linalg.norm(obj_pos - target_pos)
        return float(dist) < cfg["threshold"]

    @staticmethod
    def _check_in_container(cfg: Dict, poses: Dict[str, np.ndarray]) -> bool:
        obj = cfg["object"]
        container = cfg["container"]
        if obj not in poses or container not in poses:
            return False

        obj_pos = poses[obj]
        container_pos = poses[container]
        margin = cfg.get("height_margin", 0.03)

        # Check XY: object center within ±6 cm of container center
        xy_dist = np.linalg.norm(obj_pos[:2] - container_pos[:2])
        # Check Z: object above container bottom
        z_ok = obj_pos[2] > container_pos[2] - margin

        return bool(xy_dist < 0.06 and z_ok)
