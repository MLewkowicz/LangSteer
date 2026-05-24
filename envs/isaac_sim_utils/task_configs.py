"""Task configuration for Isaac Sim Cabinet environment.

Tasks use objects spawned on the cabinet work surface alongside the
existing USD scene (Cortado robot + cabinet shelving).

Task config schema::

    {
        "instruction": str,
        "objects": [(name, position(3,), euler_deg(3,)), ...],
        "success": {
            "type": "object_lifted" | "object_at_target" | "object_in_container",
            ...
        },
    }

Object catalog schema::

    {
        "size": np.ndarray (3,),   # Bounding box dimensions in meters
        "color": tuple (3,),       # RGB [0, 1]
        "mass": float,             # kg (0.0 = static/fixed)
    }
"""

from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Object catalog — small manipulation objects that can be spawned on the
# cabinet work surface.  Sizes calibrated for a realistic desktop scene.
# ---------------------------------------------------------------------------
OBJECT_CATALOG: Dict[str, Dict[str, Any]] = {
    "red_block": {
        "size": np.array([0.04, 0.04, 0.04]),
        "color": (0.9, 0.1, 0.1),
        "mass": 0.1,
    },
    "blue_block": {
        "size": np.array([0.04, 0.04, 0.04]),
        "color": (0.1, 0.2, 0.9),
        "mass": 0.1,
    },
    "green_block": {
        "size": np.array([0.04, 0.04, 0.04]),
        "color": (0.1, 0.8, 0.1),
        "mass": 0.1,
    },
    "small_bowl": {
        "size": np.array([0.12, 0.12, 0.05]),
        "color": (0.85, 0.85, 0.75),
        "mass": 0.2,
    },
    "target_zone": {
        "size": np.array([0.12, 0.12, 0.002]),
        "color": (0.9, 0.8, 0.1),
        "mass": 0.0,  # Fixed marker
    },
}

# ---------------------------------------------------------------------------
# Task definitions
# Work surface of the Cabinet scene is approximately at Z=0.76 m (world frame).
# Robot base is at the cabinet edge; work area spans roughly x=[0.3, 0.7].
# ---------------------------------------------------------------------------

# Table surface height in the Cabinet scene (adjust after physical calibration)
_TABLE_Z = 0.76
_WORK_X = 0.50   # center of reachable workspace
_WORK_Y_L = -0.15  # left side
_WORK_Y_R = 0.15   # right side

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "pick_up_red_block": {
        "instruction": "pick up the red block",
        "objects": [
            ("red_block", np.array([_WORK_X, _WORK_Y_L, _TABLE_Z + 0.02]), np.zeros(3)),
        ],
        "success": {
            "type": "object_lifted",
            "object": "red_block",
            "min_height": _TABLE_Z + 0.10,
        },
    },
    "move_block_to_target": {
        "instruction": "place the blue block on the yellow target",
        "objects": [
            ("blue_block", np.array([_WORK_X, _WORK_Y_L, _TABLE_Z + 0.02]), np.zeros(3)),
            ("target_zone", np.array([_WORK_X, _WORK_Y_R, _TABLE_Z + 0.001]), np.zeros(3)),
        ],
        "success": {
            "type": "object_at_target",
            "object": "blue_block",
            "target": "target_zone",
            "threshold": 0.06,
        },
    },
    "stack_blocks": {
        "instruction": "stack the green block on top of the red block",
        "objects": [
            ("red_block", np.array([_WORK_X, _WORK_Y_L, _TABLE_Z + 0.02]), np.zeros(3)),
            ("green_block", np.array([_WORK_X, _WORK_Y_R, _TABLE_Z + 0.02]), np.zeros(3)),
        ],
        "success": {
            "type": "object_at_target",
            "object": "green_block",
            "target": "red_block",
            "height_offset": 0.04,
            "threshold": 0.04,
        },
    },
    "put_block_in_bowl": {
        "instruction": "put the red block in the bowl",
        "objects": [
            ("red_block", np.array([_WORK_X, _WORK_Y_L, _TABLE_Z + 0.02]), np.zeros(3)),
            ("small_bowl", np.array([_WORK_X, _WORK_Y_R, _TABLE_Z + 0.025]), np.zeros(3)),
        ],
        "success": {
            "type": "object_in_container",
            "object": "red_block",
            "container": "small_bowl",
            "height_margin": 0.03,
        },
    },
}


def register_object(
    name: str, size: np.ndarray, color: tuple, mass: float = 0.05
) -> None:
    OBJECT_CATALOG[name] = {
        "size": np.asarray(size),
        "color": color,
        "mass": mass,
    }


def register_task(name: str, config: Dict[str, Any]) -> None:
    TASK_CONFIGS[name] = config


def get_task_config(task_name: str) -> Dict[str, Any]:
    if task_name not in TASK_CONFIGS:
        available = ", ".join(sorted(TASK_CONFIGS.keys())) or "(none registered)"
        raise ValueError(
            f"Unknown Isaac Sim task '{task_name}'. Available: {available}"
        )
    return TASK_CONFIGS[task_name]


def get_task_instruction(task_name: str) -> str:
    return get_task_config(task_name)["instruction"]


def get_all_task_names() -> List[str]:
    return sorted(TASK_CONFIGS.keys())
