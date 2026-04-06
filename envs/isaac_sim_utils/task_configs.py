"""Task configuration for Isaac Sim tabletop environment.

Defines manipulation tasks with initial object placements, language
instructions, and success criteria.  Each task is a dict consumed by
IsaacSimEnvironment.reset() to spawn / position USD prims and by
IsaacSimTaskOracle to check completion.

Task config schema::

    {
        "instruction": str,         # Natural-language task description
        "objects": [                # Objects to spawn
            (name, position(3,), euler_deg(3,)),
            ...
        ],
        "success": {                # Success criteria for IsaacSimTaskOracle
            "type": "object_lifted" | "object_at_target" | "object_in_container",
            ...                     # type-specific fields (see task_oracle.py)
        },
    }

Object catalog schema::

    {
        "usd_path": str,            # Asset path relative to asset root
        "size": np.ndarray (3,),    # Bounding box dimensions in meters
        "color": tuple (3,),        # RGB [0,1]
        "mass": float,              # kg (0.0 = static/fixed)
    }
"""

from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Object catalog — USD asset paths relative to a configurable asset root.
# Add new objects here; the scene loader resolves full paths at runtime.
# ---------------------------------------------------------------------------
OBJECT_CATALOG: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Task definitions — populated as tasks are designed.
# ---------------------------------------------------------------------------
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {}


def register_object(name: str, usd_path: str, size: np.ndarray, color: tuple, mass: float = 0.05) -> None:
    """Add an object to the catalog at runtime."""
    OBJECT_CATALOG[name] = {
        "usd_path": usd_path,
        "size": np.asarray(size),
        "color": color,
        "mass": mass,
    }


def register_task(name: str, config: Dict[str, Any]) -> None:
    """Add a task definition at runtime."""
    TASK_CONFIGS[name] = config


def get_task_config(task_name: str) -> Dict[str, Any]:
    """Return the full task config dict, raising if not found."""
    if task_name not in TASK_CONFIGS:
        available = ", ".join(sorted(TASK_CONFIGS.keys())) or "(none registered)"
        raise ValueError(f"Unknown Isaac Sim task '{task_name}'. Available: {available}")
    return TASK_CONFIGS[task_name]


def get_task_instruction(task_name: str) -> str:
    """Return the language instruction for a task."""
    return get_task_config(task_name)["instruction"]


def get_all_task_names() -> List[str]:
    """Return sorted list of available task names."""
    return sorted(TASK_CONFIGS.keys())
