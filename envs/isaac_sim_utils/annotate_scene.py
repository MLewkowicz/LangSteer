"""Isaac Sim scene data capture and OBB computation for bbox annotation.

Replaces tmp/bboxes.py for the Isaac Sim workflow.  Provides:
  - SceneSnapshot: pure-numpy store of USD-sourced prim poses/AABBs
  - capture_snapshot(): queries the live Isaac Sim stage once
  - compute_bboxes(): pure-numpy OBB math, no sim calls

The annotator session operates after env.reset() — objects are at rest, so
we capture all USD-sourced state once at startup and compute_bboxes() becomes
a pure-numpy function driven by the mutable override dicts.

OBB format::

    {
        'center': (3,) np.float32,
        'size':   (3,) np.float32,
        'rotation': (3, 3) np.float32,   # identity = axis-aligned
        'editable': bool,
        'category': 'fixture' | 'derived' | 'readonly',
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Initial override values — edit these to match the Cabinet scene geometry
# after calibration with the annotator GUI.
# ---------------------------------------------------------------------------

INITIAL_OVERRIDES: Dict[str, Dict[str, list]] = {
    "drawer": {
        "rest_position": [0.55, 0.0, 0.65],
        "size": [0.30, 0.25, 0.08],
        "euler_xyz_deg": [0.0, 0.0, 0.0],
    },
    "drawer_handle": {
        "rest_position": [0.55, 0.0, 0.70],
        "size": [0.20, 0.04, 0.03],
        "euler_xyz_deg": [0.0, 0.0, 0.0],
    },
    "shelf_upper": {
        "rest_position": [0.55, 0.0, 0.90],
        "size": [0.40, 0.35, 0.02],
        "euler_xyz_deg": [0.0, 0.0, 0.0],
    },
    "shelf_lower": {
        "rest_position": [0.55, 0.0, 0.76],
        "size": [0.40, 0.35, 0.02],
        "euler_xyz_deg": [0.0, 0.0, 0.0],
    },
}

# Derived fixtures defined as (offset, size) in the parent's local frame.
INITIAL_DERIVED: Dict[str, Dict[str, Any]] = {}

# Colors for the annotation GUI overlay
COLORS: Dict[str, str] = {
    "drawer": "#d62728",
    "drawer_handle": "#ff9896",
    "shelf_upper": "#1f77b4",
    "shelf_lower": "#aec7e8",
}


def _euler_to_matrix(euler_xyz_deg) -> np.ndarray:
    return Rotation.from_euler("xyz", euler_xyz_deg, degrees=True).as_matrix().astype(np.float32)


# ---------------------------------------------------------------------------
# Scene snapshot
# ---------------------------------------------------------------------------

@dataclass
class SceneSnapshot:
    """USD-sourced data captured once at env.reset()."""

    # World-frame position of each fixture prim (3,) per entry
    prim_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    # World-frame AABB for non-editable prims: (min_xyz, max_xyz)
    readonly_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # Spawned object OBBs from scene: {name: (center, size, R)}
    object_obbs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )


def capture_snapshot(scene) -> SceneSnapshot:
    """Query the live Isaac Sim scene once to populate a SceneSnapshot.

    Args:
        scene: IsaacSimScene instance after reset().

    Returns:
        SceneSnapshot with prim positions and object bounding boxes.
    """
    from isaacsim.core.prims import SingleXFormPrim
    from pxr import Usd, UsdGeom

    robot_prim_path = scene._cfg.get("robot_prim_path", "/World/franka")
    snap = SceneSnapshot()

    # Capture editable fixture positions from the USD stage
    stage = scene._world.stage
    for name, spec in INITIAL_OVERRIDES.items():
        # Try to find the prim by approximate path; use initial position as fallback
        # The exact path depends on the Cabinet USD hierarchy
        prim_path = f"/World/Cabinet/{name}"
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            xform = SingleXFormPrim(prim_path=prim_path)
            pos, _ = xform.get_world_pose()
            snap.prim_positions[name] = np.array(pos, dtype=np.float32)
        else:
            # Fall back to the hardcoded rest position
            snap.prim_positions[name] = np.array(
                spec["rest_position"], dtype=np.float32
            )

    # Capture spawned manipulation object OBBs
    for obj_name, obj in scene._objects.items():
        pos, quat_wxyz = obj.get_world_pose()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        R = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        from envs.isaac_sim_utils.task_configs import OBJECT_CATALOG
        size = np.asarray(OBJECT_CATALOG.get(obj_name, {}).get("size", [0.04, 0.04, 0.04]),
                          dtype=np.float32)
        snap.object_obbs[obj_name] = (
            np.array(pos, dtype=np.float32),
            size,
            R,
        )

    return snap


# ---------------------------------------------------------------------------
# Pure-numpy OBB computation (no sim calls)
# ---------------------------------------------------------------------------

def compute_bboxes(
    snapshot: SceneSnapshot,
    overrides: Dict[str, Dict[str, list]],
    derived: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return per-object OBBs keyed by name (pure numpy, no Isaac Sim calls).

    Each entry::

        {
            'center': (3,) float32,
            'size':   (3,) float32,
            'rotation': (3, 3) float32,
            'editable': bool,
            'category': 'fixture' | 'derived' | 'readonly',
        }
    """
    out: Dict[str, Dict[str, Any]] = {}
    fixture_centers: Dict[str, np.ndarray] = {}
    fixture_sizes: Dict[str, np.ndarray] = {}
    fixture_rotations: Dict[str, np.ndarray] = {}

    # --- editable fixtures (position taken from snapshot or override) ---
    for name, spec in overrides.items():
        if name not in snapshot.prim_positions and name not in INITIAL_OVERRIDES:
            continue
        rest_pos = np.asarray(spec["rest_position"], dtype=np.float32)
        size = np.asarray(spec["size"], dtype=np.float32)
        R = _euler_to_matrix(spec.get("euler_xyz_deg", [0.0, 0.0, 0.0]))
        fixture_centers[name] = rest_pos
        fixture_sizes[name] = size
        fixture_rotations[name] = R
        out[name] = {
            "center": rest_pos,
            "size": size,
            "rotation": R,
            "editable": True,
            "category": "fixture",
        }

    # --- read-only prims ---
    for name, (mn, mx) in snapshot.readonly_aabbs.items():
        center = (mn + mx) / 2.0
        size = mx - mn
        R = np.eye(3, dtype=np.float32)
        fixture_centers[name] = center
        fixture_sizes[name] = size
        fixture_rotations[name] = R
        out[name] = {
            "center": center,
            "size": size,
            "rotation": R,
            "editable": False,
            "category": "readonly",
        }

    # --- derived fixtures (offset in parent local frame) ---
    for name, spec in derived.items():
        parent = spec.get("parent")
        if parent not in fixture_centers:
            continue
        local_offset = np.asarray(spec["offset"], dtype=np.float32)
        size_spec = spec.get("size")
        size = (
            np.asarray(size_spec, dtype=np.float32)
            if size_spec is not None
            else fixture_sizes[parent].copy()
        )
        R_local = _euler_to_matrix(spec.get("euler_xyz_deg", [0.0, 0.0, 0.0]))
        R_parent = fixture_rotations[parent]
        center = fixture_centers[parent] + R_parent @ local_offset
        R = R_parent @ R_local
        out[name] = {
            "center": center.astype(np.float32),
            "size": size,
            "rotation": R.astype(np.float32),
            "editable": True,
            "category": "derived",
        }

    # --- spawned manipulation objects ---
    for name, (center, size, R) in snapshot.object_obbs.items():
        out[name] = {
            "center": center,
            "size": size,
            "rotation": R,
            "editable": False,
            "category": "readonly",
        }

    return out
