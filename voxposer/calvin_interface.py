"""CALVIN-specific LMP interface for VoxPoser value map generation.

Replaces VoxPoser's RLBench-based LMP_interface with CALVIN-specific
object detection using scene_obs ground-truth positions and hardcoded
fixture locations.
"""

import logging
from typing import Optional

import numpy as np

from voxposer.utils import (
    Observation,
    VoxelIndexingWrapper,
    normalize_vector,
)

logger = logging.getLogger(__name__)


class ObjectResolutionError(ValueError):
    """Raised when `detect()` cannot resolve an object name to any known fixture/block.

    Analog of `voxposer.lmp.VocabValidationError`: subclass of `ValueError` so
    existing `except ValueError` clauses still catch it; `StageManager.setup_episode`
    narrows the catch so it propagates to the runner (no silent fallback to
    workspace-center, which previously misled the policy into chasing the
    middle of empty space).
    """


# Grasp-closure threshold (m). Below this the gripper is "closed-ish" enough
# that we'll consult the held-block heuristic for generic 'block' queries.
# Matches the production grasp_max_width default in conf/steering/voxposer.yaml.
HELD_BLOCK_GRIPPER_CLOSED_MAX_WIDTH = 0.07
# A block is considered "held" when its centroid is within this many meters of
# the gripper position. 3cm captures the held-block-jitter envelope without
# false-positives on transport-context tasks (block traveling near but not in
# the gripper).
HELD_BLOCK_PROXIMITY_RADIUS = 0.03

# Workspace bounds in absolute world coordinates (meters).
# Covers all CALVIN objects: blocks, slider, drawer, lights, table surface.
DEFAULT_WORKSPACE_MIN = np.array([-0.35, -0.60, 0.30])
DEFAULT_WORKSPACE_MAX = np.array([0.35, 0.15, 0.85])

# Aliases for end-effector
EE_ALIAS = [
    "ee",
    "endeffector",
    "end_effector",
    "end effector",
    "gripper",
    "hand",
]

# Aliases for table/workspace
TABLE_ALIAS = [
    "table",
    "desk",
    "workstation",
    "work_station",
    "work station",
    "workspace",
    "work_space",
    "work space",
]

# CALVIN fixed fixture positions.
# Fallback only — when live PyBullet data isn't available, `_detect_fixture`
# reads from here and assumes identity rotation. With live data, the
# authoritative values (and orientations) come from
# CalvinEnvironment._get_fixture_positions().
#
# Values mirror envs/calvin.py's _FIXTURE_AABB_OVERRIDES + _DERIVED_OFFSETS
# (the handle/light_switch positions are pre-composed here since this dict
# doesn't carry rotation).
# Link mapping (playtable UID=5): 0=button, 1=switch, 2=slide, 3=drawer, 4=led, 5=light
CALVIN_FIXTURES = {
    # NOTE (2026-05-19, user directive): bare `slider` (door panel) and `drawer`
    # (body) entries removed from this voxposer-side dict. They caused value
    # maps to collapse on the door / drawer face instead of the intended
    # target (handle for articulation, *_interior for placement). The
    # composer / affordance LMP code uses only `*_handle` (contact) and
    # `*_interior` (cavity destination) — bare names are never the target
    # of a value map. Env-side `_FIXTURE_AABB_OVERRIDES` and
    # `_DERIVED_OFFSETS` still keep `slider` and `drawer` as parent
    # references for their derived children — only the name-resolution
    # dict here drops them.
    "slider_handle": {
        # Small grasp groove on the slider door's front face (parent-local offset [0, -0.038, 0.002])
        "position": np.array([0.040, 0.027, 0.540]),
        "size": np.array([0.034, 0.066, 0.161]),
    },
    "drawer_handle": {
        # Horizontal pull bar flush with the drawer's front face
        "position": np.array([0.180, -0.220, 0.360]),
        "size": np.array([0.242, 0.077, 0.036]),
    },
    "drawer_interior": {
        # Cavity inside the drawer body. Static placeholder; live tracked
        # values come from `envs/calvin.py::_DERIVED_OFFSETS['drawer_interior']`
        # via `_fixture_positions` (parent='drawer' so it follows the drawer
        # joint). `_detect_fixture` prefers live values over this stub.
        "position": np.array([0.178, -0.008, 0.354]),
        "size": np.array([0.40, 0.30, 0.08]),
    },
    "slider_interior": {
        # Cavity inside the slider cabinet body. Static placeholder; live
        # tracked values come from `envs/calvin.py::_FIXTURE_AABB_OVERRIDES['slider_interior']`
        # via `_fixture_positions`. World-frame static (cabinet doesn't move);
        # spans BOTH halves — `_detect_fixture` slices to the accessible half
        # using `_scene_context['slider_accessible_chamber']` when available.
        "position": np.array([-0.0930, 0.0980, 0.4530]),
        "size": np.array([0.5450, 0.1460, 0.0250]),
    },
    "lightbulb": {
        # light_link (Link 5)
        "position": np.array([0.300, 0.160, 0.673]),
        "size": np.array([0.062, 0.062, 0.056]),
    },
    "switch": {
        # switch_link (Link 1) — housing cube for the light switch mechanism.
        "position": np.array([0.296, 0.039, 0.499]),
        "size": np.array([0.06, 0.06, 0.06]),
    },
    "light_switch": {
        # Same link 1 as 'switch', but describes the tilted toggle lever (rotated -31.5° around X).
        "position": np.array([0.302, 0.037, 0.518]),
        "size": np.array([0.118, 0.061, 0.031]),
    },
    "led": {
        # led_link (Link 4)
        "position": np.array([-0.120, 0.160, 0.656]),
        "size": np.array([0.06, 0.046, 0.022]),
    },
    "button": {
        # button_link (Link 0) — controls the LED
        "position": np.array([-0.120, -0.120, 0.472]),
        "size": np.array([0.07, 0.07, 0.03]),
    },
}

# Block size in meters (from block_red.urdf: <box size="0.05 0.05 0.05"/>)
BLOCK_SIZE = np.array([0.05, 0.05, 0.05])

# scene_obs slices for block positions
BLOCK_SCENE_OBS = {
    "red_block": slice(6, 9),
    "red block": slice(6, 9),
    "blue_block": slice(12, 15),
    "blue block": slice(12, 15),
    "pink_block": slice(18, 21),
    "pink block": slice(18, 21),
}

# scene_obs layout for CALVIN (24-dim, deterministic state). Used by
# `format_scene_state` to emit a text block injected into composer +
# affordance LMP prompts at episode start (Task 7 Phase 2 expansion).
#
#   [0]      slider joint position (radians; -ve = left, ~0.14 = center, ~0.28 = right)
#   [1]      drawer joint position (radians; ~0 = closed, > 0.04 = open)
#   [2]      button state           (~0 / ~1, toggles led)
#   [3]      switch state           (~0 / ~1, toggles lightbulb)
#   [4]      led state              (0 = off, 1 = on)
#   [5]      lightbulb state        (0 = off, 1 = on)
#   [6:9]    red_block position     (world xyz)
#   [9:12]   red_block euler
#   [12:15]  blue_block position
#   [15:18]  blue_block euler
#   [18:21]  pink_block position
#   [21:24]  pink_block euler
SCENE_OBS_SLIDER = 0
SCENE_OBS_DRAWER = 1
SCENE_OBS_LED = 4
SCENE_OBS_LIGHTBULB = 5

# Joint-position thresholds (radians). Calibrated against CALVIN's
# TASK_INITIAL_CONDITIONS (slider 'left'/'right' → 0.0/0.28, drawer
# 'open' → 0.16, 'closed' → 0.0).
_DRAWER_OPEN_THRESH = 0.04
_SLIDER_CENTER_LO = 0.08
_SLIDER_CENTER_HI = 0.20


def format_scene_state(
    scene_obs: np.ndarray,
    block_aabbs: Optional[dict] = None,
) -> str:
    """Pretty-print deterministic CALVIN scene state as a prompt comment block.

    Task 7 Phase 2 expansion — the composer + affordance LMPs read this
    alongside the VLM grounding dict. State info comes from `scene_obs`
    (ground truth) so the VLM doesn't have to read drawer/slider/light
    states from the image (Phase 0 audit confirmed VLM is bad at those).

    Args:
        scene_obs: (24,) array per CALVIN's spec.
        block_aabbs: optional live PyBullet block AABB dict from
            `env._get_block_aabbs()`. When provided, block-location buckets
            are derived from live z + xy ranges rather than hardcoded
            thresholds.

    Returns:
        A `# Current scene state ...` comment block (multi-line string).
        Empty string when scene_obs is None.
    """
    if scene_obs is None:
        return ""
    so = np.asarray(scene_obs, dtype=np.float32)

    # Fixture states.
    drawer = "open" if so[SCENE_OBS_DRAWER] > _DRAWER_OPEN_THRESH else "closed"
    slider_v = so[SCENE_OBS_SLIDER]
    if slider_v < _SLIDER_CENTER_LO:
        slider = "left"
    elif slider_v > _SLIDER_CENTER_HI:
        slider = "right"
    else:
        slider = "center"
    led = "on" if so[SCENE_OBS_LED] > 0.5 else "off"
    lightbulb = "on" if so[SCENE_OBS_LIGHTBULB] > 0.5 else "off"

    # Block locations — z-based bucketing as a fallback when AABBs aren't
    # available. Table surface is z ≈ 0.46; drawer interior z ≈ 0.36;
    # slider interior z ≈ 0.55; held blocks are typically z > 0.55.
    block_locations: dict[str, str] = {}
    for color, slc in [
        ("red_block", slice(6, 9)),
        ("blue_block", slice(12, 15)),
        ("pink_block", slice(18, 21)),
    ]:
        pos = so[slc]
        y, z = float(pos[1]), float(pos[2])
        if z < 0.42:
            block_locations[color] = "drawer_inside"
        elif y > 0.05 and z > 0.5:
            block_locations[color] = "slider_inside"
        else:
            block_locations[color] = "table"

    lines = ["# Current scene state (deterministic, from CALVIN scene_obs):"]
    lines.append(f"#   drawer: {drawer}")
    lines.append(f"#   slider: {slider}")
    lines.append(f"#   lightbulb: {lightbulb}")
    lines.append(f"#   led: {led}")
    lines.append("#   block locations:")
    for k, v in block_locations.items():
        lines.append(f"#     {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def pc2voxel(pc, bounds_min, bounds_max, map_size):
    """Convert world-frame point(s) to voxel coordinates."""
    pc = np.asarray(pc, dtype=np.float32)
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    pc = np.clip(pc, bounds_min, bounds_max)
    voxels = (pc - bounds_min) / (bounds_max - bounds_min) * (map_size - 1)
    _out = np.empty_like(voxels)
    voxels = np.round(voxels, 0, _out).astype(np.int32)
    return voxels


def voxel2pc(voxels, bounds_min, bounds_max, map_size):
    """Convert voxel coordinates to world-frame point(s)."""
    voxels = np.asarray(voxels, dtype=np.float32)
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    pc = voxels / (map_size - 1) * (bounds_max - bounds_min) + bounds_min
    return pc


# 8 unit-cube corners (each coord in {-0.5, +0.5}), used to envelope an OBB.
_UNIT_CUBE_CORNERS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [-0.5, +0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [-0.5, +0.5, +0.5],
        [+0.5, +0.5, +0.5],
    ],
    dtype=np.float32,
)


def obb_world_corners(
    center: np.ndarray, size: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """8 world-frame corners of an OBB. `rotation` is world-frame (3, 3)."""
    local = _UNIT_CUBE_CORNERS * np.asarray(size, dtype=np.float32)  # (8, 3)
    return (np.asarray(rotation, dtype=np.float32) @ local.T).T + np.asarray(
        center, dtype=np.float32
    )


def pc2voxel_map(points, bounds_min, bounds_max, map_size):
    """Convert point cloud to 3D occupancy voxel grid."""
    points = np.asarray(points, dtype=np.float32)
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    points = np.clip(points, bounds_min, bounds_max)
    voxel_xyz = (points - bounds_min) / (bounds_max - bounds_min) * (map_size - 1)
    _out = np.empty_like(voxel_xyz)
    points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32)
    voxel_map = np.zeros((map_size, map_size, map_size))
    for i in range(points_vox.shape[0]):
        voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
    return voxel_map


class CalvinLMPInterface:
    """CALVIN-specific interface providing helper functions for LLM-generated code.

    Exposes the same API as VoxPoser's LMP_interface (detect, cm2index,
    set_voxel_by_radius, get_empty_*_map) but uses CALVIN's scene_obs for
    object detection instead of RLBench's per-object point clouds.
    """

    def __init__(self, config: dict):
        self._map_size = config.get("map_size", 100)
        self._workspace_min = np.array(
            config.get("workspace_bounds_min", DEFAULT_WORKSPACE_MIN),
            dtype=np.float32,
        )
        self._workspace_max = np.array(
            config.get("workspace_bounds_max", DEFAULT_WORKSPACE_MAX),
            dtype=np.float32,
        )

        # Voxel resolution (meters per voxel)
        self._resolution = (self._workspace_max - self._workspace_min) / self._map_size

        # Current state (updated each episode/step)
        self._robot_obs: Optional[np.ndarray] = None  # (15,)
        self._scene_obs: Optional[np.ndarray] = None  # (24,)
        self._fixture_positions: Optional[dict] = None  # live PyBullet positions
        self._block_aabbs: Optional[dict] = None  # live PyBullet block AABBs
        # Task 7 Phase 2 — VLM grounding dict (blocks_visible +
        # ambiguous_resolutions), set per-episode by StageManager when
        # `scene_grounding.enabled` is true. Read by downstream LMPs.
        self._scene_context: Optional[dict] = None

        logger.info(
            f"CalvinLMPInterface: map_size={self._map_size}, "
            f"resolution={np.round(self._resolution * 100, 1)} cm/voxel, "
            f"workspace=[{self._workspace_min}, {self._workspace_max}]"
        )

    @property
    def workspace_bounds_min(self):
        return self._workspace_min

    @property
    def workspace_bounds_max(self):
        return self._workspace_max

    def update_state(
        self,
        robot_obs: np.ndarray,
        scene_obs: np.ndarray,
        fixture_positions: Optional[dict] = None,
        block_aabbs: Optional[dict] = None,
    ):
        """Update current robot and scene state for object detection.

        Args:
            robot_obs: (15,) robot proprioception
            scene_obs: (24,) scene state (block positions, joint states)
            fixture_positions: Optional dict from CalvinEnvironment._get_fixture_positions()
                mapping fixture name → {'position': (3,), 'size': (3,)}.
                When provided, _detect_fixture uses live positions instead of
                hardcoded CALVIN_FIXTURES.
            block_aabbs: Optional dict from CalvinEnvironment._get_block_aabbs()
                mapping 'red_block'/'blue_block'/'pink_block' → {'aabb_min',
                'aabb_max', 'position'}. Gives orientation-aware bounding boxes
                that reflect the block's current pose.
        """
        self._robot_obs = np.asarray(robot_obs, dtype=np.float32)
        self._scene_obs = np.asarray(scene_obs, dtype=np.float32)
        if fixture_positions is not None:
            self._fixture_positions = fixture_positions
        if block_aabbs is not None:
            self._block_aabbs = block_aabbs

    def set_scene_context(self, grounding: Optional[dict]) -> None:
        """Stash the VLM grounding dict for downstream LMPs to read.

        Task 7 Phase 2 — `grounding` has the narrowed schema
        `{'blocks_visible': {...}, 'ambiguous_resolutions': {...}}` (no
        fixtures_state). Affordance LMPs read this directly to decide
        cavity vs surface approach. Pass `None` to clear (e.g. between
        episodes).
        """
        self._scene_context = grounding

    # ==========================================================
    # Functions exposed to LLM-generated code
    # ==========================================================

    def detect(self, obj_name: str) -> Observation:
        """Detect an object and return its observation dict.

        Supports:
        - EE aliases: 'gripper', 'ee', 'hand', etc.
        - Table aliases: 'table', 'workspace', etc.
        - Blocks: 'red block', 'blue block', 'pink block' (from scene_obs)
        - Fixtures: 'drawer', 'slider', 'lightbulb', 'led', 'button', 'switch'
        """
        name_lower = obj_name.lower().strip()

        if name_lower in EE_ALIAS:
            return self._detect_ee(obj_name)
        elif name_lower in TABLE_ALIAS:
            return self._detect_table(obj_name)
        else:
            return self._detect_object(obj_name, name_lower)

    def cm2index(self, cm, direction):
        """Convert centimeters to voxel grid index offset."""
        if isinstance(direction, str) and direction == "x":
            return int(cm / (self._resolution[0] * 100))
        elif isinstance(direction, str) and direction == "y":
            return int(cm / (self._resolution[1] * 100))
        elif isinstance(direction, str) and direction == "z":
            return int(cm / (self._resolution[2] * 100))
        else:
            assert isinstance(direction, np.ndarray) and direction.shape == (3,)
            direction = normalize_vector(direction)
            x_index = self.cm2index(cm * direction[0], "x")
            y_index = self.cm2index(cm * direction[1], "y")
            z_index = self.cm2index(cm * direction[2], "z")
            return np.array([x_index, y_index, z_index])

    def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
        """Set voxels within radius_cm of position to value."""
        voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
        if radius_cm > 0:
            radius_x = self.cm2index(radius_cm, "x")
            radius_y = self.cm2index(radius_cm, "y")
            radius_z = self.cm2index(radius_cm, "z")
            min_x = max(0, voxel_xyz[0] - radius_x)
            max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
            min_y = max(0, voxel_xyz[1] - radius_y)
            max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
            min_z = max(0, voxel_xyz[2] - radius_z)
            max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
            voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
        return voxel_map

    def set_voxel_by_box(self, voxel_map, obj, value=1, pad_cm=0.0):
        """Fill the voxels inside an object's OBB with `value`.

        Uses the tight oriented box (`obb_center_world`, `obb_size`,
        `obb_rotation`) from the Observation, so tilted fixtures and rotated
        blocks get a snug fill instead of the inflated world-AABB slab that
        `obj.aabb[0]:obj.aabb[1]` slicing would produce.

        Iterates only the voxels in the world-AABB that encloses the rotated
        box, so cost is O(enclosing volume) regardless of box tilt.

        Args:
            voxel_map: (N, N, N) array or VoxelIndexingWrapper to write into.
            obj: Observation with `obb_center_world`, `obb_size`, `obb_rotation`.
                Falls back to obj.aabb for objects without OBB fields (table, ee).
            value: Value to write (default 1).
            pad_cm: Isotropic expansion of the OBB in cm before rasterizing.
        """
        if isinstance(voxel_map, VoxelIndexingWrapper):
            target = voxel_map.array
        else:
            target = voxel_map

        center_w = obj.get("obb_center_world") if isinstance(obj, dict) else None
        if center_w is None:
            # No OBB info (e.g. table/ee): fall back to axis-aligned fill.
            aabb = np.asarray(obj["aabb"], dtype=np.int32)
            lo = np.clip(aabb[0], 0, self._map_size - 1)
            hi = np.clip(aabb[1] + 1, 0, self._map_size)
            target[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = value
            return voxel_map

        center_w = np.asarray(center_w, dtype=np.float32)
        size = np.asarray(obj["obb_size"], dtype=np.float32) + 2.0 * (pad_cm / 100.0)
        R = np.asarray(obj["obb_rotation"], dtype=np.float32)
        half = size / 2.0

        # Enclosing world AABB of the padded OBB, in voxel indices.
        corners_world = obb_world_corners(center_w, size, R)
        vox_min = pc2voxel(
            corners_world.min(axis=0),
            self._workspace_min,
            self._workspace_max,
            self._map_size,
        )
        vox_max = pc2voxel(
            corners_world.max(axis=0),
            self._workspace_min,
            self._workspace_max,
            self._map_size,
        )
        lo = np.clip(np.minimum(vox_min, vox_max), 0, self._map_size - 1)
        hi = np.clip(np.maximum(vox_min, vox_max) + 1, 0, self._map_size)
        if np.any(hi <= lo):
            return voxel_map

        # Voxel centers in world coords, vectorized over the enclosing slab.
        ix = np.arange(lo[0], hi[0], dtype=np.float32)
        iy = np.arange(lo[1], hi[1], dtype=np.float32)
        iz = np.arange(lo[2], hi[2], dtype=np.float32)
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
        grid = np.stack([gx, gy, gz], axis=-1)  # (nx, ny, nz, 3)

        span = self._workspace_max - self._workspace_min
        world_pts = grid / (self._map_size - 1) * span + self._workspace_min

        # Transform into OBB local frame. R^T @ (v - c)  ==  (v - c) @ R.
        local = (world_pts - center_w) @ R
        inside = np.all(np.abs(local) <= half, axis=-1)

        target[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]][inside] = value
        return voxel_map

    def get_empty_affordance_map(self):
        """Return an empty affordance map (zeros)."""
        return self._get_default_voxel_map("target")

    def get_empty_avoidance_map(self):
        """Return an empty avoidance map (zeros)."""
        return self._get_default_voxel_map("obstacle")

    # ==========================================================
    # Rotation helpers (for the optional rotation-target slot in
    # composer stage tuples). All return world-frame 3x3 matrices
    # so callers can compose them freely; the steering module
    # converts to its internal 6D representation.
    # ==========================================================

    def rotation_about_axis(self, axis, angle_deg: float) -> np.ndarray:
        """Build a 3x3 rotation matrix about a principal axis or arbitrary vector.

        Args:
            axis: 'x' / 'y' / 'z' or a 3-vector (will be normalized).
            angle_deg: rotation angle in degrees, right-hand rule.
        """
        if isinstance(axis, str):
            key = axis.lower()
            if key == "x":
                u = np.array([1.0, 0.0, 0.0])
            elif key == "y":
                u = np.array([0.0, 1.0, 0.0])
            elif key == "z":
                u = np.array([0.0, 0.0, 1.0])
            else:
                raise ValueError(
                    f"axis must be 'x', 'y', 'z', or a 3-vector; got {axis!r}"
                )
        else:
            u = np.asarray(axis, dtype=np.float64)
            if u.shape != (3,):
                raise ValueError(f"axis vector must be shape (3,); got {u.shape}")
            u = normalize_vector(u)

        theta = np.deg2rad(float(angle_deg))
        c, s = np.cos(theta), np.sin(theta)
        ux, uy, uz = u
        # Rodrigues' rotation formula
        return np.array(
            [
                [
                    c + ux * ux * (1 - c),
                    ux * uy * (1 - c) - uz * s,
                    ux * uz * (1 - c) + uy * s,
                ],
                [
                    uy * ux * (1 - c) + uz * s,
                    c + uy * uy * (1 - c),
                    uy * uz * (1 - c) - ux * s,
                ],
                [
                    uz * ux * (1 - c) - uy * s,
                    uz * uy * (1 - c) + ux * s,
                    c + uz * uz * (1 - c),
                ],
            ],
            dtype=np.float64,
        )

    def compose_rotation(self, *rotations) -> np.ndarray:
        """Left-to-right matrix product: compose_rotation(R1, R2, R3) = R1 @ R2 @ R3."""
        if not rotations:
            return np.eye(3)
        out = np.asarray(rotations[0], dtype=np.float64)
        for r in rotations[1:]:
            out = out @ np.asarray(r, dtype=np.float64)
        return out

    def current_ee_rotation(self) -> np.ndarray:
        """Return the current end-effector orientation as a 3x3 matrix.

        CALVIN's robot_obs encodes orientation as Euler XYZ at indices [3:6],
        following pytorch3d's "XYZ" intrinsic convention used by Diffuser
        Actor's `convert_rotation` — i.e. the matrix is built as
        Rx(rx) @ Ry(ry) @ Rz(rz). Using a different order here would yield a
        different world-frame rotation and silently swap axes downstream.

        Use as a base for relative targets, e.g. compose_rotation(
        current_ee_rotation(), rotation_about_axis('z', 90)).
        """
        if self._robot_obs is None:
            return np.eye(3)
        rx, ry, rz = self._robot_obs[3:6]
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rx @ Ry @ Rz

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _detect_ee(self, obj_name: str) -> Observation:
        """Detect end-effector position."""
        if self._robot_obs is None:
            logger.warning("No robot_obs set, returning zero EE position")
            ee_pos = np.array([0, 0, 0])
            ee_pos_world = np.zeros(3)
        else:
            ee_pos = self._world_to_voxel(self._robot_obs[:3])
            ee_pos_world = self._robot_obs[:3]
        return Observation(
            {
                "name": obj_name,
                "position": ee_pos,
                "aabb": np.array([ee_pos, ee_pos]),
                "_position_world": ee_pos_world,
            }
        )

    def _detect_table(self, obj_name: str) -> Observation:
        """Detect table/workspace surface."""
        offset = 0.1
        x_min = self._workspace_min[0] + offset * (
            self._workspace_max[0] - self._workspace_min[0]
        )
        x_max = self._workspace_max[0] - offset * (
            self._workspace_max[0] - self._workspace_min[0]
        )
        y_min = self._workspace_min[1] + offset * (
            self._workspace_max[1] - self._workspace_min[1]
        )
        y_max = self._workspace_max[1] - offset * (
            self._workspace_max[1] - self._workspace_min[1]
        )
        # Table surface at z≈0.46
        z_val = 0.46
        table_min_world = np.array([x_min, y_min, z_val])
        table_max_world = np.array([x_max, y_max, z_val])
        table_center_world = (table_min_world + table_max_world) / 2

        return Observation(
            {
                "name": obj_name,
                "position": self._world_to_voxel(table_center_world),
                "aabb": np.array(
                    [
                        self._world_to_voxel(table_min_world),
                        self._world_to_voxel(table_max_world),
                    ]
                ),
                "_position_world": table_center_world,
                "normal": np.array([0, 0, 1]),
            }
        )

    def _get_held_block(self) -> Optional[str]:
        """Return the canonical name of the block currently held by the gripper, or None.

        A block is considered held when:
          - Gripper width < `HELD_BLOCK_GRIPPER_CLOSED_MAX_WIDTH` (closed-ish), AND
          - Block centroid within `HELD_BLOCK_PROXIMITY_RADIUS` of the EE.

        Returns one of `'red_block' | 'blue_block' | 'pink_block'`, or None.
        Used by `_detect_object` to resolve the generic `'block'` query when the
        composer emits a held-block-target stage (e.g., place_in_drawer stage 2).
        """
        if self._robot_obs is None or self._scene_obs is None:
            return None
        # robot_obs schema: [pos(3), euler(3), gripper_width(1), ...]
        ee_pos = np.asarray(self._robot_obs[:3], dtype=np.float32)
        if len(self._robot_obs) < 7:
            return None
        gripper_width = float(self._robot_obs[6])
        if gripper_width >= HELD_BLOCK_GRIPPER_CLOSED_MAX_WIDTH:
            return None
        candidates = [
            ("red_block", slice(6, 9)),
            ("blue_block", slice(12, 15)),
            ("pink_block", slice(18, 21)),
        ]
        best_name: Optional[str] = None
        best_dist = float("inf")
        for canonical_name, slc in candidates:
            if (
                self._block_aabbs
                and canonical_name in self._block_aabbs
                and "position" in self._block_aabbs[canonical_name]
            ):
                block_pos = np.asarray(
                    self._block_aabbs[canonical_name]["position"], dtype=np.float32
                )
            else:
                block_pos = np.asarray(self._scene_obs[slc], dtype=np.float32)
            dist = float(np.linalg.norm(ee_pos - block_pos))
            if dist < HELD_BLOCK_PROXIMITY_RADIUS and dist < best_dist:
                best_dist = dist
                best_name = canonical_name
        return best_name

    def _detect_object(self, obj_name: str, name_lower: str) -> Observation:
        """Detect a block or fixture by name."""
        # Check blocks first (positions from scene_obs)
        for block_name, obs_slice in BLOCK_SCENE_OBS.items():
            if block_name in name_lower:
                return self._detect_block(obj_name, obs_slice)

        # Generic 'block' query (no color specified): resolve via held-block
        # heuristic when the gripper is closed-ish on a block, else fall through.
        # This unblocks place_in_drawer / push_into_drawer / stack_block where
        # the composer emits stages on the generic `'block'` vocab token.
        if name_lower.strip() == "block":
            held = self._get_held_block()
            if held is not None:
                logger.info(
                    f"detect({obj_name!r}): held-block fallback resolved to {held!r}"
                )
                return self._detect_block(obj_name, BLOCK_SCENE_OBS[held])

        # Check fixtures, longest names first so 'drawer_interior' matches
        # before 'drawer', 'drawer_handle' before 'drawer', etc. The query
        # phrasing may use a space ('drawer handle') or underscore — match
        # against both forms by also testing `name_lower` with spaces→underscores.
        name_lower_us = name_lower.replace(" ", "_")
        for fixture_name in sorted(CALVIN_FIXTURES, key=len, reverse=True):
            if fixture_name in name_lower or fixture_name in name_lower_us:
                return self._detect_fixture(
                    obj_name, fixture_name, CALVIN_FIXTURES[fixture_name]
                )

        # Fallback: try to match partial names
        logger.warning(f"Unknown object '{obj_name}', attempting fuzzy match")
        for fixture_name, fixture_info in CALVIN_FIXTURES.items():
            if any(word in name_lower for word in fixture_name.split("_")):
                logger.info(f"Fuzzy matched '{obj_name}' to fixture '{fixture_name}'")
                return self._detect_fixture(obj_name, fixture_name, fixture_info)

        # No silent workspace-center fallback. The composer's affordance closure
        # will catch this in `StageManager._eval_map` and the stage activates
        # with no affordance map — surfaces the failure mode instead of misleading
        # the policy by pointing it at the middle of empty workspace.
        raise ObjectResolutionError(
            f"Could not detect {obj_name!r} (lower={name_lower!r}). "
            f"No BLOCK_SCENE_OBS match, no CALVIN_FIXTURES match, no fuzzy match. "
            f"Composer should emit a vocab-canonical object name from "
            f"{sorted(BLOCK_SCENE_OBS)} ∪ {sorted(CALVIN_FIXTURES)}."
        )

    def _detect_block(self, obj_name: str, obs_slice: slice) -> Observation:
        """Detect a block, preferring live PyBullet OBB over scene_obs fallback.

        The live OBB from _block_aabbs is orientation-aware; scene_obs fallback
        assumes identity rotation and a hardcoded BLOCK_SIZE cube (worse for
        the pink block, which is actually 7×5×5cm).

        `aabb` on the returned Observation is the *world-axis envelope* of the
        rotated box (the tight world-AABB of the 8 OBB corners). This keeps
        directional reasoning like "max_z of the block" correct under tilt.
        OBB fields (`obb_center_world`, `obb_size`, `obb_rotation`) are also
        attached for tight rasterization via `set_voxel_by_box`.
        """
        canonical = obj_name.lower().replace(" ", "_")
        key = next(
            (k for k in ("red_block", "blue_block", "pink_block") if k in canonical),
            None,
        )

        if self._block_aabbs and key in self._block_aabbs:
            live = self._block_aabbs[key]
            pos_world = np.asarray(live["position"], dtype=np.float32)
            size = np.asarray(live.get("size", BLOCK_SIZE), dtype=np.float32)
            rotation = np.asarray(live.get("rotation", np.eye(3)), dtype=np.float32)
            if "aabb_min" in live and "aabb_max" in live:
                aabb_min_world = np.asarray(live["aabb_min"], dtype=np.float32)
                aabb_max_world = np.asarray(live["aabb_max"], dtype=np.float32)
            else:
                corners = obb_world_corners(pos_world, size, rotation)
                aabb_min_world = corners.min(axis=0)
                aabb_max_world = corners.max(axis=0)
        elif self._scene_obs is not None:
            pos_world = self._scene_obs[obs_slice].copy()
            size = BLOCK_SIZE.astype(np.float32)
            rotation = np.eye(3, dtype=np.float32)
            half_size = size / 2
            aabb_min_world = pos_world - half_size
            aabb_max_world = pos_world + half_size
        else:
            logger.warning(
                f"No scene_obs or block_aabbs set, cannot detect '{obj_name}'"
            )
            center = (self._workspace_min + self._workspace_max) / 2
            return Observation(
                {
                    "name": obj_name,
                    "position": self._world_to_voxel(center),
                    "aabb": np.array(
                        [self._world_to_voxel(center), self._world_to_voxel(center)]
                    ),
                    "_position_world": center,
                }
            )

        return Observation(
            {
                "name": obj_name,
                "position": self._world_to_voxel(pos_world),
                "aabb": np.array(
                    [
                        self._world_to_voxel(aabb_min_world),
                        self._world_to_voxel(aabb_max_world),
                    ]
                ),
                "_position_world": pos_world,
                "obb_center_world": pos_world,
                "obb_size": size,
                "obb_rotation": rotation,
            }
        )

    def _detect_fixture(
        self, obj_name: str, fixture_name: str, fixture_info: dict
    ) -> Observation:
        """Detect a fixture, preferring live PyBullet positions over hardcoded.

        `aabb` is the world-axis envelope of the rotated OBB (tight bounding
        box of the 8 rotated corners). OBB fields are also attached for tight
        rasterization via `set_voxel_by_box`.
        """
        # Use live position from PyBullet if available
        if self._fixture_positions and fixture_name in self._fixture_positions:
            live = self._fixture_positions[fixture_name]
            pos_world = np.asarray(live["position"], dtype=np.float32).copy()
            size = np.asarray(live["size"], dtype=np.float32)
            rotation = np.asarray(live.get("rotation", np.eye(3)), dtype=np.float32)
        else:
            pos_world = fixture_info["position"].copy()
            size = fixture_info["size"].astype(np.float32)
            rotation = np.eye(3, dtype=np.float32)

        # Slider-interior half-slicing per VLM grounding. The full slider
        # cabinet spans both halves; only one half is open at a time. The
        # VLM emits `slider_accessible_chamber: 'left' | 'right' | None` to
        # tell us which half to target. Fallback to full BB when the field
        # is missing (None or no scene context). VLM-only per architecture
        # decision — no scene_obs fallback.
        if fixture_name == "slider_interior" and self._scene_context is not None:
            chamber = self._scene_context.get("slider_accessible_chamber")
            if chamber in ("left", "right"):
                half_sx = size[0] / 2.0
                # World-axis x: 'left' half = [center_x - sx/2, center_x],
                #               'right' half = [center_x, center_x + sx/2].
                sign = -1.0 if chamber == "left" else 1.0
                pos_world = pos_world.copy()
                pos_world[0] += sign * (half_sx / 2.0)
                size = size.copy()
                size[0] = half_sx

        corners_world = obb_world_corners(pos_world, size, rotation)
        aabb_min_world = corners_world.min(axis=0)
        aabb_max_world = corners_world.max(axis=0)

        return Observation(
            {
                "name": obj_name,
                "position": self._world_to_voxel(pos_world),
                "aabb": np.array(
                    [
                        self._world_to_voxel(aabb_min_world),
                        self._world_to_voxel(aabb_max_world),
                    ]
                ),
                "_position_world": pos_world,
                "obb_center_world": pos_world,
                "obb_size": size,
                "obb_rotation": rotation,
            }
        )

    def _get_default_voxel_map(self, map_type: str):
        """Create a default voxel map wrapped in VoxelIndexingWrapper."""
        if map_type in ("target", "obstacle"):
            arr = np.zeros((self._map_size, self._map_size, self._map_size))
        else:
            raise ValueError(f"Unknown voxel map type: {map_type}")
        return VoxelIndexingWrapper(arr)

    def _world_to_voxel(self, world_xyz):
        """Convert world coordinates to voxel coordinates."""
        return pc2voxel(
            world_xyz, self._workspace_min, self._workspace_max, self._map_size
        )

    def _voxel_to_world(self, voxel_xyz):
        """Convert voxel coordinates to world coordinates."""
        return voxel2pc(
            voxel_xyz, self._workspace_min, self._workspace_max, self._map_size
        )

    def _points_to_voxel_map(self, points):
        """Convert world-frame point cloud to voxel occupancy map."""
        return pc2voxel_map(
            points, self._workspace_min, self._workspace_max, self._map_size
        )

    def get_object_names(self) -> list:
        """Return list of detectable object names for LMP context."""
        names = ["red block", "blue block", "pink block"]
        names += list(CALVIN_FIXTURES.keys())
        names += ["table"]
        return names

    def get_all_detections(self) -> list:
        """Detect all known objects and return a list of Observations.

        Useful for visualization — shows bounding boxes for every object.
        """
        detections = []
        for name in self.get_object_names():
            try:
                obs = self.detect(name)
                detections.append(obs)
            except Exception as e:
                logger.warning(f"Failed to detect '{name}': {e}")
        return detections
