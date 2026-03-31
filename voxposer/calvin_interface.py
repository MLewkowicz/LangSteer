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

# Workspace bounds in absolute world coordinates (meters).
# Covers all CALVIN objects: blocks, slider, drawer, lights, table surface.
DEFAULT_WORKSPACE_MIN = np.array([-0.35, -0.40, 0.30])
DEFAULT_WORKSPACE_MAX = np.array([0.35, 0.15, 0.85])

# Aliases for end-effector
EE_ALIAS = [
    'ee', 'endeffector', 'end_effector', 'end effector',
    'gripper', 'hand',
]

# Aliases for table/workspace
TABLE_ALIAS = [
    'table', 'desk', 'workstation', 'work_station', 'work station',
    'workspace', 'work_space', 'work space',
]

# CALVIN fixed fixture positions from PyBullet per-link AABBs on playtable_8 (UID=5).
# These objects don't move between episodes.
# Link mapping: 0=button, 1=switch, 2=slide, 3=drawer, 4=led, 5=light
CALVIN_FIXTURES = {
    'slider': {
        # slide_link (Link 2) — full slider track
        'position': np.array([0.040, 0.040, 0.555]),
        'size': np.array([0.289, 0.10, 0.04]),
    },
    'slider_left': {
        # Left end of slide_link track
        'position': np.array([-0.105, 0.040, 0.555]),
        'size': np.array([0.06, 0.06, 0.04]),
    },
    'slider_right': {
        # Right end of slide_link track
        'position': np.array([0.184, 0.040, 0.555]),
        'size': np.array([0.06, 0.06, 0.04]),
    },
    'drawer': {
        # drawer_link (Link 3)
        'position': np.array([0.180, -0.100, 0.350]),
        'size': np.array([0.15, 0.12, 0.08]),
    },
    'drawer_handle': {
        # Front face of drawer_link
        'position': np.array([0.180, -0.265, 0.350]),
        'size': np.array([0.06, 0.04, 0.03]),
    },
    'lightbulb': {
        # light_link (Link 5)
        'position': np.array([0.300, 0.160, 0.673]),
        'size': np.array([0.062, 0.062, 0.056]),
    },
    'light_switch': {
        # switch_link (Link 1) — controls the lightbulb
        'position': np.array([0.300, 0.037, 0.518]),
        'size': np.array([0.06, 0.06, 0.06]),
    },
    'switch': {
        # Alias for light_switch
        'position': np.array([0.300, 0.037, 0.518]),
        'size': np.array([0.06, 0.06, 0.06]),
    },
    'led': {
        # led_link (Link 4)
        'position': np.array([-0.120, 0.160, 0.656]),
        'size': np.array([0.06, 0.046, 0.022]),
    },
    'button': {
        # button_link (Link 0) — controls the LED
        'position': np.array([-0.120, -0.120, 0.472]),
        'size': np.array([0.07, 0.07, 0.03]),
    },
}

# Block size in meters (from block_red.urdf: <box size="0.05 0.05 0.05"/>)
BLOCK_SIZE = np.array([0.05, 0.05, 0.05])

# scene_obs slices for block positions
BLOCK_SCENE_OBS = {
    'red_block': slice(6, 9),
    'red block': slice(6, 9),
    'blue_block': slice(12, 15),
    'blue block': slice(12, 15),
    'pink_block': slice(18, 21),
    'pink block': slice(18, 21),
}


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
    set_voxel_by_radius, get_empty_*_map, get_ee_pos) but uses CALVIN's
    scene_obs for object detection instead of RLBench's per-object point clouds.
    """

    def __init__(self, config: dict):
        self._map_size = config.get('map_size', 100)
        self._workspace_min = np.array(
            config.get('workspace_bounds_min', DEFAULT_WORKSPACE_MIN),
            dtype=np.float32,
        )
        self._workspace_max = np.array(
            config.get('workspace_bounds_max', DEFAULT_WORKSPACE_MAX),
            dtype=np.float32,
        )

        # Voxel resolution (meters per voxel)
        self._resolution = (self._workspace_max - self._workspace_min) / self._map_size

        # Current state (updated each episode/step)
        self._robot_obs: Optional[np.ndarray] = None  # (15,)
        self._scene_obs: Optional[np.ndarray] = None  # (24,)

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

    def update_state(self, robot_obs: np.ndarray, scene_obs: np.ndarray):
        """Update current robot and scene state for object detection."""
        self._robot_obs = np.asarray(robot_obs, dtype=np.float32)
        self._scene_obs = np.asarray(scene_obs, dtype=np.float32)

    # ==========================================================
    # Functions exposed to LLM-generated code
    # ==========================================================

    def get_ee_pos(self):
        """Get end-effector position in voxel coordinates."""
        if self._robot_obs is None:
            logger.warning("No robot_obs set, returning zero EE position")
            return np.array([0, 0, 0])
        return self._world_to_voxel(self._robot_obs[:3])

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
        if isinstance(direction, str) and direction == 'x':
            return int(cm / (self._resolution[0] * 100))
        elif isinstance(direction, str) and direction == 'y':
            return int(cm / (self._resolution[1] * 100))
        elif isinstance(direction, str) and direction == 'z':
            return int(cm / (self._resolution[2] * 100))
        else:
            assert isinstance(direction, np.ndarray) and direction.shape == (3,)
            direction = normalize_vector(direction)
            x_index = self.cm2index(cm * direction[0], 'x')
            y_index = self.cm2index(cm * direction[1], 'y')
            z_index = self.cm2index(cm * direction[2], 'z')
            return np.array([x_index, y_index, z_index])

    def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
        """Set voxels within radius_cm of position to value."""
        voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
        if radius_cm > 0:
            radius_x = self.cm2index(radius_cm, 'x')
            radius_y = self.cm2index(radius_cm, 'y')
            radius_z = self.cm2index(radius_cm, 'z')
            min_x = max(0, voxel_xyz[0] - radius_x)
            max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
            min_y = max(0, voxel_xyz[1] - radius_y)
            max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
            min_z = max(0, voxel_xyz[2] - radius_z)
            max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
            voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
        return voxel_map

    def get_empty_affordance_map(self):
        """Return an empty affordance map (zeros)."""
        return self._get_default_voxel_map('target')

    def get_empty_avoidance_map(self):
        """Return an empty avoidance map (zeros)."""
        return self._get_default_voxel_map('obstacle')

    def get_empty_gripper_map(self):
        """Return an empty gripper map (ones = open)."""
        return self._get_default_voxel_map('gripper')

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _detect_ee(self, obj_name: str) -> Observation:
        """Detect end-effector position."""
        ee_pos = self.get_ee_pos()
        ee_pos_world = self._robot_obs[:3] if self._robot_obs is not None else np.zeros(3)
        return Observation({
            'name': obj_name,
            'position': ee_pos,
            'aabb': np.array([ee_pos, ee_pos]),
            '_position_world': ee_pos_world,
        })

    def _detect_table(self, obj_name: str) -> Observation:
        """Detect table/workspace surface."""
        offset = 0.1
        x_min = self._workspace_min[0] + offset * (self._workspace_max[0] - self._workspace_min[0])
        x_max = self._workspace_max[0] - offset * (self._workspace_max[0] - self._workspace_min[0])
        y_min = self._workspace_min[1] + offset * (self._workspace_max[1] - self._workspace_min[1])
        y_max = self._workspace_max[1] - offset * (self._workspace_max[1] - self._workspace_min[1])
        # Table surface at z≈0.46
        z_val = 0.46
        table_min_world = np.array([x_min, y_min, z_val])
        table_max_world = np.array([x_max, y_max, z_val])
        table_center_world = (table_min_world + table_max_world) / 2

        return Observation({
            'name': obj_name,
            'position': self._world_to_voxel(table_center_world),
            'aabb': np.array([
                self._world_to_voxel(table_min_world),
                self._world_to_voxel(table_max_world),
            ]),
            '_position_world': table_center_world,
            'normal': np.array([0, 0, 1]),
        })

    def _detect_object(self, obj_name: str, name_lower: str) -> Observation:
        """Detect a block or fixture by name."""
        # Check blocks first (positions from scene_obs)
        for block_name, obs_slice in BLOCK_SCENE_OBS.items():
            if block_name in name_lower:
                return self._detect_block(obj_name, obs_slice)

        # Check fixtures (hardcoded positions), longest names first so
        # 'slider_left' matches before 'slider', 'drawer_handle' before 'drawer'
        for fixture_name in sorted(CALVIN_FIXTURES, key=len, reverse=True):
            if fixture_name in name_lower:
                return self._detect_fixture(obj_name, CALVIN_FIXTURES[fixture_name])

        # Fallback: try to match partial names
        logger.warning(f"Unknown object '{obj_name}', attempting fuzzy match")
        for fixture_name, fixture_info in CALVIN_FIXTURES.items():
            if any(word in name_lower for word in fixture_name.split('_')):
                logger.info(f"Fuzzy matched '{obj_name}' to fixture '{fixture_name}'")
                return self._detect_fixture(obj_name, fixture_info)

        # Last resort: return workspace center
        logger.warning(f"Could not detect '{obj_name}', returning workspace center")
        center = (self._workspace_min + self._workspace_max) / 2
        return Observation({
            'name': obj_name,
            'position': self._world_to_voxel(center),
            'aabb': np.array([
                self._world_to_voxel(self._workspace_min),
                self._world_to_voxel(self._workspace_max),
            ]),
            '_position_world': center,
        })

    def _detect_block(self, obj_name: str, obs_slice: slice) -> Observation:
        """Detect a block from scene_obs ground-truth position."""
        if self._scene_obs is None:
            logger.warning(f"No scene_obs set, cannot detect '{obj_name}'")
            center = (self._workspace_min + self._workspace_max) / 2
            return Observation({
                'name': obj_name,
                'position': self._world_to_voxel(center),
                'aabb': np.array([self._world_to_voxel(center), self._world_to_voxel(center)]),
                '_position_world': center,
            })

        pos_world = self._scene_obs[obs_slice].copy()
        half_size = BLOCK_SIZE / 2
        aabb_min_world = pos_world - half_size
        aabb_max_world = pos_world + half_size

        return Observation({
            'name': obj_name,
            'position': self._world_to_voxel(pos_world),
            'aabb': np.array([
                self._world_to_voxel(aabb_min_world),
                self._world_to_voxel(aabb_max_world),
            ]),
            '_position_world': pos_world,
        })

    def _detect_fixture(self, obj_name: str, fixture_info: dict) -> Observation:
        """Detect a fixed fixture from hardcoded position."""
        pos_world = fixture_info['position'].copy()
        half_size = fixture_info['size'] / 2
        aabb_min_world = pos_world - half_size
        aabb_max_world = pos_world + half_size

        return Observation({
            'name': obj_name,
            'position': self._world_to_voxel(pos_world),
            'aabb': np.array([
                self._world_to_voxel(aabb_min_world),
                self._world_to_voxel(aabb_max_world),
            ]),
            '_position_world': pos_world,
        })

    def _get_default_voxel_map(self, map_type: str):
        """Create a default voxel map wrapped in VoxelIndexingWrapper."""
        if map_type == 'target':
            arr = np.zeros((self._map_size, self._map_size, self._map_size))
        elif map_type == 'obstacle':
            arr = np.zeros((self._map_size, self._map_size, self._map_size))
        elif map_type == 'gripper':
            arr = np.ones((self._map_size, self._map_size, self._map_size))
        else:
            raise ValueError(f'Unknown voxel map type: {map_type}')
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
        names = ['red block', 'blue block', 'pink block']
        names += list(CALVIN_FIXTURES.keys())
        names += ['table']
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
