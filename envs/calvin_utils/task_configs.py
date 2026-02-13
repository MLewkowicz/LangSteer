"""
Task configuration utilities for CALVIN environment.

Maps CALVIN task names to their required initial scene states.
Based on task preconditions from CALVIN's multistep_sequences.py.
"""

import contextlib
import logging
from typing import Dict, Any, Tuple
import numpy as np
from numpy import pi

logger = logging.getLogger(__name__)


def deterministic_hash(s: str) -> int:
    """Generate a deterministic 32-bit hash from a string."""
    # Use built-in hash but make it positive and 32-bit
    return abs(hash(s)) % (2**32)


@contextlib.contextmanager
def temp_seed(seed):
    """Temporarily set numpy random seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert initial condition dict to robot_obs and scene_obs arrays.

    Based on CALVIN's evaluation utils (lines 207-275).

    Args:
        initial_condition: Dictionary with keys:
            - slider: "left" or "right"
            - drawer: "open" or "closed"
            - lightbulb: 0 or 1
            - led: 0 or 1
            - red_block: "table", "slider_left", "slider_right", "drawer"
            - blue_block: "table", "slider_left", "slider_right", "drawer"
            - pink_block: "table", "slider_left", "slider_right", "drawer"

    Returns:
        robot_obs: (15,) Fixed robot state at neutral position
        scene_obs: (24,) Scene state with object positions and states
    """
    # Fixed robot state (neutral position)
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )

    # Block position configurations
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]

    # Use deterministic random seed based on initial condition hash
    seed = deterministic_hash(str(sorted(initial_condition.items())))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)

        # [0] Slider joint state
        if initial_condition.get("slider") == "left":
            scene_obs[0] = 0.28

        # [1] Drawer joint state
        if initial_condition.get("drawer") == "open":
            scene_obs[1] = 0.22

        # [2-3] Button and switch states (unused, kept at 0)

        # [4] Lightbulb state
        if initial_condition.get("lightbulb") == 1:
            scene_obs[3] = 0.088  # switch state when lightbulb is on
        scene_obs[4] = initial_condition.get("lightbulb", 0)

        # [5] LED state
        scene_obs[5] = initial_condition.get("led", 0)

        # [6-11] Red block position and orientation
        red_block_pos = initial_condition.get("red_block", "table")
        if red_block_pos == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif red_block_pos == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:  # "table" or "drawer" (drawer uses table position as fallback)
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)

        # [12-17] Blue block position and orientation
        blue_block_pos = initial_condition.get("blue_block", "table")
        if blue_block_pos == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif blue_block_pos == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif red_block_pos == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)

        # [18-23] Pink block position and orientation
        pink_block_pos = initial_condition.get("pink_block", "table")
        if pink_block_pos == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif pink_block_pos == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


# Default initial condition (all objects on table, drawer/slider closed)
DEFAULT_INITIAL_CONDITION = {
    "slider": "right",
    "drawer": "closed",
    "lightbulb": 0,
    "led": 0,
    "red_block": "table",
    "blue_block": "table",
    "pink_block": "table"
}


# Task-to-initial-condition mapping based on CALVIN task preconditions
# From calvin/calvin_models/calvin_agent/evaluation/multistep_sequences.py
TASK_INITIAL_CONDITIONS = {
    # Rotation tasks (blocks on table, gripper not grasping)
    "rotate_red_block_right": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "rotate_red_block_left": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "rotate_blue_block_right": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "rotate_blue_block_left": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "rotate_pink_block_right": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "rotate_pink_block_left": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },

    # Push tasks (blocks on table)
    "push_red_block_right": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "push_red_block_left": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "push_blue_block_right": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "push_blue_block_left": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "push_pink_block_right": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "push_pink_block_left": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },

    # Slider tasks
    "move_slider_left": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "move_slider_right": {
        "slider": "left", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },

    # Drawer tasks
    "open_drawer": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "close_drawer": {
        "slider": "right", "drawer": "open", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },

    # Lift from table
    "lift_red_block_table": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "lift_blue_block_table": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "lift_pink_block_table": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },

    # Lift from slider (slider open, block on slider)
    "lift_red_block_slider": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "slider_left", "blue_block": "table", "pink_block": "table"
    },
    "lift_blue_block_slider": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "slider_left", "pink_block": "table"
    },
    "lift_pink_block_slider": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "slider_left"
    },

    # Lift from drawer (drawer open, block in drawer position on table)
    # Note: CALVIN doesn't actually place blocks "in" drawer in scene_obs
    # The drawer task checker uses proximity detection
    "lift_red_block_drawer": {
        "slider": "right", "drawer": "open", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "lift_blue_block_drawer": {
        "slider": "right", "drawer": "open", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "lift_pink_block_drawer": {
        "slider": "right", "drawer": "open", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },

    # Place tasks (assume some block is grasped - use default state)
    "place_in_slider": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "place_in_drawer": {
        "slider": "right", "drawer": "open", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },

    # Push into drawer (one block on table, drawer open)
    "push_into_drawer": {
        "slider": "right", "drawer": "open", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "slider_left", "pink_block": "slider_right"
    },

    # Stack/unstack (blocks on table)
    "stack_block": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "unstack_block": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },

    # Light tasks
    "turn_on_lightbulb": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "turn_off_lightbulb": {
        "slider": "right", "drawer": "closed", "lightbulb": 1, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "turn_on_led": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 0,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
    "turn_off_led": {
        "slider": "right", "drawer": "closed", "lightbulb": 0, "led": 1,
        "red_block": "table", "blue_block": "table", "pink_block": "table"
    },
}


def get_initial_condition_for_task(task_name: str) -> Dict[str, Any]:
    """
    Get the required initial condition for a CALVIN task.

    Args:
        task_name: CALVIN task name (e.g., "open_drawer", "lift_red_block_table")

    Returns:
        Dictionary with initial condition (slider, drawer, lightbulb, led, block positions)
    """
    initial_condition = TASK_INITIAL_CONDITIONS.get(task_name, DEFAULT_INITIAL_CONDITION)
    logger.debug(f"Initial condition for task '{task_name}': {initial_condition}")
    return initial_condition.copy()
