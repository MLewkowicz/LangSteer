"""PyBullet GUI renderer for visualizing robot rollouts.

Extracted from scripts/rollout_reference.py
"""

import time
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class PyBulletRenderer:
    """Renders rollouts in PyBullet GUI with configurable playback settings."""

    def __init__(self, config):
        """
        Initialize PyBullet renderer.

        Args:
            config: RolloutVisualizationConfig instance
        """
        self.config = config
        self.frame_skip = config.frame_skip
        self.playback_speed = config.playback_speed
        self.keep_gui_open = config.keep_gui_open

    def render_episode(self, env, actions: Optional[np.ndarray] = None):
        """
        Render an episode in PyBullet GUI.

        This method assumes the environment already has use_gui=True and
        simply controls the playback timing.

        Args:
            env: Environment instance (must have use_gui=True)
            actions: Optional action sequence to play back. If None, renders current state only.
        """
        if actions is None:
            # Just render current state
            env.render()
            return

        total_frames = len(actions)
        frames_to_play = actions[::self.frame_skip]

        logger.info(f"Rendering {len(frames_to_play)} frames (frame_skip={self.frame_skip}, "
                   f"speed={self.playback_speed}x)")

        try:
            for step_idx, action in enumerate(frames_to_play):
                original_frame = step_idx * self.frame_skip

                # Convert action to required format
                if not isinstance(action, np.ndarray):
                    action_array = np.array(action, dtype=np.float32)
                else:
                    action_array = action.astype(np.float32)

                # Execute action
                env.step(action)

                # Print progress periodically
                if step_idx % 10 == 0 or step_idx == len(frames_to_play) - 1:
                    logger.info(f"Step {step_idx}/{len(frames_to_play)-1} "
                               f"(frame {original_frame}/{total_frames-1})")

                # Add delay for visualization (adjusted by playback speed)
                time.sleep(0.01 / self.playback_speed)

            logger.info(f"Rendering completed - played {len(frames_to_play)} frames")

            # Keep GUI open if requested
            if self.keep_gui_open:
                logger.info("GUI is open. Press Ctrl+C to exit...")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Exiting GUI...")

        except KeyboardInterrupt:
            logger.info(f"Rendering interrupted by user at step {step_idx}/{len(frames_to_play)-1}")

    def render_step(self, env, action: Optional[np.ndarray] = None):
        """
        Render a single step.

        Args:
            env: Environment instance
            action: Optional action to execute. If None, just renders current state.
        """
        if action is not None:
            env.step(action)

        # Add visualization delay
        time.sleep(0.01 / self.playback_speed)
