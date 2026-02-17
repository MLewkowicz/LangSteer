"""Matplotlib renderer for reference trajectory visualization.

Extracted from scripts/visualize_reference.py
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional
import sys

# Import existing visualization utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.visualize_steering import (
    visualize_reference_trajectory,
    visualize_sliding_window,
    compare_trajectories
)

logger = logging.getLogger(__name__)


class MatplotlibRenderer:
    """Renders reference trajectories using matplotlib plots."""

    def __init__(self, config):
        """
        Initialize matplotlib renderer.

        Args:
            config: ReferenceVisualizationConfig instance
        """
        self.config = config
        self.output_dir = Path(config.save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_reference_trajectory(
        self,
        actions: np.ndarray,
        task_name: str = "task",
        horizon: int = 16
    ):
        """
        Render reference trajectory with full and sliding window views.

        Args:
            actions: Action sequence array (T, 7) - [x, y, z, roll, pitch, yaw, gripper]
            task_name: Task name for plot titles and filenames
            horizon: Prediction horizon for sliding window visualization
        """
        logger.info(f"Generating matplotlib visualizations for task: {task_name}")
        logger.info(f"Trajectory length: {len(actions)} frames")

        # Generate full trajectory visualization
        if self.config.plot_full_trajectory:
            save_path = self.output_dir / f"reference_{task_name}_full.png"
            logger.info(f"Creating full trajectory plot: {save_path}")

            visualize_reference_trajectory(
                actions,
                save_path=str(save_path),
                title=f"Reference Trajectory: {task_name}"
            )

        # Generate sliding window visualizations
        if self.config.plot_sliding_windows:
            # Use configured timesteps, or default based on trajectory length
            timesteps = self.config.window_timesteps
            if len(actions) <= 30:
                timesteps = [0, len(actions) // 2]

            for step in timesteps:
                if step < len(actions):
                    save_path = self.output_dir / f"reference_{task_name}_window_step{step}.png"
                    logger.info(f"Creating sliding window plot at step {step}: {save_path}")

                    visualize_sliding_window(
                        actions,
                        current_step=step,
                        horizon=horizon,
                        save_path=str(save_path)
                    )

        # Print trajectory statistics
        logger.info("Trajectory Statistics:")
        logger.info(f"  Position range:")
        logger.info(f"    X: [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}]")
        logger.info(f"    Y: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]")
        logger.info(f"    Z: [{actions[:, 2].min():.3f}, {actions[:, 2].max():.3f}]")
        logger.info(f"  Gripper range: [{actions[:, 6].min():.3f}, {actions[:, 6].max():.3f}]")

        logger.info(f"All visualizations saved to: {self.output_dir}")

    def render_trajectory_comparison(
        self,
        reference_actions: np.ndarray,
        executed_actions: np.ndarray,
        task_name: str = "task"
    ):
        """
        Compare reference vs executed trajectories.

        Args:
            reference_actions: Reference action sequence (T, 7)
            executed_actions: Executed action sequence (T, 7)
            task_name: Task name for plot titles and filenames
        """
        save_path = self.output_dir / f"comparison_{task_name}.png"
        logger.info(f"Creating trajectory comparison plot: {save_path}")

        metrics = compare_trajectories(
            reference_actions,
            executed_actions,
            save_path=str(save_path),
            title=f"Trajectory Comparison: {task_name}"
        )

        logger.info("Comparison Metrics:")
        logger.info(f"  Position MSE: {metrics['position_mse']:.4f}")
        logger.info(f"  Position MAE: {metrics['position_mae']:.4f}")
        logger.info(f"  Orientation MSE: {metrics['orientation_mse']:.4f}")
        logger.info(f"  Max Error: {metrics['max_error']:.4f}")

        return metrics
