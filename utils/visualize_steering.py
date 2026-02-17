"""Visualization utilities for Tweedie steering and reference trajectories."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def visualize_reference_trajectory(
    reference_actions,
    save_path=None,
    title="Reference Trajectory"
):
    """
    Visualize a reference trajectory from CALVIN.

    Args:
        reference_actions: (T, 7) numpy array of actions [x, y, z, rx, ry, rz, gripper]
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    if isinstance(reference_actions, list):
        reference_actions = np.array(reference_actions)

    fig = plt.figure(figsize=(15, 5))

    # Plot 1: 3D trajectory (position only)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(reference_actions[:, 0], reference_actions[:, 1], reference_actions[:, 2],
             'b-', linewidth=2, label='Reference')
    ax1.scatter(reference_actions[0, 0], reference_actions[0, 1], reference_actions[0, 2],
                c='g', s=100, marker='o', label='Start')
    ax1.scatter(reference_actions[-1, 0], reference_actions[-1, 1], reference_actions[-1, 2],
                c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Position Trajectory')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Position over time
    ax2 = fig.add_subplot(132)
    timesteps = np.arange(len(reference_actions))
    ax2.plot(timesteps, reference_actions[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(timesteps, reference_actions[:, 1], 'g-', label='Y', linewidth=2)
    ax2.plot(timesteps, reference_actions[:, 2], 'b-', label='Z', linewidth=2)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Position')
    ax2.set_title('Position Components Over Time')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Orientation and gripper
    ax3 = fig.add_subplot(133)
    ax3.plot(timesteps, reference_actions[:, 3], label='Roll', linewidth=2)
    ax3.plot(timesteps, reference_actions[:, 4], label='Pitch', linewidth=2)
    ax3.plot(timesteps, reference_actions[:, 5], label='Yaw', linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(timesteps, reference_actions[:, 6], 'k--', label='Gripper', linewidth=2)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Orientation (rad)')
    ax3_twin.set_ylabel('Gripper')
    ax3.set_title('Orientation and Gripper Over Time')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved reference trajectory visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_trajectories(
    reference_actions,
    executed_actions,
    save_path=None,
    title="Trajectory Comparison"
):
    """
    Compare reference trajectory with executed trajectory.

    Args:
        reference_actions: (T_ref, 7) numpy array of reference actions
        executed_actions: (T_exec, 7) numpy array of executed actions
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    if isinstance(reference_actions, list):
        reference_actions = np.array(reference_actions)
    if isinstance(executed_actions, list):
        executed_actions = np.array(executed_actions)

    # Align trajectories to same length for comparison
    min_len = min(len(reference_actions), len(executed_actions))
    ref_aligned = reference_actions[:min_len]
    exec_aligned = executed_actions[:min_len]

    fig = plt.figure(figsize=(18, 10))

    # Plot 1: 3D trajectory comparison
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(ref_aligned[:, 0], ref_aligned[:, 1], ref_aligned[:, 2],
             'b-', linewidth=2, alpha=0.7, label='Reference')
    ax1.plot(exec_aligned[:, 0], exec_aligned[:, 1], exec_aligned[:, 2],
             'r-', linewidth=2, alpha=0.7, label='Executed')
    ax1.scatter(ref_aligned[0, 0], ref_aligned[0, 1], ref_aligned[0, 2],
                c='g', s=100, marker='o', label='Start')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Position Trajectory')
    ax1.legend()
    ax1.grid(True)

    timesteps = np.arange(min_len)

    # Plot 2: X position comparison
    ax2 = fig.add_subplot(232)
    ax2.plot(timesteps, ref_aligned[:, 0], 'b-', label='Reference', linewidth=2)
    ax2.plot(timesteps, exec_aligned[:, 0], 'r--', label='Executed', linewidth=2)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('X Position')
    ax2.set_title('X Position Over Time')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Y position comparison
    ax3 = fig.add_subplot(233)
    ax3.plot(timesteps, ref_aligned[:, 1], 'b-', label='Reference', linewidth=2)
    ax3.plot(timesteps, exec_aligned[:, 1], 'r--', label='Executed', linewidth=2)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Y Position Over Time')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Z position comparison
    ax4 = fig.add_subplot(234)
    ax4.plot(timesteps, ref_aligned[:, 2], 'b-', label='Reference', linewidth=2)
    ax4.plot(timesteps, exec_aligned[:, 2], 'r--', label='Executed', linewidth=2)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Z Position')
    ax4.set_title('Z Position Over Time')
    ax4.legend()
    ax4.grid(True)

    # Plot 5: MSE per timestep
    ax5 = fig.add_subplot(235)
    mse_per_step = np.mean((ref_aligned[:, :3] - exec_aligned[:, :3]) ** 2, axis=1)
    ax5.plot(timesteps, mse_per_step, 'g-', linewidth=2)
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Position MSE')
    ax5.set_title(f'Position MSE Over Time (Avg: {np.mean(mse_per_step):.4f})')
    ax5.grid(True)

    # Plot 6: Gripper comparison
    ax6 = fig.add_subplot(236)
    ax6.plot(timesteps, ref_aligned[:, 6], 'b-', label='Reference', linewidth=2)
    ax6.plot(timesteps, exec_aligned[:, 6], 'r--', label='Executed', linewidth=2)
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Gripper State')
    ax6.set_title('Gripper State Over Time')
    ax6.legend()
    ax6.grid(True)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved trajectory comparison to {save_path}")
    else:
        plt.show()

    plt.close()

    # Compute and return metrics
    metrics = {
        'position_mse': np.mean((ref_aligned[:, :3] - exec_aligned[:, :3]) ** 2),
        'position_mae': np.mean(np.abs(ref_aligned[:, :3] - exec_aligned[:, :3])),
        'orientation_mse': np.mean((ref_aligned[:, 3:6] - exec_aligned[:, 3:6]) ** 2),
        'gripper_mse': np.mean((ref_aligned[:, 6] - exec_aligned[:, 6]) ** 2),
        'max_position_error': np.max(np.linalg.norm(ref_aligned[:, :3] - exec_aligned[:, :3], axis=1))
    }

    return metrics


def visualize_sliding_window(
    reference_trajectory,
    current_step,
    horizon=16,
    save_path=None
):
    """
    Visualize the sliding window at a specific timestep.

    Args:
        reference_trajectory: (T, 7) full reference trajectory
        current_step: Current episode step
        horizon: Prediction horizon
        save_path: Optional path to save the figure
    """
    if isinstance(reference_trajectory, list):
        reference_trajectory = np.array(reference_trajectory)

    fig = plt.figure(figsize=(15, 5))

    # Get sliding window
    start_idx = current_step
    end_idx = min(start_idx + horizon, len(reference_trajectory))
    window = reference_trajectory[start_idx:end_idx]

    # Plot 1: Full trajectory with window highlighted
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(reference_trajectory[:, 0], reference_trajectory[:, 1], reference_trajectory[:, 2],
             'b-', linewidth=1, alpha=0.3, label='Full Reference')
    ax1.plot(window[:, 0], window[:, 1], window[:, 2],
             'r-', linewidth=3, label=f'Current Window (step {current_step})')
    ax1.scatter(window[0, 0], window[0, 1], window[0, 2],
                c='g', s=150, marker='o', label='Window Start')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Sliding Window in 3D Space')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Position components of window
    ax2 = fig.add_subplot(122)
    window_steps = np.arange(len(window)) + current_step
    ax2.plot(window_steps, window[:, 0], 'r-', label='X', linewidth=2, marker='o')
    ax2.plot(window_steps, window[:, 1], 'g-', label='Y', linewidth=2, marker='o')
    ax2.plot(window_steps, window[:, 2], 'b-', label='Z', linewidth=2, marker='o')
    ax2.axvline(x=current_step, color='k', linestyle='--', alpha=0.5, label='Current Step')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Position')
    ax2.set_title(f'Window Position Components (Steps {current_step}-{current_step + len(window) - 1})')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(f'Sliding Window Visualization (Step {current_step}, Horizon {horizon})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved sliding window visualization to {save_path}")
    else:
        plt.show()

    plt.close()
