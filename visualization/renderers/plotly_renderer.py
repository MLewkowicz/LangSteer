"""Plotly renderer for 3D trajectory visualization.

Extracted from scripts/visualize_trajectories.py
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Optional
import sys

# Import trajectory visualization utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.visualization.trajectory_viz import TrajectoryVisualizer
from visualization.collectors import TrajectoryCollector
from utils.state_management.env_snapshots import EnvSnapshot, EnvSnapshotManager

logger = logging.getLogger(__name__)


class PlotlyRenderer:
    """Renders multi-rollout 3D trajectory visualizations using Plotly."""

    def __init__(self, config):
        """
        Initialize Plotly renderer.

        Args:
            config: TrajectoryVisualizationConfig instance
        """
        self.config = config
        self.output_dir = Path(config.save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_multi_rollout_trajectories(
        self,
        env,
        policy,
        steering=None,
        snapshot: Optional[EnvSnapshot] = None
    ):
        """
        Collect and visualize trajectories from multiple rollouts.

        Args:
            env: Environment instance
            policy: Policy instance
            steering: Optional steering module
            snapshot: Optional pre-captured environment snapshot. If None, creates new snapshot.

        Returns:
            Dictionary with statistics and saved file paths
        """
        logger.info("="*70)
        logger.info("PLOTLY 3D TRAJECTORY VISUALIZATION - Multi-Rollout Analysis")
        logger.info("="*70)

        # ====================================================================
        # STEP 1: Load or create environment snapshot
        # ====================================================================
        snapshot_manager = EnvSnapshotManager()

        if snapshot is None:
            if self.config.snapshot_load_path:
                logger.info(f"Loading snapshot from: {self.config.snapshot_load_path}")
                snapshot = EnvSnapshot.load(self.config.snapshot_load_path)
            else:
                logger.info("Creating new environment snapshot...")
                obs = env.reset()
                snapshot = snapshot_manager.capture_state(env)

                # Save snapshot if requested
                if self.config.snapshot_auto_save and self.config.snapshot_save_path:
                    snapshot_path = Path(self.config.snapshot_save_path)
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.save(str(snapshot_path))
                    logger.info(f"Saved snapshot to: {snapshot_path}")

        logger.info(f"Task: {snapshot.task}")
        logger.info(f"Instruction: {snapshot.instruction}")

        # ====================================================================
        # STEP 2: Collect trajectories from multiple rollouts
        # ====================================================================
        logger.info("\n" + "="*70)
        logger.info(f"Collecting {self.config.num_rollouts} rollout trajectories")
        logger.info("="*70)

        collector = TrajectoryCollector(
            env=env,
            policy=policy,
            snapshot=snapshot,
            num_rollouts=self.config.num_rollouts,
            steering=steering
        )

        trajectories = collector.collect(
            max_steps=self.config.max_steps,
            show_progress=True
        )

        # Get statistics
        stats = collector.get_summary_statistics()
        logger.info("\nCollection Statistics:")
        logger.info(f"  Total trajectory points: {stats['total_points']}")
        logger.info(f"  Success rate: {stats['success_rate']:.1%}")
        logger.info(f"  Average steps per rollout: {stats['avg_steps']:.1f} ± {stats['std_steps']:.1f}")
        logger.info(f"  Spatial extent:")
        logger.info(f"    X: [{stats['spatial_extent']['x_min']:.3f}, {stats['spatial_extent']['x_max']:.3f}]")
        logger.info(f"    Y: [{stats['spatial_extent']['y_min']:.3f}, {stats['spatial_extent']['y_max']:.3f}]")
        logger.info(f"    Z: [{stats['spatial_extent']['z_min']:.3f}, {stats['spatial_extent']['z_max']:.3f}]")

        # Verify same initial state
        same_initial_state = collector.verify_same_initial_state()
        if same_initial_state:
            logger.info("✓ Verified: All rollouts started from the same initial state")
        else:
            logger.warning("⚠ Warning: Rollouts may have started from different initial states")

        # ====================================================================
        # STEP 3: Generate visualizations
        # ====================================================================
        logger.info("\n" + "="*70)
        logger.info("Generating Plotly visualizations")
        logger.info("="*70)

        visualizer = TrajectoryVisualizer(trajectories)
        plot_options = self.config.plot_options
        saved_files = []

        for mode in self.config.modes:
            try:
                if mode == 'scatter':
                    logger.info("Generating 3D scatter plot...")
                    fig = visualizer.plot_3d_scatter(
                        point_size=plot_options.get('point_size', 3),
                        opacity=plot_options.get('opacity', 0.7),
                        colormap=plot_options.get('colormap', 'tab10')
                    )
                    path = self.output_dir / 'scatter_plot.html'
                    visualizer.save_html(fig, str(path))
                    saved_files.append(str(path))
                    logger.info(f"  Saved: {path}")

                elif mode == 'lines':
                    logger.info("Generating 3D line plot...")
                    fig = visualizer.plot_3d_lines(
                        line_width=plot_options.get('line_width', 2),
                        opacity=plot_options.get('opacity', 0.7),
                        colormap=plot_options.get('colormap', 'tab10')
                    )
                    path = self.output_dir / 'lines_plot.html'
                    visualizer.save_html(fig, str(path))
                    saved_files.append(str(path))
                    logger.info(f"  Saved: {path}")

                elif mode == 'combined':
                    logger.info("Generating combined scatter + line plot...")
                    fig = visualizer.plot_combined(
                        point_size=plot_options.get('point_size', 2),
                        line_width=plot_options.get('line_width', 2),
                        opacity=plot_options.get('opacity', 0.6),
                        colormap=plot_options.get('colormap', 'tab10')
                    )
                    path = self.output_dir / 'combined_plot.html'
                    visualizer.save_html(fig, str(path))
                    saved_files.append(str(path))
                    logger.info(f"  Saved: {path}")

                elif mode == 'success_comparison':
                    logger.info("Generating success/failure comparison plot...")
                    fig = visualizer.plot_success_comparison(
                        line_width=plot_options.get('line_width', 2)
                    )
                    path = self.output_dir / 'success_comparison.html'
                    visualizer.save_html(fig, str(path))
                    saved_files.append(str(path))
                    logger.info(f"  Saved: {path}")

                else:
                    logger.warning(f"Unknown visualization mode: {mode}")

            except Exception as e:
                logger.error(f"Error generating {mode} visualization: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # ====================================================================
        # Return results
        # ====================================================================
        logger.info("\n" + "="*70)
        logger.info("PLOTLY VISUALIZATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Generated {len(saved_files)} visualizations:")
        for file_path in saved_files:
            logger.info(f"  - {file_path}")
        logger.info(f"\nSuccess rate: {stats['success_rate']:.1%} ({collector.success_count}/{self.config.num_rollouts})")
        logger.info("="*70)

        return {
            'statistics': stats,
            'saved_files': saved_files,
            'success_count': collector.success_count,
            'total_rollouts': self.config.num_rollouts,
            'snapshot': snapshot
        }
