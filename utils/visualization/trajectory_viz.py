"""
Trajectory visualization utilities using Plotly.

Creates interactive 3D visualizations of robot end-effector trajectories
from multiple rollouts.
"""

import numpy as np
import plotly.graph_objs as go
import logging
from typing import List, Optional
from pathlib import Path

from visualization.collectors.trajectory_collector import TrajectoryData
from utils.visualization import plotly_viz

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Creates Plotly visualizations of robot trajectories."""

    def __init__(self, trajectories: List[TrajectoryData]):
        """
        Initialize visualizer with collected trajectory data.

        Args:
            trajectories: List of trajectory data points from TrajectoryCollector
        """
        self.trajectories = trajectories
        self.num_rollouts = len(set(t.rollout_id for t in trajectories))

        logger.info(f"TrajectoryVisualizer initialized")
        logger.info(f"  Total points: {len(trajectories)}")
        logger.info(f"  Rollouts: {self.num_rollouts}")

    def _group_by_rollout(self) -> List[List[TrajectoryData]]:
        """
        Group trajectories by rollout ID.

        Returns:
            List of lists, each containing trajectories for one rollout
        """
        grouped = [[] for _ in range(self.num_rollouts)]
        for traj in self.trajectories:
            grouped[traj.rollout_id].append(traj)

        # Sort each rollout by timestep
        for rollout_trajs in grouped:
            rollout_trajs.sort(key=lambda t: t.timestep)

        return grouped

    def plot_3d_scatter(
        self,
        point_size: int = 3,
        opacity: float = 0.7,
        colormap: str = 'tab10'
    ) -> go.Figure:
        """
        Create 3D scatter plot of all end-effector positions.

        Each rollout is shown in a different color. Points are colored
        by rollout ID to visualize trajectory variance across rollouts.

        Args:
            point_size: Marker size
            opacity: Marker opacity (0-1)
            colormap: Matplotlib colormap name for rollout colors

        Returns:
            Plotly Figure object
        """
        logger.info("Generating 3D scatter plot")

        # Group trajectories by rollout
        grouped = self._group_by_rollout()

        # Generate colors for each rollout
        rollout_colors = plotly_viz.generate_colormap_colors(self.num_rollouts, colormap)

        # Create traces for each rollout
        traces = []
        for rollout_id, rollout_trajs in enumerate(grouped):
            if len(rollout_trajs) == 0:
                continue

            # Extract positions
            positions = np.array([t.ee_position for t in rollout_trajs])

            # Create color list (same color for all points in this rollout)
            colors = [rollout_colors[rollout_id]] * len(positions)

            # Create trace
            trace = plotly_viz.create_3d_scatter_trace(
                points=positions,
                colors=colors,
                size=point_size,
                opacity=opacity,
                name=f'Rollout {rollout_id}'
            )
            traces.append(trace)

        # Create figure with layout
        layout = plotly_viz.create_figure_layout(
            title='End-Effector Trajectories (Scatter)',
            show_grid=True,
            show_background=False
        )

        fig = go.Figure(data=traces, layout=layout)

        logger.info(f"Created scatter plot with {len(traces)} rollout traces")
        return fig

    def plot_3d_lines(
        self,
        line_width: int = 2,
        opacity: float = 0.8,
        colormap: str = 'tab10'
    ) -> go.Figure:
        """
        Create 3D line plot showing continuous trajectory paths.

        Each rollout is shown as a continuous line in a different color.
        This visualization makes it easy to see the temporal progression
        of the robot's movement.

        Args:
            line_width: Line width
            opacity: Line opacity (0-1)
            colormap: Matplotlib colormap name for rollout colors

        Returns:
            Plotly Figure object
        """
        logger.info("Generating 3D line plot")

        # Group trajectories by rollout
        grouped = self._group_by_rollout()

        # Generate colors for each rollout
        rollout_colors = plotly_viz.generate_colormap_colors(self.num_rollouts, colormap)

        # Create traces for each rollout
        traces = []
        for rollout_id, rollout_trajs in enumerate(grouped):
            if len(rollout_trajs) == 0:
                continue

            # Extract positions (already sorted by timestep)
            positions = np.array([t.ee_position for t in rollout_trajs])

            # Create line trace
            trace = plotly_viz.create_3d_line_trace(
                points=positions,
                color=rollout_colors[rollout_id],
                width=line_width,
                name=f'Rollout {rollout_id}'
            )

            # Note: opacity is not a valid attribute for 3D line traces in Plotly
            # The color string already includes opacity if needed

            traces.append(trace)

        # Create figure with layout
        layout = plotly_viz.create_figure_layout(
            title='End-Effector Trajectories (Lines)',
            show_grid=True,
            show_background=False
        )

        fig = go.Figure(data=traces, layout=layout)

        logger.info(f"Created line plot with {len(traces)} rollout traces")
        return fig

    def plot_combined(
        self,
        point_size: int = 2,
        line_width: int = 2,
        opacity: float = 0.6,
        colormap: str = 'tab10'
    ) -> go.Figure:
        """
        Create combined plot with both scatter points and lines.

        Shows the continuous path (lines) and individual timestep positions (points).

        Args:
            point_size: Marker size for scatter points
            line_width: Line width
            opacity: Opacity for both markers and lines
            colormap: Matplotlib colormap name

        Returns:
            Plotly Figure object
        """
        logger.info("Generating combined scatter + line plot")

        # Group trajectories by rollout
        grouped = self._group_by_rollout()

        # Generate colors for each rollout
        rollout_colors = plotly_viz.generate_colormap_colors(self.num_rollouts, colormap)

        # Create traces for each rollout (both line and scatter)
        traces = []
        for rollout_id, rollout_trajs in enumerate(grouped):
            if len(rollout_trajs) == 0:
                continue

            # Extract positions
            positions = np.array([t.ee_position for t in rollout_trajs])
            color = rollout_colors[rollout_id]

            # Create combined line+markers trace
            trace = go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='lines+markers',
                line=dict(color=color, width=line_width),
                marker=dict(size=point_size, color=color, opacity=opacity),
                name=f'Rollout {rollout_id}'
            )
            traces.append(trace)

        # Create figure with layout
        layout = plotly_viz.create_figure_layout(
            title='End-Effector Trajectories (Combined)',
            show_grid=True,
            show_background=False
        )

        fig = go.Figure(data=traces, layout=layout)

        logger.info(f"Created combined plot with {len(traces)} rollout traces")
        return fig

    def plot_success_comparison(
        self,
        line_width: int = 2,
        successful_color: str = 'rgb(0,200,0)',
        failed_color: str = 'rgb(200,0,0)'
    ) -> go.Figure:
        """
        Create plot comparing successful vs failed trajectories.

        Successful rollouts are shown in green, failed in red.

        Args:
            line_width: Line width
            successful_color: Color for successful trajectories (RGB string)
            failed_color: Color for failed trajectories (RGB string)

        Returns:
            Plotly Figure object
        """
        logger.info("Generating success/failure comparison plot")

        # Group trajectories by rollout
        grouped = self._group_by_rollout()

        # Create traces
        traces = []
        successful_count = 0
        failed_count = 0

        for rollout_id, rollout_trajs in enumerate(grouped):
            if len(rollout_trajs) == 0:
                continue

            # Extract positions
            positions = np.array([t.ee_position for t in rollout_trajs])

            # Check if rollout was successful (last trajectory point)
            was_successful = rollout_trajs[-1].reward > 0

            # Choose color based on success
            if was_successful:
                color = successful_color
                label = f'Rollout {rollout_id} (Success)'
                successful_count += 1
            else:
                color = failed_color
                label = f'Rollout {rollout_id} (Failed)'
                failed_count += 1

            # Create line trace
            trace = plotly_viz.create_3d_line_trace(
                points=positions,
                color=color,
                width=line_width,
                name=label
            )
            traces.append(trace)

        # Create figure with layout
        success_rate = successful_count / (successful_count + failed_count) if (successful_count + failed_count) > 0 else 0
        title = f'Success vs Failure ({success_rate:.1%} success rate)'

        layout = plotly_viz.create_figure_layout(
            title=title,
            show_grid=True,
            show_background=False
        )

        fig = go.Figure(data=traces, layout=layout)

        logger.info(f"Created comparison plot: {successful_count} successful, {failed_count} failed")
        return fig

    def save_html(self, fig: go.Figure, path: str) -> None:
        """
        Save Plotly figure as interactive HTML file.

        Args:
            fig: Plotly Figure object
            path: Output file path (.html)
        """
        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save figure
        plotly_viz.save_plotly_html(fig, path)

    def save_all_visualizations(
        self,
        output_dir: str,
        plot_options: Optional[dict] = None
    ) -> List[str]:
        """
        Generate and save all visualization modes.

        Args:
            output_dir: Directory to save HTML files
            plot_options: Optional dict with plotting parameters:
                - point_size: int
                - line_width: int
                - opacity: float
                - colormap: str

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Default plot options
        if plot_options is None:
            plot_options = {}

        point_size = plot_options.get('point_size', 3)
        line_width = plot_options.get('line_width', 2)
        opacity = plot_options.get('opacity', 0.7)
        colormap = plot_options.get('colormap', 'tab10')

        saved_files = []

        # Generate and save each visualization
        try:
            # Scatter plot
            fig_scatter = self.plot_3d_scatter(point_size, opacity, colormap)
            scatter_path = output_dir / 'scatter_plot.html'
            self.save_html(fig_scatter, str(scatter_path))
            saved_files.append(str(scatter_path))

            # Line plot
            fig_lines = self.plot_3d_lines(line_width, opacity, colormap)
            lines_path = output_dir / 'lines_plot.html'
            self.save_html(fig_lines, str(lines_path))
            saved_files.append(str(lines_path))

            # Combined plot
            fig_combined = self.plot_combined(point_size, line_width, opacity, colormap)
            combined_path = output_dir / 'combined_plot.html'
            self.save_html(fig_combined, str(combined_path))
            saved_files.append(str(combined_path))

            # Success comparison (if there are both successes and failures)
            has_successes = any(t.reward > 0 for t in self.trajectories)
            has_failures = any(t.reward == 0 for t in self.trajectories)
            if has_successes and has_failures:
                fig_comparison = self.plot_success_comparison(line_width)
                comparison_path = output_dir / 'success_comparison.html'
                self.save_html(fig_comparison, str(comparison_path))
                saved_files.append(str(comparison_path))

            logger.info(f"Saved {len(saved_files)} visualizations to {output_dir}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise

        return saved_files
