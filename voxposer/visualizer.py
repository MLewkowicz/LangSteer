"""Plotly-based 3D visualization for value maps overlaid on CALVIN scenes.

Ported from VoxPoser/src/visualizers.py with adaptations for CALVIN.
"""

import datetime
import logging
import os
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go

from voxposer.utils import Observation
from voxposer.value_map import ValueMap

logger = logging.getLogger(__name__)


# Quality presets: (downsample_ratio, max_scene_points, opacity, surface_count)
QUALITY_PRESETS = {
    'low': (4, 150_000, 0.12, 10),
    'medium': (2, 300_000, 0.06, 30),
    'high': (1, 500_000, 0.042, 50),
    'best': (1, 500_000, 0.03, 100),
}


class ValueMapVisualizer:
    """Interactive 3D visualization of value maps and CALVIN scene point clouds.

    Renders affordance/avoidance maps as Plotly Volume isosurfaces overlaid
    on the scene point cloud, with optional markers for gripper position
    and target voxels. Saves interactive HTML files.
    """

    def __init__(self, config: dict):
        self.save_dir = config.get('visualization_save_dir')
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        quality = config.get('visualization_quality', 'low')
        if quality not in QUALITY_PRESETS:
            logger.warning(f"Unknown quality '{quality}', using 'low'")
            quality = 'low'

        ds, max_pts, opacity, surface_count = QUALITY_PRESETS[quality]
        self.downsample_ratio = ds
        self.max_scene_points = max_pts
        self.costmap_opacity = opacity
        self.costmap_surface_count = surface_count

        self._scene_points: Optional[np.ndarray] = None
        self._scene_colors: Optional[np.ndarray] = None

    def update_scene_points(self, points: np.ndarray,
                            colors: Optional[np.ndarray] = None):
        """Set the scene point cloud for visualization overlay.

        Args:
            points: (N, 3) world-frame point cloud
            colors: (N, 3) uint8 RGB colors, or None for height-based coloring
        """
        self._scene_points = points.astype(np.float32)
        self._scene_colors = colors

    def visualize(self, value_map: ValueMap,
                  ee_pos_world: Optional[np.ndarray] = None,
                  objects: Optional[List[Observation]] = None,
                  show: bool = False, save: bool = True,
                  filename: Optional[str] = None) -> go.Figure:
        """Render value map visualization.

        Args:
            value_map: The ValueMap to visualize
            ee_pos_world: (3,) end-effector world position, or None
            objects: List of detected object Observations to render as bounding boxes
            show: Whether to open in browser
            save: Whether to save HTML to disk
            filename: Custom filename stem (without extension), or None for timestamp

        Returns:
            Plotly Figure object
        """
        bounds_min = value_map.workspace_bounds_min
        bounds_max = value_map.workspace_bounds_max
        map_size = value_map.map_size

        # Add 15% padding for plot bounds
        padding = 0.15 * (bounds_max - bounds_min)
        plot_min = bounds_min - padding
        plot_max = bounds_max + padding

        fig_data = []

        # 1. Render affordance map as 3D volume isosurface
        aff = value_map.affordance
        if aff.max() > 0:
            ds = self.downsample_ratio
            aff_ds = aff[::ds, ::ds, ::ds]
            skip = (bounds_max - bounds_min) / (map_size / ds)
            x, y, z = np.mgrid[
                bounds_min[0]:bounds_max[0]:skip[0],
                bounds_min[1]:bounds_max[1]:skip[1],
                bounds_min[2]:bounds_max[2]:skip[2],
            ]
            # Trim to match downsampled grid shape
            gs = aff_ds.shape
            x = x[:gs[0], :gs[1], :gs[2]]
            y = y[:gs[0], :gs[1], :gs[2]]
            z = z[:gs[0], :gs[1], :gs[2]]

            fig_data.append(go.Volume(
                x=x.flatten(), y=y.flatten(), z=z.flatten(),
                value=aff_ds.flatten(),
                isomin=0.01, isomax=float(aff_ds.max()),
                opacity=self.costmap_opacity,
                surface_count=self.costmap_surface_count,
                colorscale='Hot',
                showscale=False,
                name='affordance',
            ))

        # 2. Render avoidance map (if present) with different colorscale
        if value_map.avoidance is not None and value_map.avoidance.max() > 0:
            avoid = value_map.avoidance
            ds = self.downsample_ratio
            avoid_ds = avoid[::ds, ::ds, ::ds]
            skip = (bounds_max - bounds_min) / (map_size / ds)
            x, y, z = np.mgrid[
                bounds_min[0]:bounds_max[0]:skip[0],
                bounds_min[1]:bounds_max[1]:skip[1],
                bounds_min[2]:bounds_max[2]:skip[2],
            ]
            gs = avoid_ds.shape
            x = x[:gs[0], :gs[1], :gs[2]]
            y = y[:gs[0], :gs[1], :gs[2]]
            z = z[:gs[0], :gs[1], :gs[2]]

            fig_data.append(go.Volume(
                x=x.flatten(), y=y.flatten(), z=z.flatten(),
                value=avoid_ds.flatten(),
                isomin=0.01, isomax=float(avoid_ds.max()),
                opacity=self.costmap_opacity * 0.7,
                surface_count=self.costmap_surface_count,
                colorscale='Blues',
                showscale=False,
                name='avoidance',
            ))

        # 3. Mark original LLM-set target voxels as green dots (pre-smooth sparse mask)
        raw_aff = getattr(value_map, '_raw_affordance', None)
        if raw_aff is not None and raw_aff.max() > 0:
            target_voxels = np.argwhere(raw_aff > 0)
        elif aff.max() > 0:
            target_voxels = np.argwhere(aff >= aff.max() * 0.95)
        else:
            target_voxels = np.empty((0, 3), dtype=np.int32)
        if len(target_voxels) > 0:
                # Convert to world coordinates
                from voxposer.calvin_interface import voxel2pc
                target_world = voxel2pc(
                    target_voxels.astype(np.float32),
                    bounds_min, bounds_max, map_size
                )
                fig_data.append(go.Scatter3d(
                    x=target_world[:, 0],
                    y=target_world[:, 1],
                    z=target_world[:, 2],
                    mode='markers',
                    marker=dict(size=4, color='green', opacity=0.8),
                    name='target',
                ))

        # 4. Mark gripper position as blue sphere
        if ee_pos_world is not None:
            fig_data.append(go.Scatter3d(
                x=[ee_pos_world[0]],
                y=[ee_pos_world[1]],
                z=[ee_pos_world[2]],
                mode='markers',
                marker=dict(size=8, color='blue'),
                name='gripper',
            ))

        # 5. Overlay scene point cloud
        if self._scene_points is not None:
            pts = self._scene_points
            colors = self._scene_colors

            # Subsample if too many points
            if pts.shape[0] > self.max_scene_points:
                idx = np.random.choice(
                    pts.shape[0], self.max_scene_points, replace=False
                )
                pts = pts[idx]
                if colors is not None:
                    colors = colors[idx]

            if colors is not None:
                color_vals = colors.astype(np.float32) / 255.0
                # Convert to plotly color strings
                color_strs = [
                    f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                    for r, g, b in color_vals
                ]
            else:
                color_strs = pts[:, 2].tolist()  # Height-based coloring

            fig_data.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=color_strs, opacity=0.6),
                name='scene',
            ))

        # 6. Render object bounding boxes as wireframes with labels
        if objects is not None:
            from voxposer.calvin_interface import voxel2pc
            # Color map for object categories
            _OBJ_COLORS = {
                'red': 'red', 'blue': 'royalblue', 'pink': 'hotpink',
                'drawer': 'orange', 'slider': 'orange', 'lightbulb': 'gold',
                'light_switch': 'gold', 'switch': 'gold', 'led': 'gold',
                'button': 'orange', 'table': 'gray',
            }
            for obj in objects:
                name = obj.get('name', '?')
                aabb = obj.get('aabb')
                pos_world = obj.get('_position_world')
                if aabb is None or pos_world is None:
                    continue
                # Convert AABB from voxel to world coords
                aabb_world = voxel2pc(
                    np.array(aabb, dtype=np.float32),
                    value_map.workspace_bounds_min,
                    value_map.workspace_bounds_max,
                    value_map.map_size,
                )
                bmin, bmax = aabb_world[0], aabb_world[1]
                # Pick color
                color = 'orange'
                for key, col in _OBJ_COLORS.items():
                    if key in name.lower():
                        color = col
                        break
                # 12 edges of a box as line segments (with None separators)
                corners = np.array([
                    [bmin[0], bmin[1], bmin[2]],
                    [bmax[0], bmin[1], bmin[2]],
                    [bmax[0], bmax[1], bmin[2]],
                    [bmin[0], bmax[1], bmin[2]],
                    [bmin[0], bmin[1], bmax[2]],
                    [bmax[0], bmin[1], bmax[2]],
                    [bmax[0], bmax[1], bmax[2]],
                    [bmin[0], bmax[1], bmax[2]],
                ])
                # Edge connectivity: pairs of corner indices
                edges = [
                    (0,1),(1,2),(2,3),(3,0),  # bottom face
                    (4,5),(5,6),(6,7),(7,4),  # top face
                    (0,4),(1,5),(2,6),(3,7),  # verticals
                ]
                xs, ys, zs = [], [], []
                for i, j in edges:
                    xs.extend([corners[i,0], corners[j,0], None])
                    ys.extend([corners[i,1], corners[j,1], None])
                    zs.extend([corners[i,2], corners[j,2], None])
                fig_data.append(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=name,
                    showlegend=True,
                ))
                # Label at center
                fig_data.append(go.Scatter3d(
                    x=[pos_world[0]], y=[pos_world[1]], z=[pos_world[2]],
                    mode='text',
                    text=[name],
                    textfont=dict(size=10, color=color),
                    showlegend=False,
                ))

        fig = go.Figure(data=fig_data)

        # Set axis bounds and aspect ratio
        xyz_range = bounds_max - bounds_min
        max_range = xyz_range.max()
        scale = xyz_range / max_range

        fig.update_layout(
            title=dict(text=value_map.instruction, font=dict(size=14)),
            scene=dict(
                xaxis=dict(range=[plot_min[0], plot_max[0]], autorange=False,
                           showgrid=False, showticklabels=False, visible=False),
                yaxis=dict(range=[plot_min[1], plot_max[1]], autorange=False,
                           showgrid=False, showticklabels=False, visible=False),
                zaxis=dict(range=[plot_min[2], plot_max[2]], autorange=False,
                           showgrid=False, showticklabels=False, visible=False),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=scale[0], y=scale[1], z=scale[2]),
            template='none',
        )

        # Save
        if save and self.save_dir is not None:
            if filename:
                stem = filename
            else:
                now = datetime.datetime.now()
                stem = f'{now.hour:02d}_{now.minute:02d}_{now.second:02d}'
            save_path = os.path.join(self.save_dir, f'{stem}.html')
            latest_path = os.path.join(self.save_dir, 'latest.html')
            fig.write_html(save_path)
            fig.write_html(latest_path)
            logger.info(f"Saved visualization to {save_path}")

        if show:
            fig.show()

        return fig
