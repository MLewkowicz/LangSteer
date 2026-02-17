"""
Plotly visualization utilities for 3D trajectories.

Extracted and simplified from 3D-Diffusion-Policy visualizer.
"""

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_3d_scatter_trace(
    points: np.ndarray,
    colors: List[str],
    size: int = 3,
    opacity: float = 0.7,
    name: Optional[str] = None
) -> go.Scatter3d:
    """
    Create a Plotly 3D scatter trace.

    Args:
        points: Array of shape (N, 3) containing x, y, z coordinates
        colors: List of N color strings in 'rgb(r,g,b)' format
        size: Marker size
        opacity: Marker opacity (0-1)
        name: Trace name for legend

    Returns:
        Plotly Scatter3d trace object
    """
    if points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
    if len(colors) != len(points):
        raise ValueError(f"Number of colors ({len(colors)}) must match number of points ({len(points)})")

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            opacity=opacity,
            color=colors
        ),
        name=name
    )


def create_3d_line_trace(
    points: np.ndarray,
    color: str,
    width: int = 2,
    name: Optional[str] = None
) -> go.Scatter3d:
    """
    Create a Plotly 3D line trace.

    Args:
        points: Array of shape (N, 3) containing x, y, z coordinates
        color: Line color as 'rgb(r,g,b)' string
        width: Line width
        name: Trace name for legend

    Returns:
        Plotly Scatter3d trace object
    """
    if points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='lines',
        line=dict(
            color=color,
            width=width
        ),
        name=name
    )


def generate_colors_from_coordinates(points: np.ndarray) -> List[str]:
    """
    Generate RGB colors based on normalized 3D coordinates.

    Args:
        points: Array of shape (N, 3) containing x, y, z coordinates

    Returns:
        List of color strings in 'rgb(r,g,b)' format
    """
    # Normalize coordinates to [0, 1] range
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Avoid division by zero
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1.0

    normalized_coords = (points - min_coords) / range_coords

    try:
        # Use normalized coordinates as RGB values
        colors = [
            f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
            for r, g, b in normalized_coords
        ]
    except (ValueError, TypeError):
        # Fallback to cyan if there are NaN/Inf values
        logger.warning("Invalid values in coordinates, using default cyan color")
        colors = ['rgb(0,255,255)' for _ in range(len(points))]

    return colors


def generate_colormap_colors(n_colors: int, colormap: str = 'tab10') -> List[str]:
    """
    Generate N distinct colors from a matplotlib colormap.

    Args:
        n_colors: Number of colors to generate
        colormap: Matplotlib colormap name (e.g., 'tab10', 'Set3', 'viridis')

    Returns:
        List of color strings in 'rgb(r,g,b)' format
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # Get colormap
    cmap = cm.get_cmap(colormap)

    # Generate colors
    colors = []
    for i in range(n_colors):
        # For qualitative colormaps like tab10, use index directly
        if n_colors <= 10 and colormap == 'tab10':
            rgba = cmap(i)
        else:
            # For continuous colormaps, interpolate
            rgba = cmap(i / max(n_colors - 1, 1))

        # Convert to RGB string
        rgb = [int(c * 255) for c in rgba[:3]]
        colors.append(f'rgb({rgb[0]},{rgb[1]},{rgb[2]})')

    return colors


def create_figure_layout(
    title: Optional[str] = None,
    show_grid: bool = True,
    show_background: bool = False,
    axis_color: str = 'grey',
    background_color: str = 'white'
) -> go.Layout:
    """
    Create a standard layout for 3D plots.

    Args:
        title: Plot title
        show_grid: Whether to show axis grids
        show_background: Whether to show axis background
        axis_color: Color for axes and grid
        background_color: Background color

    Returns:
        Plotly Layout object
    """
    axis_config = dict(
        showbackground=show_background,
        showgrid=show_grid,
        showline=True,
        linecolor=axis_color,
        zerolinecolor=axis_color,
        zeroline=False,
        gridcolor=axis_color,
    )

    layout = go.Layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=30 if title else 0),
        scene=dict(
            xaxis=dict(**axis_config, title='X (m)'),
            yaxis=dict(**axis_config, title='Y (m)'),
            zaxis=dict(**axis_config, title='Z (m)'),
            bgcolor=background_color
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return layout


def save_plotly_html(fig: go.Figure, file_path: str) -> None:
    """
    Save Plotly figure as interactive HTML file.

    Args:
        fig: Plotly Figure object
        file_path: Output file path (.html)
    """
    pio.write_html(fig, file_path, auto_open=False)
    logger.info(f"Saved interactive visualization to {file_path}")


def save_plotly_image(fig: go.Figure, file_path: str, width: int = 800, height: int = 600) -> None:
    """
    Save Plotly figure as static image (PNG).

    Args:
        fig: Plotly Figure object
        file_path: Output file path (.png)
        width: Image width in pixels
        height: Image height in pixels

    Note:
        Requires kaleido package: pip install kaleido
    """
    try:
        fig.write_image(file_path, width=width, height=height)
        logger.info(f"Saved static image to {file_path}")
    except Exception as e:
        logger.warning(f"Could not save static image (kaleido not installed?): {e}")
