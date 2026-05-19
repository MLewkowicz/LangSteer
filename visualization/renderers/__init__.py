"""Visualization renderers for different output modes.

Concrete `Renderer` Protocol implementations dispatched by
`VisualizationManager`. Iter 2 strips the legacy plotly/matplotlib/pybullet
renderers; iter 3 renames `costmap_tk` → `live_costmap_tk` and slims
`camera_renderer` → `video_recorder`.
"""

from .camera_renderer import CameraRenderer
from .costmap_tk import LiveCostmapWindow
from .matplotlib_renderer import MatplotlibRenderer
from .plotly_renderer import PlotlyRenderer
from .pybullet_renderer import PyBulletRenderer
from .stage_html import StageHtmlRenderer

__all__ = [
    'CameraRenderer',
    'LiveCostmapWindow',
    'MatplotlibRenderer',
    'PlotlyRenderer',
    'PyBulletRenderer',
    'StageHtmlRenderer',
]
