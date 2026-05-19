"""Concrete `Renderer` Protocol implementations dispatched by `VisualizationManager`.

Iter 2 of Task 5: stripped the legacy plotly/matplotlib/pybullet renderers
and the trajectory collector. The three survivors are the artifacts the
user-facing brief asked us to keep: stage HTML, live tk window, MP4 video.

Iter 3 renames `costmap_tk` → `live_costmap_tk` and slims `camera_renderer`
→ `video_recorder`.
"""

from .camera_renderer import CameraRenderer
from .costmap_tk import LiveCostmapWindow
from .stage_html import StageHtmlRenderer

__all__ = [
    "CameraRenderer",
    "LiveCostmapWindow",
    "StageHtmlRenderer",
]
