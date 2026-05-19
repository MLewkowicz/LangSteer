"""Concrete `Renderer` Protocol implementations dispatched by `VisualizationManager`.

Three artifacts the user-facing brief asked us to keep:
    StageHtmlRenderer       — Plotly 3D HTML per stage activation.
    LiveCostmapTkRenderer   — live tk window mirroring the active value map.
    VideoRecorder           — MP4 recording for the static + gripper cameras.

Iter 3 of Task 5 renamed `costmap_tk` → `live_costmap_tk` and
`camera_renderer` → `video_recorder` (and renamed the contained classes to
match).
"""

from .live_costmap_tk import LiveCostmapTkRenderer
from .stage_html import StageHtmlRenderer
from .video_recorder import VideoRecorder

__all__ = [
    "LiveCostmapTkRenderer",
    "StageHtmlRenderer",
    "VideoRecorder",
]
