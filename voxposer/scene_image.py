"""Annotated overhead scene-image renderer for VLM grounding (Task 7).

Captures a high-res static-camera frame, projects each detected object's OBB
corners to pixel coordinates, draws the OBB edges + name label, and encodes
the result as JPEG bytes for the OpenAI multimodal API.

Annotation discipline (locked, team-lead 2026-05-19): the overlay carries
**static identity + geometry** only — OBB edges + canonical name labels.
State info (drawer open/closed, slider position, light on/off, gripper
state) is deliberately NOT overlaid; the VLM infers state from the raw
pixels. The renderer's signature enforces this: no state arg exists.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from PIL import Image, ImageDraw, ImageFont

from voxposer.calvin_interface import obb_world_corners

logger = logging.getLogger(__name__)


# OBB edge indices (matches the 8-corner layout of `obb_world_corners`).
_OBB_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
    (4, 5), (5, 7), (7, 6), (6, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

# Object name → RGB color (matches `LiveCostmapTkRenderer._OBJ_COLORS` —
# same palette so live tk + scene_image stay visually consistent).
_OBJ_COLORS = {
    "red": (255, 85, 85),
    "blue": (85, 153, 255),
    "pink": (255, 119, 204),
    "drawer": (255, 170, 51),
    "slider": (255, 170, 51),
    "lightbulb": (255, 214, 51),
    "light_switch": (255, 214, 51),
    "switch": (255, 214, 51),
    "led": (255, 214, 51),
    "button": (255, 170, 51),
    "table": (136, 136, 136),
}


def _color_for_object(name: str) -> tuple[int, int, int]:
    lname = (name or "").lower()
    for key, col in _OBJ_COLORS.items():
        if key in lname:
            return col
    return (255, 170, 51)


def _project_world_to_pixel(
    points_world: np.ndarray,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Project (N, 3) world points to (N, 2) pixel coords via PyBullet matrices.

    PyBullet's view+projection matrices already include the y-flip needed for
    image-space coords. The result has pixel x growing right, pixel y growing
    down.
    """
    n = points_world.shape[0]
    homo = np.concatenate([points_world.astype(np.float64), np.ones((n, 1))], axis=1)
    clip = (proj_matrix @ view_matrix @ homo.T).T  # (N, 4)
    # Perspective divide
    w = clip[:, 3]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = clip[:, :3] / w[:, None]  # (N, 3) in [-1, 1]
    # NDC → pixel. PyBullet's NDC y is +up; image y is +down → flip.
    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    return np.stack([px, py], axis=1)


def render_annotated_overhead(
    rgb_frame: np.ndarray,
    detections: list,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    *,
    jpeg_quality: int = 85,
    skip_names: Optional[set[str]] = None,
) -> bytes:
    """Overlay OBB edges + name labels on the captured frame; return JPEG bytes.

    Args:
        rgb_frame: (H, W, 3) uint8 RGB frame from `env.render_high_res_static`.
        detections: list of Observation dicts from `CalvinLMPInterface.get_all_detections()`.
            Each dict carries `name`, `obb_center_world`, `obb_size`,
            `obb_rotation`. Missing OBB fields → object skipped.
        view_matrix: (4, 4) PyBullet view matrix (column-major reshape).
        proj_matrix: (4, 4) PyBullet projection matrix (column-major reshape).
        jpeg_quality: encoder quality 1-100.
        skip_names: object names to skip from annotation (e.g. {"switch"} — the
            same-pose alias of `light_switch`).

    Returns:
        JPEG-encoded bytes, ready for OpenAI vision content.
    """
    if skip_names is None:
        skip_names = {"switch"}  # hidden; light_switch overlays it

    height, width = rgb_frame.shape[:2]
    out = rgb_frame.copy()

    # Draw OBB wireframes via OpenCV (line drawing only — efficient even on
    # high-res frames). Pillow handles label text rendering after.
    for det in detections:
        name = det.get("name", "?")
        if name in skip_names:
            continue
        center = det.get("obb_center_world")
        size = det.get("obb_size")
        rot = det.get("obb_rotation")
        if center is None or size is None or rot is None:
            continue

        corners_world = obb_world_corners(
            np.asarray(center, dtype=np.float32),
            np.asarray(size, dtype=np.float32),
            np.asarray(rot, dtype=np.float32),
        )
        pix = _project_world_to_pixel(corners_world, view_matrix, proj_matrix, width, height)
        # Clamp to a generous off-screen range so very-out-of-frame OBBs don't
        # explode the line drawer.
        pix = np.clip(pix, -4 * width, 4 * width)
        pix_int = pix.astype(int)
        color = _color_for_object(name)

        if _CV2_AVAILABLE:
            # cv2.line writes raw tuple values into the array; since PIL
            # reads the array as RGB on JPEG encoding, pass RGB directly
            # (no BGR swap — that would render as the channel-flipped colour).
            for i, j in _OBB_EDGES:
                cv2.line(out, tuple(pix_int[i]), tuple(pix_int[j]), color, 2, lineType=cv2.LINE_AA)
        else:
            # Pillow fallback — slower but functional.
            pil = Image.fromarray(out)
            draw = ImageDraw.Draw(pil)
            for i, j in _OBB_EDGES:
                draw.line([tuple(pix_int[i]), tuple(pix_int[j])], fill=color, width=2)
            out = np.array(pil)

    # Pillow handles labels in one pass (cv2.putText looks rougher).
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(12, height // 35))
    except OSError:
        font = ImageFont.load_default()
    for det in detections:
        name = det.get("name", "?")
        if name in skip_names:
            continue
        center = det.get("obb_center_world")
        if center is None:
            continue
        center_pix = _project_world_to_pixel(
            np.asarray(center, dtype=np.float32)[None, :],
            view_matrix, proj_matrix, width, height,
        )[0]
        if not (0 <= center_pix[0] < width and 0 <= center_pix[1] < height):
            continue
        color = _color_for_object(name)
        # Black halo for legibility, then colored fill.
        x, y = float(center_pix[0]), float(center_pix[1])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), name, fill=(0, 0, 0), font=font, anchor="mm")
        draw.text((x, y), name, fill=color, font=font, anchor="mm")

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()
