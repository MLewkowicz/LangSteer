"""Isaac Sim keypoint detection for VLS steering.

Isaac provides ``obs.depth['static']`` as a ``(H, W, 3) float32`` array of
world-space XYZ coordinates (when ``provide_pcd_images=True``).  Detecting a
keypoint therefore only requires knowing which pixel the object lives at — no
camera-intrinsics projection is needed.

The VLM is asked to identify pixel coordinates for each relevant object, and
those coordinates are looked up directly in the depth image.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _encode_image(image_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _build_localization_prompt(instruction: str, image_w: int, image_h: int) -> str:  # noqa: E501
    return f"""You are helping a robot identify object locations in an image.

Task: "{instruction}"

The image is {image_w}x{image_h} pixels (width x height).

Identify the pixel coordinates of the key objects the robot needs to interact with
to complete the task.
For each object, give a short name and its pixel coordinates (x from left, y from top).

Output ONLY a JSON list in this format (no explanation):
[
  {{"name": "object_name", "x": <pixel_x>, "y": <pixel_y>}},
  ...
]

Keep it to 2-5 objects. Focus on objects the robot must grasp or interact with."""


def detect(
    obs,
    instruction: str,
    openai_client,
    model: str = "gpt-4o",
    camera_key: str = "static",
) -> tuple[list[str], np.ndarray]:
    """Detect key object positions from an Isaac Sim observation.

    Uses the VLM to localise objects in the RGB image, then reads their 3-D
    world positions from the corresponding pixel in the pre-computed world-XYZ
    depth map.

    Args:
        obs: ``Observation`` from the Isaac environment.  Must have
            ``obs.rgb[camera_key]`` and ``obs.depth[camera_key]``.
        instruction: Language task description.
        openai_client: Initialised ``openai.OpenAI`` client.
        model: Vision-capable OpenAI model to use.
        camera_key: Which camera to use (``'static'`` or ``'gripper'``).

    Returns:
        Tuple of (names, positions) where positions is ``(K, 3) float32``
        in world frame.
    """
    image_rgb: np.ndarray = obs.rgb.get(camera_key)
    depth_xyz: Optional[np.ndarray] = (
        obs.depth.get(camera_key) if obs.depth is not None else None
    )

    if image_rgb is None:
        logger.warning(f"VLS keypoint_handler: no RGB image for camera '{camera_key}'")
        return [], np.zeros((0, 3), dtype=np.float32)

    H, W = image_rgb.shape[:2]
    prompt = _build_localization_prompt(instruction, W, H)
    img_b64 = _encode_image(image_rgb)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ],
        }
    ]

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        logger.debug(f"VLS keypoint VLM response: {raw}")
    except Exception as e:
        logger.error(f"VLS keypoint detection LLM call failed: {e}")
        return [], np.zeros((0, 3), dtype=np.float32)

    # Parse JSON from response
    import json

    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        logger.warning(f"VLS: could not parse keypoint JSON from: {raw}")
        return [], np.zeros((0, 3), dtype=np.float32)

    try:
        detections = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"VLS: JSON parse error: {e}")
        return [], np.zeros((0, 3), dtype=np.float32)

    names: list[str] = []
    positions: list[np.ndarray] = []

    for det in detections:
        name = det.get("name", "object")
        px = int(np.clip(det.get("x", 0), 0, W - 1))
        py = int(np.clip(det.get("y", 0), 0, H - 1))

        if depth_xyz is not None:
            pos_3d = depth_xyz[py, px].astype(np.float32)
            # Skip invalid depth (zeros or near-zero from sensor limits)
            if np.linalg.norm(pos_3d) < 1e-3:
                logger.warning(
                    f"VLS: invalid depth at ({px},{py}) for '{name}', skipping"
                )
                continue
        else:
            # No depth available — store pixel as placeholder, caller must handle
            logger.warning(f"VLS: no depth image, storing pixel coords for '{name}'")
            pos_3d = np.array([float(px), float(py), 0.0], dtype=np.float32)

        names.append(name)
        positions.append(pos_3d)
        logger.info(f"VLS keypoint '{name}': pixel=({px},{py}) world={pos_3d.tolist()}")

    if not positions:
        return [], np.zeros((0, 3), dtype=np.float32)

    return names, np.stack(positions, axis=0)


def overlay_keypoints(
    image_rgb: np.ndarray, names: list[str], positions_2d: np.ndarray
) -> np.ndarray:
    """Draw numbered keypoint markers on an RGB image (for VLM prompt images)."""
    out = image_rgb.copy()
    for i, (name, pos) in enumerate(zip(names, positions_2d)):
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            out,
            f"{i}:{name[:8]}",
            (x + 8, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
    return out
