"""Observation processing for Isaac Sim environment.

Converts Isaac Sim camera outputs (RGB, depth) and robot state into
the standardized Observation DTO.  Supports two modes:
  - Point cloud mode (for DP3): fused multi-camera depth → FPS-downsampled PCD
  - PCD image mode (for Diffuser Actor): per-pixel world-space XYZ images
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def deproject_depth(
    depth: np.ndarray,
    intrinsics: Dict[str, float],
    extrinsics: Optional[np.ndarray] = None,
    min_depth: float = 0.01,
    max_depth: float = 3.0,
) -> np.ndarray:
    """Deproject a depth image to a 3D point cloud.

    Args:
        depth: (H, W) depth in meters.
        intrinsics: Dict with fx, fy, cx, cy.
        extrinsics: Optional (4, 4) camera-to-world transform.
            If provided, points are returned in world frame.
        min_depth: Minimum valid depth.
        max_depth: Maximum valid depth.

    Returns:
        (N, 3) point cloud (valid points only).
    """
    h, w = depth.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (points[:, 2] > min_depth) & (points[:, 2] < max_depth)
    points = points[valid]

    # Transform to world frame if extrinsics provided
    if extrinsics is not None:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points = (R @ points.T).T + t

    return points


def deproject_depth_with_colors(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Dict[str, float],
    extrinsics: Optional[np.ndarray] = None,
    min_depth: float = 0.01,
    max_depth: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deproject depth to 3D with per-point RGB colors.

    Returns:
        points: (N, 3) XYZ in camera or world frame.
        colors: (N, 3) uint8 RGB.
    """
    h, w = depth.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    valid = (points[:, 2] > min_depth) & (points[:, 2] < max_depth)
    points = points[valid]
    colors = colors[valid]

    if extrinsics is not None:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points = (R @ points.T).T + t

    return points, colors


def deproject_depth_per_pixel(
    depth: np.ndarray,
    intrinsics: Dict[str, float],
    extrinsics: np.ndarray,
) -> np.ndarray:
    """Deproject depth to per-pixel world-space XYZ image (for Diffuser Actor).

    Args:
        depth: (H, W) depth in meters.
        intrinsics: Dict with fx, fy, cx, cy.
        extrinsics: (4, 4) camera-to-world transform.

    Returns:
        (H, W, 3) world-space XYZ per pixel.
    """
    h, w = depth.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # (H, W, 3) in camera frame
    pts_cam = np.stack([x, y, z], axis=-1)

    # Transform to world frame
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    pts_world = np.einsum("ij,hwj->hwi", R, pts_cam) + t

    return pts_world.astype(np.float32)


def sample_farthest_points(points: np.ndarray, k: int) -> np.ndarray:
    """Farthest Point Sampling (FPS) in NumPy.

    Args:
        points: (N, D) point cloud (typically D=3 or D=6 for XYZRGB).
        k: Number of points to sample.

    Returns:
        (k, D) sampled subset.
    """
    n = points.shape[0]
    if n <= k:
        if n < k:
            padding = np.repeat(points[[-1]], k - n, axis=0)
            return np.vstack([points, padding])
        return points

    sampled = np.zeros(k, dtype=int)
    distances = np.full(n, 1e10)
    farthest = np.random.randint(0, n)

    for i in range(k):
        sampled[i] = farthest
        centroid = points[farthest, :3]  # distance computed on XYZ only
        dist = np.sum((points[:, :3] - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)

    return points[sampled]


def process_isaac_obs(
    rgb: Dict[str, np.ndarray],
    depth: Dict[str, np.ndarray],
    intrinsics: Dict[str, Dict[str, float]],
    extrinsics: Dict[str, np.ndarray],
    robot_obs: np.ndarray,
    num_points: int = 2048,
) -> Dict[str, Any]:
    """Process Isaac Sim observations to DP3-compatible format (fused PCD).

    Args:
        rgb: Camera name → (H, W, 3) uint8 RGB.
        depth: Camera name → (H, W) float32 depth in meters.
        intrinsics: Camera name → dict with fx, fy, cx, cy.
        extrinsics: Camera name → (4, 4) cam-to-world transform.
        robot_obs: (15,) robot proprioception vector:
            [tcp_pos(3), tcp_euler(3), gripper_width(1), joints(7), grip_action(1)]
        num_points: Target point count after FPS.

    Returns:
        Dict with point_cloud, robot_obs, ee_pose, rgb_static.
    """
    all_points = []
    for cam_name in rgb:
        if cam_name not in depth:
            continue
        pts = deproject_depth(
            depth[cam_name],
            intrinsics[cam_name],
            extrinsics=extrinsics.get(cam_name),
        )
        all_points.append(pts)

    if all_points:
        point_cloud = np.vstack(all_points)
        if point_cloud.shape[0] > num_points:
            point_cloud = sample_farthest_points(point_cloud, num_points)
        elif point_cloud.shape[0] < num_points:
            pad = np.zeros((num_points - point_cloud.shape[0], 3))
            point_cloud = np.vstack([point_cloud, pad])
    else:
        logger.warning("No valid depth data — returning dummy point cloud")
        point_cloud = np.zeros((num_points, 3))

    # EE pose: [tcp_pos(3), tcp_euler(3), gripper_width(1)]
    ee_pose = robot_obs[:7].copy()

    # Use first camera as "static" for RGB
    static_cam = next(iter(rgb))
    rgb_static = rgb[static_cam]

    return {
        "point_cloud": point_cloud.astype(np.float32),
        "robot_obs": robot_obs.astype(np.float32),
        "ee_pose": ee_pose.astype(np.float32),
        "rgb_static": rgb_static,
    }


def prepare_pcd_images(
    rgb: Dict[str, np.ndarray],
    depth: Dict[str, np.ndarray],
    intrinsics: Dict[str, Dict[str, float]],
    extrinsics: Dict[str, np.ndarray],
    robot_obs: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> Dict[str, Any]:
    """Process Isaac Sim observations to Diffuser Actor format (per-pixel PCD).

    Returns dict with rgb_static, rgb_wrist (float [0,1]),
    pcd_static, pcd_wrist (H,W,3), robot_obs, ee_pose.
    """
    cam_names = list(rgb.keys())
    # Expect "static" and "wrist" cameras
    static_name = "static" if "static" in cam_names else cam_names[0]
    wrist_name = "wrist" if "wrist" in cam_names else (cam_names[1] if len(cam_names) > 1 else cam_names[0])

    # Per-pixel PCD images
    pcd_static = deproject_depth_per_pixel(
        depth[static_name], intrinsics[static_name], extrinsics[static_name]
    )
    pcd_wrist = deproject_depth_per_pixel(
        depth[wrist_name], intrinsics[wrist_name], extrinsics[wrist_name]
    )

    # Normalize RGB to [0, 1]
    rgb_static = rgb[static_name].astype(np.float32) / 255.0
    rgb_wrist = rgb[wrist_name].astype(np.float32) / 255.0

    # Resize wrist images to match static resolution
    h, w = target_size
    if rgb_wrist.shape[:2] != (h, w):
        rgb_wrist = (
            F.interpolate(
                torch.as_tensor(rgb_wrist).permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )
        pcd_wrist = (
            F.interpolate(
                torch.as_tensor(pcd_wrist).permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="nearest",
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )

    ee_pose = robot_obs[:7].copy()

    return {
        "rgb_static": rgb_static,
        "rgb_wrist": rgb_wrist,
        "pcd_static": pcd_static.astype(np.float32),
        "pcd_wrist": pcd_wrist.astype(np.float32),
        "robot_obs": robot_obs.astype(np.float32),
        "ee_pose": ee_pose.astype(np.float32),
    }
