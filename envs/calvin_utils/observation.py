"""Observation processing for CALVIN environment.

Handles conversion from CALVIN's native observation format to
standardized point clouds and robot state.
"""

import logging
from typing import Dict, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)

# CALVIN camera intrinsics (from CALVIN codebase)
# Static camera: 200x200 resolution
CALVIN_STATIC_INTRINSICS = {
    'fx': 67.500,  # Focal length x
    'fy': 67.500,  # Focal length y
    'cx': 80.0,    # Principal point x (half of 160 after crop)
    'cy': 80.0,    # Principal point y
    'width': 160,
    'height': 160,
}

# Gripper camera: 84x84 resolution
CALVIN_GRIPPER_INTRINSICS = {
    'fx': 28.350,
    'fy': 28.350,
    'cx': 34.0,    # Half of 68 after crop
    'cy': 34.0,
    'width': 68,
    'height': 68,
}


def deproject_depth(depth: np.ndarray, intrinsics: Dict) -> np.ndarray:
    """
    Deproject depth image to 3D point cloud using camera intrinsics.

    Args:
        depth: (H, W) depth image in meters
        intrinsics: Dictionary with fx, fy, cx, cy, width, height

    Returns:
        (N, 3) point cloud array where N = H * W
    """
    height, width = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # Create pixel coordinate grids
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # Deproject to 3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack to (H, W, 3) and reshape to (N, 3)
    points_3d = np.stack([x, y, z], axis=-1)
    points_3d = points_3d.reshape(-1, 3)

    # Filter out invalid points (zero depth or too far)
    valid_mask = (points_3d[:, 2] > 0.01) & (points_3d[:, 2] < 2.0)  # 1cm to 2m range
    points_3d = points_3d[valid_mask]

    return points_3d


def process_calvin_obs(calvin_obs: Dict, num_points: int = 2048) -> Dict:
    """
    Process CALVIN native observation to standardized format.

    CALVIN provides observations as:
    - 'rgb_obs': Dict with 'rgb_static' and 'rgb_gripper' images
    - 'depth_obs': Dict with 'depth_static' and 'depth_gripper' depth maps
    - 'robot_obs': (15,) array with [tcp_pos(3), tcp_orient(3), gripper_width(1), arm_joints(7), gripper_action(1)]

    Args:
        calvin_obs: Native CALVIN observation dictionary
        num_points: Target number of points after downsampling

    Returns:
        Dictionary containing:
            - 'point_cloud': (num_points, 3) point cloud
            - 'robot_obs': (15,) robot proprioception
            - 'ee_pose': (7,) end-effector pose [x, y, z, euler_x, euler_y, euler_z, gripper_width]
            - 'rgb_static': (H, W, 3) static camera RGB image
    """
    try:
        # Extract observations
        rgb_obs = calvin_obs.get('rgb_obs', {})
        depth_obs = calvin_obs.get('depth_obs', {})
        robot_obs = calvin_obs.get('robot_obs', np.zeros(15))

        # Get RGB and depth from static camera
        rgb_static = rgb_obs.get('rgb_static', None)
        depth_static = depth_obs.get('depth_static', None)

        # Option 1: If depth is available, generate point cloud from RGB-D
        if depth_static is not None and depth_static.size > 0:
            # Deproject to 3D
            pcd_static = deproject_depth(depth_static, CALVIN_STATIC_INTRINSICS)

            # Optional: include gripper camera points
            depth_gripper = depth_obs.get('depth_gripper', None)
            if depth_gripper is not None and depth_gripper.size > 0:
                pcd_gripper = deproject_depth(depth_gripper, CALVIN_GRIPPER_INTRINSICS)
                # Concatenate point clouds
                point_cloud = np.vstack([pcd_static, pcd_gripper])
            else:
                point_cloud = pcd_static

            # Downsample using farthest point sampling
            if len(point_cloud) > num_points:
                point_cloud = sample_farthest_points_numpy(point_cloud, num_points)
            elif len(point_cloud) < num_points:
                # Pad with zeros if not enough points
                padding = np.zeros((num_points - len(point_cloud), 3))
                point_cloud = np.vstack([point_cloud, padding])

        # Option 2: Fallback to dummy point cloud if no depth available
        else:
            logger.warning("No depth data available, using dummy point cloud")
            point_cloud = np.random.randn(num_points, 3) * 0.1  # Small random cloud

        # Extract end-effector pose from robot_obs
        # robot_obs format: [tcp_pos(3), tcp_orient(3), gripper_width(1), arm_joints(7), gripper_action(1)]
        tcp_pos = robot_obs[:3]
        tcp_orient = robot_obs[3:6]
        gripper_width = robot_obs[6:7]
        ee_pose = np.concatenate([tcp_pos, tcp_orient, gripper_width])  # (7,)

        return {
            'point_cloud': point_cloud.astype(np.float32),
            'robot_obs': robot_obs.astype(np.float32),
            'ee_pose': ee_pose.astype(np.float32),
            'rgb_static': rgb_static if rgb_static is not None else np.zeros((160, 160, 3), dtype=np.uint8),
        }

    except Exception as e:
        logger.error(f"Error processing CALVIN observation: {e}")
        # Return safe fallback
        return {
            'point_cloud': np.random.randn(num_points, 3).astype(np.float32) * 0.1,
            'robot_obs': np.zeros(15, dtype=np.float32),
            'ee_pose': np.zeros(7, dtype=np.float32),
            'rgb_static': np.zeros((160, 160, 3), dtype=np.uint8),
        }


def sample_farthest_points_numpy(points: np.ndarray, K: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) in NumPy.

    Args:
        points: (N, 3) point cloud
        K: Number of points to sample

    Returns:
        (K, 3) sampled point cloud
    """
    N = points.shape[0]
    if N <= K:
        # Pad with last point if not enough
        if N < K:
            padding = np.repeat(points[[-1]], K - N, axis=0)
            return np.vstack([points, padding])
        return points

    # Initialize
    sampled_indices = np.zeros(K, dtype=int)
    distances = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(K):
        sampled_indices[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)

    return points[sampled_indices]
