"""CALVIN-specific utilities for 3D Diffuser Actor preprocessing.

Ported from 3d_diffuser_actor/utils/utils_with_calvin.py.
"""

import numpy as np
from scipy.signal import argrelextrema
import torch

from .pytorch3d_transforms import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
)


def get_eef_velocity_from_trajectories(trajectories):
    trajectories = np.stack([trajectories[0]] + trajectories, axis=0)
    velocities = trajectories[1:] - trajectories[:-1]

    V = np.linalg.norm(velocities[:, :3], axis=-1)
    W = np.linalg.norm(velocities[:, 3:6], axis=-1)

    velocities = np.concatenate(
        [velocities, [velocities[-1]]],
        axis=0
    )
    accelerations = velocities[1:] - velocities[:-1]

    A = np.linalg.norm(accelerations[:, :3], axis=-1)

    return V, W, A


def gripper_state_changed(trajectories):
    trajectories = np.stack(
        [trajectories[0]] + trajectories, axis=0
    )
    openess = trajectories[:, -1]
    changed = openess[:-1] != openess[1:]

    return np.where(changed)[0]


def keypoint_discovery(trajectories, scene_states=None, task=None,
                       buffer_size=5):
    """Determine waypoints from the trajectories.

    Args:
        trajectories: a list of 1-D np arrays.  Each array is
            7-dimensional (x, y, z, euler_x, euler_y, euler_z, opene).

    Returns:
        keyframes: list of trajectory arrays at waypoint indices
        keyframe_inds: Integer array of waypoint indices
    """
    V, W, A = get_eef_velocity_from_trajectories(trajectories)

    # waypoints are local minima of gripper movement
    _local_max_A = argrelextrema(A, np.greater)[0]
    topK = np.sort(A)[::-1][int(A.shape[0] * 0.2)]
    large_A = A[_local_max_A] >= topK
    _local_max_A = _local_max_A[large_A].tolist()

    local_max_A = [_local_max_A.pop(0)]
    for i in _local_max_A:
        if i - local_max_A[-1] >= buffer_size:
            local_max_A.append(i)

    # waypoints are frames with changing gripper states
    gripper_changed = gripper_state_changed(trajectories)
    one_frame_before_gripper_changed = (
        gripper_changed[gripper_changed > 1] - 1
    )

    # waypoints is the last pose in the trajectory
    last_frame = [len(trajectories) - 1]

    keyframe_inds = (
        local_max_A +
        gripper_changed.tolist() +
        one_frame_before_gripper_changed.tolist() +
        last_frame
    )
    keyframe_inds = np.unique(keyframe_inds)

    keyframes = [trajectories[i] for i in keyframe_inds]

    return keyframes, keyframe_inds


def get_gripper_camera_view_matrix(cam):
    """Get the view matrix for the gripper camera using PyBullet."""
    import pybullet as pb

    camera_ls = pb.getLinkState(
        bodyUniqueId=cam.robot_uid,
        linkIndex=cam.gripper_cam_link,
        physicsClientId=cam.cid
    )
    camera_pos, camera_orn = camera_ls[:2]
    cam_rot = pb.getMatrixFromQuaternion(camera_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    view_matrix = pb.computeViewMatrix(
        camera_pos, camera_pos + cam_rot_y, -cam_rot_z
    )
    return view_matrix


def deproject(cam, depth_img, homogeneous=False, sanity_check=False):
    """
    Deprojects a depth image to 3D world coordinates.

    Args:
        cam: Camera object with viewMatrix, height, width, fov attributes
        depth_img: (H, W) depth image
        homogeneous: if True, return homogeneous coordinates (4, N)
        sanity_check: if True, verify against camera's deproject function

    Returns:
        (3, N) or (4, N) world coordinates of deprojected points
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    T_world_cam = np.linalg.inv(np.array(cam.viewMatrix).reshape((4, 4)).T)
    z = depth_img[v, u]
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    x = (u - cam.width // 2) * z / foc
    y = -(v - cam.height // 2) * z / foc
    z = -z
    ones = np.ones_like(z)

    cam_pos = np.stack([x, y, z, ones], axis=0)
    world_pos = T_world_cam @ cam_pos

    if sanity_check:
        sample_inds = np.random.permutation(u.shape[0])[:2000]
        for ind in sample_inds:
            cam_world_pos = cam.deproject((u[ind], v[ind]), depth_img, homogeneous=True)
            assert np.abs(cam_world_pos - world_pos[:, ind]).max() <= 1e-3

    if not homogeneous:
        world_pos = world_pos[:3]

    return world_pos


def convert_rotation(rot):
    """Convert Euler angles to quaternion using pytorch3d conventions."""
    rot = torch.as_tensor(rot)
    mat = euler_angles_to_matrix(rot, "XYZ")
    quat = matrix_to_quaternion(mat)
    quat = quat.numpy()
    return quat


def to_relative_action(actions, robot_obs, max_pos=1.0, max_orn=1.0, clip=True):
    """Convert absolute action to relative.

    rel_pos = target_pos - base_pos          (world-frame translation delta)
    rel_R   = R_base.T @ R_target            (body-frame rotation delta)

    The rotation delta is returned as its *principal* Euler XYZ so the
    downstream convert_rotation() (Euler->matrix->quaternion) produces the
    same quaternion as matrix_to_quaternion(rel_R). This keeps the model
    target consistent with the inference-side composition
        R_world = R_base @ R_rel
    instead of pretending Euler components add like Cartesian deltas.
    """
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)

    rel_pos = actions[..., :3] - robot_obs[..., :3]
    if clip:
        rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
    else:
        rel_pos = rel_pos / max_pos

    target_e = torch.as_tensor(actions[..., 3:6], dtype=torch.float64)
    base_e = torch.as_tensor(robot_obs[..., 3:6], dtype=torch.float64)
    target_R = euler_angles_to_matrix(target_e, "XYZ")
    base_R = euler_angles_to_matrix(base_e, "XYZ")
    rel_R = base_R.transpose(-1, -2) @ target_R
    rel_orn = matrix_to_euler_angles(rel_R, "XYZ").numpy()

    if clip:
        rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
    else:
        rel_orn = rel_orn / max_orn

    gripper = actions[..., -1:]
    return np.concatenate([rel_pos, rel_orn, gripper])
