"""Package CALVIN dataset for 3D Diffuser Actor training.

Converts raw CALVIN episodes into compressed .dat files containing
RGB images, per-pixel point clouds, gripper states, and trajectories.

Ported from 3d_diffuser_actor/data_preprocessing/package_calvin.py.

Usage:
    python -m training.policies.diffuser_actor.preprocessing.package_calvin \
        --root_dir /path/to/calvin/dataset/task_ABC_D \
        --save_path /path/to/output/packaged_ABC_D \
        --split training
"""

from typing import List, Optional
from pathlib import Path
import os
import pickle
import argparse

import cv2
import numpy as np
import torch
import blosc

from .calvin_utils import (
    keypoint_discovery,
    deproject,
    get_gripper_camera_view_matrix,
)


def make_env(dataset_path, split):
    from calvin_env.envs.play_table_env import get_env
    val_folder = Path(dataset_path) / f"{split}"
    env = get_env(val_folder, show_gui=False)
    return env


def process_datas(datas, mode, traj_len, execute_every, keyframe_inds):
    """Process collected episode data into the packaged format.

    Returns:
        state_dict: [frame_ids, rgb_pcd, action_tensors, camera_dicts,
                      gripper_tensors, trajectories, annotation_id]
    """
    # upscale gripper camera
    h, w = datas['static_rgb'][0].shape[:2]
    datas['gripper_rgb'] = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        for m in datas['gripper_rgb']
    ]
    datas['gripper_pcd'] = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        for m in datas['gripper_pcd']
    ]
    static_rgb = np.stack(datas['static_rgb'], axis=0)
    static_pcd = np.stack(datas['static_pcd'], axis=0)
    gripper_rgb = np.stack(datas['gripper_rgb'], axis=0)
    gripper_pcd = np.stack(datas['gripper_pcd'], axis=0)
    rgb = np.stack([static_rgb, gripper_rgb], axis=1)
    pcd = np.stack([static_pcd, gripper_pcd], axis=1)
    rgb_pcd = np.stack([rgb, pcd], axis=2)
    rgb_pcd = rgb_pcd.transpose(0, 1, 2, 5, 3, 4)
    rgb_pcd = torch.as_tensor(rgb_pcd, dtype=torch.float32)

    # prepare keypose actions
    keyframe_indices = torch.as_tensor(keyframe_inds)[None, :]
    gripper_indices = torch.arange(len(datas['proprios'])).view(-1, 1)
    action_indices = torch.argmax(
        (gripper_indices < keyframe_indices).float(), dim=1
    ).tolist()
    action_indices[-1] = len(keyframe_inds) - 1
    actions = [datas['proprios'][keyframe_inds[i]] for i in action_indices]
    action_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1) for a in actions
    ]

    camera_dicts = [{'front': (0, 0), 'wrist': (0, 0)}]

    gripper_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
        for a in datas['proprios']
    ]

    if mode == 'keypose':
        trajectories = []
        for i in range(len(action_indices)):
            target_frame = keyframe_inds[action_indices[i]]
            current_frame = i
            trajectories.append(
                torch.cat(
                    [
                        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
                        for a in datas['proprios'][current_frame:target_frame+1]
                    ],
                    dim=0
                )
            )
    else:
        trajectories = []
        for i in range(len(gripper_tensors)):
            traj = datas['proprios'][i:i+traj_len]
            if len(traj) < traj_len:
                traj += [traj[-1]] * (traj_len - len(traj))
            traj = [
                torch.as_tensor(a, dtype=torch.float32).view(1, -1)
                for a in traj
            ]
            traj = torch.cat(traj, dim=0)
            trajectories.append(traj)

    # Filter out data
    if mode == 'keypose':
        keyframe_inds = [0] + keyframe_inds[:-1].tolist()
        keyframe_indices = torch.as_tensor(keyframe_inds)
        rgb_pcd = torch.index_select(rgb_pcd, 0, keyframe_indices)
        action_tensors = [action_tensors[i] for i in keyframe_inds]
        gripper_tensors = [gripper_tensors[i] for i in keyframe_inds]
        trajectories = [trajectories[i] for i in keyframe_inds]
    else:
        rgb_pcd = rgb_pcd[:-1]
        action_tensors = action_tensors[:-1]
        gripper_tensors = gripper_tensors[:-1]
        trajectories = trajectories[:-1]

        rgb_pcd = rgb_pcd[::execute_every]
        action_tensors = action_tensors[::execute_every]
        gripper_tensors = gripper_tensors[::execute_every]
        trajectories = trajectories[::execute_every]

    frame_ids = [i for i in range(len(rgb_pcd))]

    state_dict = [
        frame_ids,
        rgb_pcd,
        action_tensors,
        camera_dicts,
        gripper_tensors,
        trajectories,
        datas['annotation_id']
    ]

    return state_dict


def load_episode(env, root_dir, split, episode, datas, ann_id):
    """Load a single CALVIN episode and extract RGB, PCD, and gripper data."""
    data = np.load(f'{root_dir}/{split}/{episode}')

    rgb_static = data['rgb_static']
    rgb_gripper = data['rgb_gripper']
    depth_static = data['depth_static']
    depth_gripper = data['depth_gripper']

    env.reset(robot_obs=data['robot_obs'], scene_obs=data['scene_obs'])
    static_cam = env.cameras[0]
    gripper_cam = env.cameras[1]
    gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)

    static_pcd = deproject(
        static_cam, depth_static,
        homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    static_pcd = np.reshape(
        static_pcd, (depth_static.shape[0], depth_static.shape[1], 3)
    )
    gripper_pcd = deproject(
        gripper_cam, depth_gripper,
        homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    gripper_pcd = np.reshape(
        gripper_pcd, (depth_gripper.shape[0], depth_gripper.shape[1], 3)
    )

    # map RGB to [-1, 1]
    rgb_static = rgb_static / 255. * 2 - 1
    rgb_gripper = rgb_gripper / 255. * 2 - 1

    # Map gripper openess to [0, 1]
    proprio = np.concatenate([
        data['robot_obs'][:3],
        data['robot_obs'][3:6],
        (data['robot_obs'][[-1]] > 0).astype(np.float32)
    ], axis=-1)

    datas['static_pcd'].append(static_pcd)
    datas['static_rgb'].append(rgb_static)
    datas['gripper_pcd'].append(gripper_pcd)
    datas['gripper_rgb'].append(rgb_gripper)
    datas['proprios'].append(proprio)
    datas['annotation_id'].append(ann_id)


def init_datas():
    return {
        'static_pcd': [],
        'static_rgb': [],
        'gripper_pcd': [],
        'gripper_rgb': [],
        'proprios': [],
        'annotation_id': []
    }


def main(split, root_dir, save_path, mode='keypose', traj_len=16,
         execute_every=4, tasks=None):
    """Package CALVIN episodes into compressed .dat files."""
    annotations = np.load(
        f'{root_dir}/{split}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()
    env = make_env(root_dir, split)

    for anno_ind, (start_id, end_id) in enumerate(annotations['info']['indx']):
        len_anno = len(annotations['info']['indx'])
        if tasks is not None and annotations['language']['task'][anno_ind] not in tasks:
            continue
        print(f'Processing {anno_ind}/{len_anno}, start_id:{start_id}, end_id:{end_id}')
        datas = init_datas()
        for ep_id in range(start_id, end_id + 1):
            episode = 'episode_{:07d}.npz'.format(ep_id)
            episode_path = f'{root_dir}/{split}/{episode}'
            if not os.path.isfile(episode_path):
                continue
            load_episode(env, root_dir, split, episode, datas, anno_ind)
        if len(datas['proprios']) < 2:
            print(f'  Skipping ann {anno_ind}: too few episodes ({len(datas["proprios"])})')
            continue

        _, keyframe_inds = keypoint_discovery(datas['proprios'])

        state_dict = process_datas(
            datas, mode, traj_len, execute_every, keyframe_inds
        )

        # Determine scene
        if split == 'training':
            scene_info = np.load(
                f'{root_dir}/training/scene_info.npy',
                allow_pickle=True
            ).item()
            if ("calvin_scene_B" in scene_info and
                start_id <= scene_info["calvin_scene_B"][1]):
                scene = "B"
            elif ("calvin_scene_C" in scene_info and
                  start_id <= scene_info["calvin_scene_C"][1]):
                scene = "C"
            elif ("calvin_scene_A" in scene_info and
                  start_id <= scene_info["calvin_scene_A"][1]):
                scene = "A"
            else:
                scene = "D"
        else:
            scene = 'D'

        ep_save_path = f'{save_path}/{split}/{scene}+0/ann_{anno_ind}.dat'
        os.makedirs(os.path.dirname(ep_save_path), exist_ok=True)
        with open(ep_save_path, "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package CALVIN for 3D Diffuser Actor")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to CALVIN dataset root (e.g., calvin/dataset/task_ABC_D)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output directory for packaged data")
    parser.add_argument("--split", type=str, default="training",
                        choices=["training", "validation"])
    parser.add_argument("--mode", type=str, default="keypose",
                        choices=["keypose", "close_loop"])
    parser.add_argument("--traj_len", type=int, default=16)
    parser.add_argument("--execute_every", type=int, default=4)
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()
    main(args.split, args.root_dir, args.save_path, args.mode,
         args.traj_len, args.execute_every, args.tasks)
