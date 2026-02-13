"""
Convert CALVIN dataset to DP3 Zarr format.
FIXED: Explicitly handles Zarr v3 API 'shape' requirement.
"""

import os
import sys
import shutil
import zarr
import tqdm
import numpy as np
import gc
import pybullet as p
from pathlib import Path
from termcolor import cprint
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R

# Import Blosc
try:
    from numcodecs import Blosc
except ImportError:
    cprint("Error: 'numcodecs' not found. Please run 'pip install numcodecs'", "red")
    raise

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from calvin_env.envs.play_table_env import get_env
    from utils.utils_with_calvin import deproject, get_gripper_camera_view_matrix
    from utils.visualize_point_clouds import visualize_point_clouds
except ImportError as e:
    cprint(f"Error importing modules: {e}", "red")
    cprint("Ensure your folder structure is: root/utils/utils_with_calvin.py", "red")
    raise

# --- Helper: Force State ---
def set_state_directly(env, robot_obs, scene_obs):
    """Manually forces PyBullet to the state recorded in the dataset."""
    pb_client = env.p if hasattr(env, 'p') else p
    
    # Robot State
    joint_angles = robot_obs[7:14]
    if hasattr(env.robot, 'joint_ids'):
        for i, angle in enumerate(joint_angles):
            if i < len(env.robot.joint_ids):
                pb_client.resetJointState(env.robot.robot_uid, env.robot.joint_ids[i], angle)

    # Gripper State
    gripper_width = robot_obs[6]
    if hasattr(env.robot, 'gripper_joint_ids'):
        for j_id in env.robot.gripper_joint_ids:
            pb_client.resetJointState(env.robot.robot_uid, j_id, gripper_width / 2.0)

    # Scene State
    scene_cfg = {
        "sliding_door": {"idx": 0, "type": "joint"},
        "drawer":       {"idx": 1, "type": "joint"},
        "button":       {"idx": 2, "type": "joint"},
        "switch":       {"idx": 3, "type": "joint"},
        "block_red":    {"idx": 6, "type": "6d"},
        "block_blue":   {"idx": 12, "type": "6d"},
        "block_pink":   {"idx": 18, "type": "6d"},
    }

    if hasattr(env, 'scene') and hasattr(env.scene, 'objects'):
        for name, obj in env.scene.objects.items():
            if name not in scene_cfg: continue
            cfg = scene_cfg[name]
            idx = cfg['idx']
            
            if cfg['type'] == 'joint':
                val = scene_obs[idx]
                if hasattr(obj, 'joint_ids') and len(obj.joint_ids) > 0:
                     pb_client.resetJointState(obj.uid, obj.joint_ids[0], val)
                elif hasattr(obj, 'cid'): 
                     pb_client.resetJointState(obj.cid, 0, val)
            elif cfg['type'] == '6d':
                pos = scene_obs[idx : idx+3]
                euler = scene_obs[idx+3 : idx+6]
                quat = R.from_euler('xyz', euler).as_quat()
                uid = obj.uid if hasattr(obj, 'uid') else obj.cid
                pb_client.resetBasePositionAndOrientation(uid, pos, quat)

# --- Sampling ---
def farthest_point_sampling(points: np.ndarray, num_points: int = 1024, use_cuda: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    N, D = points.shape
    if N < num_points:
        padding = np.tile(points[-1:], (num_points - N, 1))
        sampled_points = np.vstack([points, padding])
        indices = np.concatenate([np.arange(N), np.full(num_points - N, N-1)])
        return sampled_points, indices

    xyz = points[:, :3]
    centroids = np.zeros((num_points,), dtype=np.int64)
    distance = np.ones((N,), dtype=np.float64) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(num_points):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=0)
    sampled_points = points[centroids]
    return sampled_points, centroids

# --- Main Logic ---
def process_calvin_frame(env, rgb_static, rgb_gripper, depth_static, depth_gripper, 
                         robot_obs, scene_obs, use_cuda=False, visualize=False, frame_idx=0, visualize_save_dir=None):
    env.reset()
    set_state_directly(env, robot_obs, scene_obs)
    
    static_cam = env.cameras[0]
    gripper_cam = env.cameras[1]
    
    gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)
    
    # Deproject Full Images
    pcd_static_full = deproject(static_cam, depth_static, homogeneous=False) 
    pcd_gripper_full = deproject(gripper_cam, depth_gripper, homogeneous=False) 
    
    h_s, w_s = rgb_static.shape[:2]
    pcd_static_img = pcd_static_full.T.reshape(h_s, w_s, 3)
    
    h_g, w_g = rgb_gripper.shape[:2]
    pcd_gripper_img = pcd_gripper_full.T.reshape(h_g, w_g, 3)
    
    # Visualize deprojected point clouds if requested (before cropping)
    # if visualize:
    #     # Normalize RGB for visualization if needed
    #     static_rgb_vis = rgb_static.copy()
    #     gripper_rgb_vis = rgb_gripper.copy()
    #     if static_rgb_vis.min() < 0:  # If normalized to [-1, 1]
    #         static_rgb_vis = (static_rgb_vis + 1) / 2
    #     if static_rgb_vis.max() > 1.0:
    #         static_rgb_vis = static_rgb_vis / 255.0
    #     if gripper_rgb_vis.min() < 0:
    #         gripper_rgb_vis = (gripper_rgb_vis + 1) / 2
    #     if gripper_rgb_vis.max() > 1.0:
    #         gripper_rgb_vis = gripper_rgb_vis / 255.0
        
    #     visualize_point_clouds(
    #         pcd_static_img,
    #         pcd_gripper_img,
    #         static_rgb=static_rgb_vis,
    #         gripper_rgb=gripper_rgb_vis,
    #         static_depth=depth_static,
    #         gripper_depth=depth_gripper,
    #         frame_idx=frame_idx,
    #         title="Deprojected Point Clouds",
    #         save_path=visualize_save_dir,
    #         stage="before_cropping"
    #     )
    
    # Crop
    off_y_s = (h_s - 160) // 2
    off_x_s = (w_s - 160) // 2
    rgb_static_crop = rgb_static[off_y_s:off_y_s+160, off_x_s:off_x_s+160]
    pcd_static_crop = pcd_static_img[off_y_s:off_y_s+160, off_x_s:off_x_s+160]
    depth_static_crop = depth_static[off_y_s:off_y_s+160, off_x_s:off_x_s+160]

    off_y_g = (h_g - 68) // 2
    off_x_g = (w_g - 68) // 2
    rgb_gripper_crop = rgb_gripper[off_y_g:off_y_g+68, off_x_g:off_x_g+68]
    pcd_gripper_crop = pcd_gripper_img[off_y_g:off_y_g+68, off_x_g:off_x_g+68]
    
    # Visualize after cropping (before downsampling) if requested
    # if visualize:
    #     static_rgb_vis = rgb_static_crop.copy()
    #     gripper_rgb_vis = rgb_gripper_crop.copy()
    #     if static_rgb_vis.min() < 0:
    #         static_rgb_vis = (static_rgb_vis + 1) / 2
    #     if static_rgb_vis.max() > 1.0:
    #         static_rgb_vis = static_rgb_vis / 255.0
    #     if gripper_rgb_vis.min() < 0:
    #         gripper_rgb_vis = (gripper_rgb_vis + 1) / 2
    #     if gripper_rgb_vis.max() > 1.0:
    #         gripper_rgb_vis = gripper_rgb_vis / 255.0
        
    #     visualize_point_clouds(
    #         pcd_static_crop,
    #         pcd_gripper_crop,
    #         static_rgb=static_rgb_vis,
    #         gripper_rgb=gripper_rgb_vis,
    #         static_depth=depth_static_crop,
    #         gripper_depth=None,  # Gripper depth crop not stored
    #         frame_idx=frame_idx,
    #         title="Point Clouds After Cropping",
    #         save_path=visualize_save_dir,
    #         stage="after_cropping"
    #     )
    
    # Filter & Sample
    def sample_view(pcd, rgb):
        pts = pcd.reshape(-1, 3)
        colors = rgb.reshape(-1, 3)
        if colors.max() <= 1.0: colors *= 255.0
        
        valid = ~np.isnan(pts).any(axis=1) & ~np.isinf(pts).any(axis=1)
        pts = pts[valid]
        colors = colors[valid]
        
        sampled_pts, idx = farthest_point_sampling(pts, 1024, use_cuda)
        sampled_colors = colors[idx]
        return sampled_pts, sampled_colors

    s_pts, s_rgb = sample_view(pcd_static_crop, rgb_static_crop)
    g_pts, g_rgb = sample_view(pcd_gripper_crop, rgb_gripper_crop)
    
    # Visualize after downsampling if requested
    # if visualize:
    #     # Convert RGB colors to [0, 1] for visualization
    #     s_rgb_vis = s_rgb / 255.0
    #     g_rgb_vis = g_rgb / 255.0
        
    #     visualize_point_clouds(
    #         s_pts,
    #         g_pts,
    #         static_rgb=s_rgb_vis,
    #         gripper_rgb=g_rgb_vis,
    #         static_depth=None,
    #         gripper_depth=None,
    #         frame_idx=frame_idx,
    #         title="Point Clouds After Downsampling",
    #         save_path=visualize_save_dir,
    #         stage="after_downsampling"
    #     )
    
    points = np.vstack([s_pts, g_pts])
    colors = np.vstack([s_rgb, g_rgb]) / 255.0
    
    return {
        'img': rgb_static_crop, 
        'depth': depth_static_crop,
        'point_cloud': np.hstack([points, colors])
    }

def make_env(dataset_path, split):
    val_folder = Path(dataset_path) / split
    return get_env(val_folder, show_gui=False)

def convert_calvin_to_dp3(root_dir, save_path, split=None, tasks=None, use_cuda=False, overwrite=False, process_both_splits=True, visualize_samples=False, visualize_every_n=100, visualize_save_dir=None):
    # Hard Delete
    if os.path.exists(save_path):
        if overwrite:
            cprint(f"Deleting existing store: {save_path}", "yellow")
            try:
                shutil.rmtree(save_path)
            except NotADirectoryError:
                os.remove(save_path)
        else:
            cprint(f"File {save_path} exists. Use --overwrite.", "red")
            return

    splits = ['training', 'validation'] if process_both_splits else [split or 'training']
    
    img_arrays, pc_arrays, depth_arrays = [], [], []
    action_arrays, state_arrays, episode_ends = [], [], []
    total_count, training_end_idx = 0, 0
    
    try: p.disconnect()
    except: pass

    for cur_split in splits:
        cprint(f"Processing split: {cur_split}", "cyan")
        ann_path = Path(root_dir) / cur_split / "lang_annotations" / "auto_lang_ann.npy"
        if not ann_path.exists(): continue
            
        annotations = np.load(ann_path, allow_pickle=True).item()
        env = make_env(root_dir, cur_split)
        
        try:
            for i, (start_id, end_id) in enumerate(tqdm.tqdm(annotations['info']['indx'], desc=cur_split)):
                task_name = annotations['language']['task'][i]
                if tasks and task_name not in tasks: continue
                
                episode_start_val = total_count
                for ep_id in range(start_id, end_id + 1):
                    ep_path = Path(root_dir) / cur_split / f"episode_{ep_id:07d}.npz"
                    if not ep_path.exists(): continue
                    
                    data = np.load(ep_path)
                    try:
                        # Determine if we should visualize this frame
                        should_visualize = visualize_samples and (total_count % visualize_every_n == 0)
                        
                        res = process_calvin_frame(env, data['rgb_static'], data['rgb_gripper'],
                                                 data['depth_static'], data['depth_gripper'],
                                                 data['robot_obs'], data['scene_obs'], 
                                                 use_cuda=use_cuda,
                                                 visualize=should_visualize,
                                                 frame_idx=total_count,
                                                 visualize_save_dir=visualize_save_dir)
                        img_arrays.append(res['img'])
                        pc_arrays.append(res['point_cloud'])
                        depth_arrays.append(res['depth'])
                        action_arrays.append(data['rel_actions'])
                        state_arrays.append(data['robot_obs'])
                        total_count += 1
                    except Exception as e:
                        cprint(f"Error frame {ep_id}: {e}", "red")
                        if "Not connected" in str(e): raise e
                        continue
                if total_count > episode_start_val:
                    episode_ends.append(total_count)
        
        finally:
            # Destructor Safety
            try:
                env.close()
                if hasattr(env, 'p'): env.p = None
            except: pass
            del env
            gc.collect()
            try: p.disconnect()
            except: pass
        
        if cur_split == 'training': training_end_idx = len(episode_ends)

    if total_count == 0:
        cprint("No data processed.", "red")
        return

    cprint("Saving Zarr...", "yellow")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Use mode='w' and zarr_format=2 for compatibility
    root = zarr.open_group(save_path, mode='w', zarr_format=2)
    data_g = root.create_group('data')
    meta_g = root.create_group('meta')
    
    compressor = Blosc(cname='zstd', clevel=3, shuffle=1)
    
    def save_arr(name, arr, dtype):
        arr_np = np.stack(arr, axis=0)
        cprint(f"Saving {name}: {arr_np.shape}", "green")
        chunks = (100,) + arr_np.shape[1:]
        
        # FIX: Explicit shape and compressors list
        data_g.create_dataset(
            name, 
            data=arr_np, 
            shape=arr_np.shape, # Mandatory in Zarr v3
            chunks=chunks, 
            dtype=dtype, 
            compressors=[compressor], # List required in Zarr v3
            overwrite=True
        )
        del arr_np

    save_arr('img', img_arrays, 'uint8')
    save_arr('point_cloud', pc_arrays, 'float64')
    save_arr('depth', depth_arrays, 'float64')
    save_arr('action', action_arrays, 'float32')
    save_arr('state', state_arrays, 'float32')
    
    # Save meta
    meta_g.create_dataset(
        'episode_ends', 
        data=np.array(episode_ends, dtype='int64'), 
        shape=(len(episode_ends),), 
        chunks=(100,), 
        compressors=[compressor], 
        overwrite=True
    )
    
    if training_end_idx > 0:
        meta_g.create_dataset(
            'training_episode_count', 
            data=np.array([training_end_idx]), 
            shape=(1,), 
            dtype='int64', 
            overwrite=True
        )
                              
    cprint(f"Saved to {save_path}", "green")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--tasks', nargs='+', default=None)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--process_both_splits', action='store_true', default=True)
    parser.add_argument('--visualize_samples', action='store_true',
                        help='Visualize deprojected point clouds during processing')
    parser.add_argument('--visualize_every_n', type=int, default=100,
                        help='Visualize every Nth frame when --visualize_samples is enabled (default: 100)')
    parser.add_argument('--visualize_save_dir', type=str, default=None,
                        help='Directory to save visualization files (PLY point clouds, images). If None, tries to display interactively.')
    args = parser.parse_args()
    
    convert_calvin_to_dp3(
        args.root_dir, args.save_path, 
        split=args.split, tasks=args.tasks, 
        use_cuda=False, overwrite=args.overwrite,
        process_both_splits=args.process_both_splits,
        visualize_samples=args.visualize_samples,
        visualize_every_n=args.visualize_every_n,
        visualize_save_dir=args.visualize_save_dir
    )