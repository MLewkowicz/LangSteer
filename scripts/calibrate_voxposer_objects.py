"""Calibrate VoxPoser hardcoded fixture positions against CALVIN PyBullet scene.

Loads the CALVIN environment, queries PyBullet for actual object AABBs,
and compares against the hardcoded positions in voxposer/calvin_interface.py.

Requires the CALVIN environment to be installed and configured.

Usage:
    uv run python scripts/calibrate_voxposer_objects.py
    uv run python scripts/calibrate_voxposer_objects.py --visualize
    uv run python scripts/calibrate_voxposer_objects.py --save-dir outputs/calibration
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)


def get_pybullet_client(env):
    """Extract the PyBullet physics client ID from the CALVIN environment."""
    import pybullet as p
    # Access via camera objects
    calvin_env = env._gym_env._env
    if hasattr(calvin_env, 'cameras') and len(calvin_env.cameras) > 0:
        cid = calvin_env.cameras[0].cid
    else:
        cid = 0  # default physics client
    return p, cid


def get_scene_objects(env):
    """Extract all scene objects from the CALVIN environment.

    Returns dict mapping object_name -> PyBullet body UID.
    """
    calvin_env = env._gym_env._env
    objects = {}

    # Try different scene access patterns
    scene = getattr(calvin_env, 'scene', None)
    if scene is None:
        # Some CALVIN versions store it differently
        for attr in ['_scene', 'play_table', '_play_table']:
            scene = getattr(calvin_env, attr, None)
            if scene is not None:
                break

    if scene is None:
        logger.warning("Could not find scene object on CALVIN env. "
                       "Trying to enumerate PyBullet bodies directly.")
        return objects

    # Collect objects from scene attributes
    for attr_name in ['fixed_objects', 'movable_objects']:
        obj_dict = getattr(scene, attr_name, {})
        if isinstance(obj_dict, dict):
            for name, obj in obj_dict.items():
                uid = getattr(obj, 'uid', None) or getattr(obj, 'body_uid', None)
                if uid is not None:
                    objects[name] = uid

    for attr_name in ['doors', 'buttons', 'switches', 'lights']:
        items = getattr(scene, attr_name, [])
        if isinstance(items, dict):
            items = list(items.values())
        elif not isinstance(items, (list, tuple)):
            items = [items] if items is not None else []
        for i, obj in enumerate(items):
            uid = getattr(obj, 'uid', None) or getattr(obj, 'body_uid', None)
            name = getattr(obj, 'name', f'{attr_name}_{i}')
            if uid is not None:
                objects[name] = uid

    return objects


def enumerate_all_bodies(p, cid):
    """List all PyBullet bodies in the simulation."""
    num_bodies = p.getNumBodies(physicsClientId=cid)
    bodies = []
    for i in range(num_bodies):
        info = p.getBodyInfo(i, physicsClientId=cid)
        name = info[1].decode('utf-8') if isinstance(info[1], bytes) else str(info[1])
        aabb_min, aabb_max = p.getAABB(i, physicsClientId=cid)
        center = np.array([(a + b) / 2 for a, b in zip(aabb_min, aabb_max)])
        size = np.array([b - a for a, b in zip(aabb_min, aabb_max)])
        bodies.append({
            'uid': i,
            'name': name,
            'aabb_min': np.array(aabb_min),
            'aabb_max': np.array(aabb_max),
            'center': center,
            'size': size,
        })
    return bodies


def render_segmentation(env, p, cid, save_path):
    """Render a segmentation image from the static camera and save it."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.error("matplotlib required for --visualize. Install with: uv add matplotlib")
        return

    calvin_env = env._gym_env._env
    cam = calvin_env.cameras[0]  # static camera

    # Render with segmentation
    width, height = cam.width, cam.height
    view_matrix = cam.viewMatrix
    proj_matrix = cam.projMatrix if hasattr(cam, 'projMatrix') else \
        p.computeProjectionMatrixFOV(cam.fov, width / height, cam.nearval, cam.farval,
                                     physicsClientId=cid)

    _, _, rgb, depth, seg = p.getCameraImage(
        width, height, view_matrix, proj_matrix,
        physicsClientId=cid
    )

    # Map UIDs to colors
    unique_uids = np.unique(seg)
    cmap = plt.cm.get_cmap('tab20', len(unique_uids))

    seg_colored = np.zeros((height, width, 3), dtype=np.uint8)
    uid_to_name = {}

    # Get body names
    for uid in unique_uids:
        if uid < 0:
            uid_to_name[uid] = 'background'
            continue
        try:
            info = p.getBodyInfo(uid, physicsClientId=cid)
            uid_to_name[uid] = info[1].decode('utf-8') if isinstance(info[1], bytes) else str(info[1])
        except Exception:
            uid_to_name[uid] = f'body_{uid}'

    patches = []
    for i, uid in enumerate(unique_uids):
        color = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        mask = seg == uid
        seg_colored[mask] = color
        patches.append(mpatches.Patch(
            color=cmap(i), label=f'{uid_to_name.get(uid, uid)} (uid={uid})'
        ))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(rgb[:, :, :3] if rgb.ndim == 3 else rgb)
    axes[0].set_title('RGB')
    axes[0].axis('off')

    axes[1].imshow(seg_colored)
    axes[1].set_title('Segmentation')
    axes[1].axis('off')

    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved segmentation image to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate VoxPoser object positions")
    parser.add_argument("--visualize", action="store_true",
                        help="Render segmentation overlay image")
    parser.add_argument("--save-dir", type=str, default="outputs/calibration")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Import our hardcoded positions
    from voxposer.calvin_interface import CALVIN_FIXTURES, BLOCK_SIZE

    # Initialize CALVIN environment
    logger.info("Initializing CALVIN environment...")
    import os
    from envs.calvin import CalvinEnvironment

    dataset_path = os.environ.get('CALVIN_DATASET_PATH', '')
    if not dataset_path:
        logger.error("CALVIN_DATASET_PATH not set. Export it first:\n"
                      "  export CALVIN_DATASET_PATH=/path/to/task_D_D")
        return

    env_cfg = {
        'name': 'calvin',
        'dataset_path': dataset_path,
        'split': 'validation',
        'use_gui': False,
    }

    try:
        env = CalvinEnvironment(env_cfg)
        obs = env.reset()
    except Exception as e:
        logger.error(f"Failed to initialize CALVIN environment: {e}")
        logger.info("Make sure CALVIN is installed and CALVIN_DATASET_PATH is set.")
        return

    # Get PyBullet client
    p, cid = get_pybullet_client(env)
    logger.info(f"PyBullet client ID: {cid}")

    # Enumerate all bodies
    logger.info("\n--- All PyBullet Bodies ---")
    bodies = enumerate_all_bodies(p, cid)
    for body in bodies:
        logger.info(f"  UID={body['uid']:3d} | {body['name']:30s} | "
                    f"center=[{body['center'][0]:7.3f}, {body['center'][1]:7.3f}, {body['center'][2]:7.3f}] | "
                    f"size=[{body['size'][0]:6.3f}, {body['size'][1]:6.3f}, {body['size'][2]:6.3f}]")

    # Get named scene objects
    scene_objects = get_scene_objects(env)
    if scene_objects:
        logger.info(f"\n--- Named Scene Objects ({len(scene_objects)}) ---")
        for name, uid in scene_objects.items():
            aabb_min, aabb_max = p.getAABB(uid, physicsClientId=cid)
            center = np.array([(a + b) / 2 for a, b in zip(aabb_min, aabb_max)])
            size = np.array([b - a for a, b in zip(aabb_min, aabb_max)])
            logger.info(f"  {name:30s} uid={uid:3d} | "
                        f"center=[{center[0]:7.3f}, {center[1]:7.3f}, {center[2]:7.3f}] | "
                        f"size=[{size[0]:6.3f}, {size[1]:6.3f}, {size[2]:6.3f}]")

    # Get per-link AABBs for playtable (UID=5) — individual fixture links
    playtable_uid = 5
    num_joints = p.getNumJoints(playtable_uid, physicsClientId=cid)
    link_aabbs = {}
    logger.info(f"\n--- Playtable Per-Link AABBs ({num_joints} links) ---")
    for link_idx in range(-1, num_joints):
        if link_idx >= 0:
            joint_info = p.getJointInfo(playtable_uid, link_idx, physicsClientId=cid)
            link_name = joint_info[12].decode()
        else:
            link_name = 'base_link'
        aabb_min, aabb_max = p.getAABB(playtable_uid, link_idx, physicsClientId=cid)
        center = np.array([(a + b) / 2 for a, b in zip(aabb_min, aabb_max)])
        size = np.array([b - a for a, b in zip(aabb_min, aabb_max)])
        link_aabbs[link_name] = {'center': center, 'size': size}
        if size.max() > 0.005:
            logger.info(f"  Link {link_idx:3d} ({link_name:30s}): "
                        f"center=[{center[0]:7.3f}, {center[1]:7.3f}, {center[2]:7.3f}], "
                        f"size=[{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")

    # Map fixtures to link names for comparison
    fixture_to_link = {
        'slider': 'slide_link', 'slider_left': 'slide_link', 'slider_right': 'slide_link',
        'drawer': 'drawer_link', 'drawer_handle': 'drawer_link',
        'lightbulb': 'light_link', 'light_switch': 'switch_link',
        'switch': 'switch_link', 'led': 'led_link', 'button': 'button_link',
    }

    # Compare hardcoded positions against per-link AABBs
    logger.info(f"\n--- Calibration: Hardcoded vs PyBullet Link AABBs ---")
    logger.info(f"{'Object':<20s} | {'Hardcoded Center':>30s} | {'Link AABB Center':>30s} | {'Error (mm)':>10s}")
    logger.info("-" * 100)

    warnings = []
    matches_found = 0

    for fixture_name, fixture_info in CALVIN_FIXTURES.items():
        hardcoded_pos = fixture_info['position']
        link_name = fixture_to_link.get(fixture_name)
        if link_name and link_name in link_aabbs:
            matches_found += 1
            link_center = link_aabbs[link_name]['center']
            error_mm = np.linalg.norm(hardcoded_pos - link_center) * 1000
            status = "OK" if error_mm < 50 else "WARN"
            logger.info(
                f"{fixture_name:<20s} | "
                f"[{hardcoded_pos[0]:7.3f}, {hardcoded_pos[1]:7.3f}, {hardcoded_pos[2]:7.3f}] | "
                f"[{link_center[0]:7.3f}, {link_center[1]:7.3f}, {link_center[2]:7.3f}] | "
                f"{error_mm:8.1f} {status}"
            )
            if error_mm >= 50:
                warnings.append(f"  {fixture_name}: {error_mm:.0f}mm from {link_name} center")

    if warnings:
        logger.warning(f"\n{len(warnings)} fixtures have >50mm error:")
        for w in warnings:
            logger.info(w)

    # Render segmentation if requested
    if args.visualize:
        seg_path = save_dir / "segmentation.png"
        render_segmentation(env, p, cid, str(seg_path))

    # Capture point cloud and render 3D visualization with bounding boxes
    logger.info("\n--- Capturing scene point cloud ---")
    from training.policies.diffuser_actor.preprocessing.calvin_utils import deproject
    from voxposer.calvin_interface import (
        CalvinLMPInterface, DEFAULT_WORKSPACE_MIN, DEFAULT_WORKSPACE_MAX,
    )
    from voxposer.value_map import ValueMap
    from voxposer.visualizer import ValueMapVisualizer

    calvin_env = env._gym_env._env
    calvin_obs = calvin_env.get_obs()
    static_cam = calvin_env.cameras[0]

    # Get depth and RGB from static camera
    rgb_obs = calvin_obs.get('rgb_obs', {})
    depth_obs = calvin_obs.get('depth_obs', {})
    depth_static = depth_obs.get('depth_static')
    rgb_static = rgb_obs.get('rgb_static')

    if depth_static is not None:
        # Deproject to world-frame point cloud using camera view matrix
        world_pts = deproject(static_cam, depth_static, homogeneous=False).T  # (N, 3)

        # Filter to workspace bounds (raw depth buffer values are NOT metric,
        # so we filter in world space after deprojection)
        ws_min = np.array(DEFAULT_WORKSPACE_MIN) - 0.05
        ws_max = np.array(DEFAULT_WORKSPACE_MAX) + 0.05
        valid = np.all((world_pts >= ws_min) & (world_pts <= ws_max), axis=1)
        pcd = world_pts[valid]

        # Get per-point RGB colors
        if rgb_static is not None:
            colors = rgb_static.reshape(-1, 3)[valid]
        else:
            colors = None

        logger.info(f"Point cloud: {pcd.shape[0]} points")

        # Save point cloud for offline use
        npz_path = save_dir / "scene_pointcloud.npz"
        save_data = {'points': pcd.astype(np.float32)}
        if colors is not None:
            save_data['colors'] = colors.astype(np.uint8)
        np.savez_compressed(str(npz_path), **save_data)
        logger.info(f"Saved point cloud to {npz_path}")

        # Set up object detection
        robot_obs = calvin_obs.get('robot_obs', np.zeros(15))
        scene_obs = calvin_obs.get('scene_obs', np.zeros(24))
        config = {
            'map_size': 100,
            'workspace_bounds_min': list(DEFAULT_WORKSPACE_MIN),
            'workspace_bounds_max': list(DEFAULT_WORKSPACE_MAX),
        }
        iface = CalvinLMPInterface(config)
        iface.update_state(robot_obs, scene_obs)
        detections = iface.get_all_detections()

        # Build dummy value map for visualization frame
        map_size = 100
        dummy_aff = np.zeros((map_size, map_size, map_size))
        value_map = ValueMap(
            affordance=dummy_aff,
            workspace_bounds_min=np.array(DEFAULT_WORKSPACE_MIN),
            workspace_bounds_max=np.array(DEFAULT_WORKSPACE_MAX),
            map_size=map_size,
            instruction="calibration (no value map)",
        )

        # Visualize
        vis_config = {
            'visualization_save_dir': str(save_dir),
            'visualization_quality': 'low',
        }
        visualizer = ValueMapVisualizer(vis_config)
        visualizer.update_scene_points(pcd, colors)
        visualizer.visualize(
            value_map,
            ee_pos_world=robot_obs[:3],
            objects=detections,
            show=False,
            save=True,
            filename='calibration_scene',
        )
        logger.info(f"Saved 3D visualization to {save_dir}/calibration_scene.html")
    else:
        logger.warning("No depth data available from static camera")

    logger.info(f"\nCalibration complete. {matches_found}/{len(CALVIN_FIXTURES)} fixtures matched.")


if __name__ == "__main__":
    main()
