"""Standalone test for VoxPoser LMP pipeline — no CALVIN env required.

Tests the full LLM composer → value map generation → visualization pipeline
using mock scene state. Useful for validating LLM outputs and prompt quality.

Usage:
    uv run python scripts/test_voxposer.py                          # cached
    uv run python scripts/test_voxposer.py --no-cache               # fresh LLM call
    uv run python scripts/test_voxposer.py --instruction "open the drawer"
    uv run python scripts/test_voxposer.py --provider openai --model gpt-4o
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxposer.lmp import setup_lmp, set_lmp_objects
from voxposer.value_map import ValueMap
from voxposer.visualizer import ValueMapVisualizer
from voxposer.calvin_interface import DEFAULT_WORKSPACE_MIN, DEFAULT_WORKSPACE_MAX

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Default test instructions covering main CALVIN task categories
DEFAULT_INSTRUCTIONS = [
    "open the drawer",
    "push the red block to the left",
    "turn on the lightbulb",
    "slide the door to the right",
    "lift the blue block",
]

# Mock robot state: neutral EE position from task_configs.py
MOCK_ROBOT_OBS = np.array([
    0.02586889, -0.2313129, 0.5712808,   # tcp position
    3.09045411, -0.02908596, 1.50013585,  # tcp orientation
    0.07999963,                           # gripper width
    -1.21779124, 1.03987629, 2.11978254,  # joints
    -2.34205014, -0.87015899, 1.64119093,
    0.55344928, 1.0,                      # gripper action
])

# Mock scene state: blocks on table, drawer closed, lights off
MOCK_SCENE_OBS = np.zeros(24)
MOCK_SCENE_OBS[6:9] = [0.05, -0.12, 0.46]    # red block on table
MOCK_SCENE_OBS[12:15] = [0.23, -0.12, 0.46]   # blue block on table
MOCK_SCENE_OBS[18:21] = [0.23, -0.12, 0.46]   # pink block on table


def eval_map(map_fn):
    """Evaluate a voxel map result, handling callables and wrappers."""
    if map_fn is None:
        return None
    try:
        if callable(map_fn):
            result = map_fn()
        else:
            result = map_fn
        if hasattr(result, 'array'):
            return result.array
        return np.asarray(result)
    except Exception as e:
        logger.warning(f"Failed to evaluate map: {e}")
        return None


def load_scene_pcd(path):
    """Load a saved scene point cloud from .npz file."""
    data = np.load(path)
    points = data['points']
    colors = data.get('colors', None)
    logger.info(f"Loaded scene point cloud: {points.shape[0]} points from {path}")
    return points, colors


def run_test(config, instruction, save_dir, scene_pcd=None):
    """Run the LMP composer for a single instruction and visualize results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: '{instruction}'")
    logger.info(f"{'='*60}")

    # Setup LMP system
    lmps, lmp_interface = setup_lmp(config)
    lmp_interface.update_state(MOCK_ROBOT_OBS, MOCK_SCENE_OBS)
    set_lmp_objects(lmps, lmp_interface.get_object_names())

    # Run composer
    try:
        result = lmps['composer'](instruction)
    except Exception as e:
        logger.error(f"Composer failed: {e}")
        return False

    # Parse result
    if not isinstance(result, tuple) or len(result) != 3:
        logger.error(f"Unexpected result type: {type(result)}, expected 3-tuple")
        return False

    aff_fn, avoid_fn, grip_fn = result
    affordance = eval_map(aff_fn)
    avoidance = eval_map(avoid_fn)
    gripper = eval_map(grip_fn)

    if affordance is None:
        logger.error("No affordance map generated")
        return False

    # Print statistics
    logger.info(f"Affordance: shape={affordance.shape}, max={affordance.max():.3f}, "
                f"non-zero={np.count_nonzero(affordance)}")
    if avoidance is not None:
        logger.info(f"Avoidance:  shape={avoidance.shape}, max={avoidance.max():.3f}, "
                    f"non-zero={np.count_nonzero(avoidance)}")
    if gripper is not None:
        logger.info(f"Gripper:    shape={gripper.shape}, "
                    f"open={np.count_nonzero(gripper == 1)}, "
                    f"closed={np.count_nonzero(gripper == 0)}")

    # Build ValueMap
    ws_min = np.array(config.get('workspace_bounds_min', DEFAULT_WORKSPACE_MIN))
    ws_max = np.array(config.get('workspace_bounds_max', DEFAULT_WORKSPACE_MAX))
    map_size = config.get('map_size', 100)

    value_map = ValueMap(
        affordance=affordance,
        avoidance=avoidance,
        gripper=gripper,
        workspace_bounds_min=ws_min,
        workspace_bounds_max=ws_max,
        map_size=map_size,
        instruction=instruction,
    )
    value_map.smooth()

    # Get all object detections for bounding box overlay
    detections = lmp_interface.get_all_detections()

    # Visualize
    vis_config = {
        'visualization_save_dir': str(save_dir),
        'visualization_quality': 'low',
    }
    visualizer = ValueMapVisualizer(vis_config)

    # Overlay scene point cloud if available
    if scene_pcd is not None:
        points, colors = scene_pcd
        visualizer.update_scene_points(points, colors)

    # Use instruction slug as filename
    slug = instruction.replace(' ', '_')[:40]
    fig = visualizer.visualize(
        value_map,
        ee_pos_world=MOCK_ROBOT_OBS[:3],
        objects=detections,
        show=False,
        save=True,
        filename=slug,
    )
    logger.info(f"Saved visualization to {save_dir}/{slug}.html")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test VoxPoser LMP pipeline")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Single instruction to test (default: run all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable LLM response cache")
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model (default: provider-dependent)")
    parser.add_argument("--save-dir", type=str, default="outputs/voxposer_test",
                        help="Directory for HTML visualizations")
    parser.add_argument("--scene-pcd", type=str, default=None,
                        help="Path to .npz scene point cloud (from calibrate_voxposer_objects.py)")
    args = parser.parse_args()

    # Build config
    config = {
        'map_size': 100,
        'workspace_bounds_min': [-0.35, -0.40, 0.40],
        'workspace_bounds_max': [0.35, 0.15, 0.85],
        'llm_provider': args.provider,
        'llm_model': args.model or (
            'claude-sonnet-4-20250514' if args.provider == 'anthropic' else 'gpt-4o'
        ),
        'llm_temperature': 0,
        'llm_max_tokens': 512,
        'cache_dir': 'cache/voxposer_llm',
        'load_cache': not args.no_cache,
    }

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load scene point cloud (explicit path, or auto-detect from calibration output)
    scene_pcd = None
    pcd_path = args.scene_pcd
    if pcd_path is None:
        default_pcd = Path('outputs/calibration/scene_pointcloud.npz')
        if default_pcd.exists():
            pcd_path = str(default_pcd)
            logger.info(f"Auto-detected scene point cloud: {pcd_path}")
    if pcd_path:
        scene_pcd = load_scene_pcd(pcd_path)

    instructions = [args.instruction] if args.instruction else DEFAULT_INSTRUCTIONS

    successes = 0
    for instruction in instructions:
        ok = run_test(config, instruction, save_dir, scene_pcd=scene_pcd)
        if ok:
            successes += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {successes}/{len(instructions)} instructions succeeded")
    logger.info(f"Visualizations saved to: {save_dir}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
