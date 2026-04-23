"""Multi-task CALVIN benchmark evaluation.

Runs evaluation conditions across tasks, recording per-episode data
(success, steps, reward). Results JSON is updated after every episode
for crash safety. Supports resuming interrupted runs and extending
with additional episodes.

Usage:
    uv run python scripts/run_evaluation.py --evaluation baseline
    uv run python scripts/run_evaluation.py --evaluation langsteer
    uv run python scripts/run_evaluation.py --evaluation baseline langsteer

    # Resume an interrupted run (picks up where it left off):
    uv run python scripts/run_evaluation.py --evaluation baseline --output-dir outputs/evaluation/2026-04-10_14-30-00

    # Add more episodes to an existing run (e.g. go from 25 to 50):
    uv run python scripts/run_evaluation.py --evaluation baseline --num-episodes 50 --output-dir outputs/evaluation/2026-04-10_14-30-00
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_experiment import instantiate_env, instantiate_policy, instantiate_steering
from utils.rollout import EpisodeRunner

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-task CALVIN evaluation")
    parser.add_argument(
        "--evaluation", nargs="+", required=True,
        help="Evaluation condition config(s) from conf/evaluation/ (e.g., baseline langsteer)",
    )
    parser.add_argument("--num-episodes", type=int, default=25, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=360, help="Max env steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for starting conditions")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. If it contains existing results, the run resumes/extends.",
    )
    parser.add_argument(
        "--perturbation-axis", type=str, default=None, choices=["P1", "P2", "P3", "P4"],
        help="Use perturbed instructions from perturbed_language_annotations.json at this "
             "axis (P1..P4). P4 additionally filters starting conditions to indices listed "
             "in conf/evaluation/p4_valid_indices.json.",
    )
    return parser.parse_args()


def deterministic_hash(s: str) -> int:
    """Deterministic 32-bit hash from a string."""
    import hashlib
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Starting-condition sampling (deterministic, extensible)
# ---------------------------------------------------------------------------

def presample_starting_conditions(
    dataset_path: Path,
    split: str,
    task_list: List[str],
    num_episodes: int,
    seed: int,
) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], Dict[str, List[int]]]:
    """Pre-sample N (robot_obs, scene_obs) pairs per task from dataset episodes.

    The RNG is seeded once, then tasks are sampled in sorted order so that
    requesting more episodes later extends the sequence deterministically
    (the first K samples are always the same regardless of N >= K).
    """
    ann_path = dataset_path / split / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(str(ann_path), allow_pickle=True).item()
    tasks = ann["language"]["task"]
    start_end = ann["info"]["indx"]

    task_episode_ids: Dict[str, List[int]] = {}
    for i, task_name in enumerate(tasks):
        task_episode_ids.setdefault(task_name, []).append(start_end[i][0])

    starting_conditions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    sampled_episode_ids: Dict[str, List[int]] = {}

    for task_name in sorted(task_list):
        ep_ids = task_episode_ids.get(task_name, [])
        if not ep_ids:
            logger.warning(f"No dataset episodes for task '{task_name}', skipping")
            starting_conditions[task_name] = []
            sampled_episode_ids[task_name] = []
            continue

        # Per-task RNG seeded from (global seed, task_name) — independent of task order
        rng = random.Random(seed + deterministic_hash(task_name))
        sampled_ids = [rng.choice(ep_ids) for _ in range(num_episodes)]
        sampled_episode_ids[task_name] = sampled_ids

        conditions = []
        for ep_id in sampled_ids:
            ep_path = dataset_path / split / f"episode_{ep_id:07d}.npz"
            data = np.load(str(ep_path))
            conditions.append((
                data["robot_obs"].astype(np.float32),
                data["scene_obs"].astype(np.float32),
            ))
        starting_conditions[task_name] = conditions

    logger.info(f"Pre-sampled {num_episodes} starting conditions for {len(task_list)} tasks")
    return starting_conditions, sampled_episode_ids


# ---------------------------------------------------------------------------
# Results file I/O (incremental, resumable)
# ---------------------------------------------------------------------------

def load_results(results_path: Path) -> Dict:
    """Load existing results JSON, or return empty structure."""
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def save_results(results_path: Path, results: Dict):
    """Atomically write results JSON (write tmp then rename)."""
    tmp_path = results_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    tmp_path.rename(results_path)


def get_completed_episodes(results: Dict, task_name: str) -> int:
    """Return how many episodes are already recorded for a task."""
    return len(results.get("tasks", {}).get(task_name, {}).get("episodes", []))


# ---------------------------------------------------------------------------
# Steering helpers
# ---------------------------------------------------------------------------

def wire_steering(steering, policy, cfg):
    """Wire policy noise schedulers and trajectory loader to steering module."""
    if hasattr(steering, "set_position_scheduler"):
        steering.set_position_scheduler(policy._model.position_noise_scheduler)
        logger.info("Set position scheduler for steering")
    if hasattr(steering, "set_rotation_scheduler"):
        steering.set_rotation_scheduler(policy._model.rotation_noise_scheduler)
        logger.info("Set rotation scheduler for steering")
    if hasattr(steering, "set_trajectory_loader"):
        from utils.reference_trajectory_loader import ReferenceTrajectoryLoader
        loader = ReferenceTrajectoryLoader(
            dataset_path=cfg.env.dataset_path,
            split=cfg.env.split,
            lang_ann_path=cfg.env.lang_ann_path,
        )
        steering.set_trajectory_loader(loader)
        logger.info("Initialized reference trajectory loader for steering")


def setup_voxposer_episode(steering, env):
    """Setup VoxPoser steering for a new episode (after env.reset)."""
    state = env.get_scene_state()
    vp_instruction = env.task_description
    steering.setup_episode(
        env._task_name,
        instruction=vp_instruction,
        robot_obs=state["robot_obs"],
        scene_obs=state["scene_obs"],
        fixture_positions=state.get("fixture_positions"),
        block_aabbs=state.get("block_aabbs"),
    )
    if steering._value_map is not None:
        logger.info(f"VoxPoser value maps generated for: '{vp_instruction}'")
    else:
        logger.error(f"VoxPoser value map generation FAILED for: '{vp_instruction}'")


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def run_condition(
    eval_cfg: Dict,
    hydra_cfg,
    starting_conditions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    sampled_episode_ids: Dict[str, List[int]],
    num_episodes: int,
    max_steps: int,
    base_seed: int,
    results_path: Path,
    perturbation_axis: str = None,
    p4_valid_indices: Dict[str, List[int]] = None,
):
    """Run evaluation for a single condition, saving after every episode."""
    condition_name = eval_cfg["condition_name"]
    task_list = eval_cfg["tasks"]

    # Load existing results (for resume)
    results = load_results(results_path)
    if not results:
        results = {
            "condition_name": condition_name,
            "seed": base_seed,
            "max_steps": max_steps,
            "perturbation_axis": perturbation_axis,
            "tasks": {},
        }

    logger.info(f"\n{'='*70}")
    logger.info(f"CONDITION: {condition_name}")
    logger.info(f"{'='*70}")

    # Instantiate components
    env = instantiate_env(hydra_cfg)
    policy = instantiate_policy(hydra_cfg)
    steering = instantiate_steering(hydra_cfg)

    if not env._provide_pcd_images:
        env._provide_pcd_images = True

    if steering is not None:
        wire_steering(steering, policy, hydra_cfg)

    runner = EpisodeRunner(
        env=env, policy=policy, steering=steering,
        max_steps=max_steps * 2, collect_data=False,
    )

    for task_name in task_list:
        env.set_task(task_name)
        env._max_steps = max_steps

        conditions = starting_conditions.get(task_name, [])
        ep_ids = sampled_episode_ids.get(task_name, [])

        # P4: filter to starts manually verified as unambiguous for this task.
        # Tasks without an entry in the valid-indices map are unambiguous and run as-is.
        if perturbation_axis == "P4" and p4_valid_indices and task_name in p4_valid_indices:
            valid = [i for i in p4_valid_indices[task_name] if 0 <= i < len(conditions)]
            conditions = [conditions[i] for i in valid]
            ep_ids = [ep_ids[i] for i in valid]
            logger.info(f"  [P4] Filtered {task_name} to {len(valid)} valid starting conditions")

        n_available = min(num_episodes, len(conditions))

        # Initialize task entry if missing
        if task_name not in results["tasks"]:
            results["tasks"][task_name] = {
                "instruction": env.task_description,
                "episodes": [],
            }

        completed = get_completed_episodes(results, task_name)

        if completed >= num_episodes:
            logger.info(f"\n  {task_name}: {completed}/{num_episodes} already done, skipping")
            continue

        if n_available == 0:
            logger.warning(f"  No starting conditions for {task_name}, skipping")
            continue

        logger.info(f"\n  Task: {task_name} ({env.task_description})")
        logger.info(f"    Resuming from episode {completed + 1}/{num_episodes}")

        for ep_idx in range(completed, n_available):
            robot_obs, scene_obs = conditions[ep_idx]
            source_episode_id = int(ep_ids[ep_idx]) if ep_idx < len(ep_ids) else None

            # Reproducible diffusion noise
            ep_seed = base_seed + deterministic_hash(task_name) + ep_idx
            set_seed(ep_seed)

            obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

            # Setup steering
            if steering is not None and hasattr(steering, "setup_episode"):
                if hydra_cfg.steering.name == "voxposer":
                    setup_voxposer_episode(steering, env)

            # Step callback for steering
            def step_callback(timestep, obs, action, reward, done, info):
                if steering is not None and hasattr(steering, "increment_step"):
                    steering.increment_step()
                if steering is not None and hasattr(steering, "refresh_costmap"):
                    state = env.get_scene_state()
                    steering.refresh_costmap(
                        state["robot_obs"], state["scene_obs"],
                        fixture_positions=state.get("fixture_positions"),
                        block_aabbs=state.get("block_aabbs"),
                    )
                if steering is not None and hasattr(steering, "check_stage_transition"):
                    steering.check_stage_transition(obs.ee_pose[:3])
                if steering is not None and hasattr(steering, "update_dash"):
                    steering.update_dash(obs.ee_pose[:3])

            result = runner.run_episode(
                initial_obs=obs, reset_env=False, reset_policy=True,
                step_callback=step_callback,
            )

            # Record episode data
            ep_record = {
                "episode_idx": ep_idx,
                "source_episode_id": source_episode_id,
                "success": result.success,
                "steps": result.episode_length,
                "reward": float(result.episode_reward),
            }
            results["tasks"][task_name]["episodes"].append(ep_record)

            # Save immediately (crash-safe)
            save_results(results_path, results)

            if result.success:
                logger.info(
                    f"    ✓ Episode {ep_idx+1}/{num_episodes} SUCCEEDED "
                    f"(steps={result.episode_length}, reward={result.episode_reward:.2f})"
                )
            else:
                logger.info(
                    f"    ✗ Episode {ep_idx+1}/{num_episodes} FAILED "
                    f"(steps={result.episode_length}, reward={result.episode_reward:.2f})"
                )

        # Log task summary so far
        eps = results["tasks"][task_name]["episodes"]
        n_success = sum(1 for e in eps if e["success"])
        avg_steps = np.mean([e["steps"] for e in eps if e["success"]]) if n_success else float("nan")
        logger.info(
            f"    -> {task_name}: {n_success}/{len(eps)} succeeded, "
            f"avg steps to success: {avg_steps:.0f}"
        )

    env.close()

    # Print final summary
    print_summary(results)
    return results


def print_summary(results: Dict):
    """Print a per-task summary table."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{'Task':<35} {'Success':>10} {'Rate':>8} {'Avg Steps (succ)':>18}")
    logger.info(f"{'-'*80}")

    total_success = 0
    total_episodes = 0

    for task_name, task_data in results["tasks"].items():
        eps = task_data["episodes"]
        n = len(eps)
        n_success = sum(1 for e in eps if e["success"])
        rate = n_success / n if n else 0.0
        succ_steps = [e["steps"] for e in eps if e["success"]]
        avg_steps = np.mean(succ_steps) if succ_steps else float("nan")

        total_success += n_success
        total_episodes += n

        logger.info(f"  {task_name:<33} {n_success:>3}/{n:<4}  {rate:>7.1%}  {avg_steps:>16.0f}")

    overall_rate = total_success / total_episodes if total_episodes else 0.0
    logger.info(f"{'-'*80}")
    logger.info(f"  {'OVERALL':<33} {total_success:>3}/{total_episodes:<4}  {overall_rate:>7.1%}")
    logger.info(f"{'='*80}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Output directory (axis-prefixed when using a perturbed run so axes don't collide)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dir_name = f"{args.perturbation_axis}_{timestamp}" if args.perturbation_axis else timestamp
        output_dir = Path("outputs/evaluation") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation configs
    conf_dir = Path(__file__).parent.parent / "conf"
    eval_configs = []
    for name in args.evaluation:
        eval_path = conf_dir / "evaluation" / f"{name}.yaml"
        if not eval_path.exists():
            logger.error(f"Evaluation config not found: {eval_path}")
            sys.exit(1)
        eval_cfg = OmegaConf.to_container(OmegaConf.load(str(eval_path)), resolve=True)
        eval_configs.append((name, eval_cfg))

    # All tasks (union across configs, preserving order from first)
    all_tasks = eval_configs[0][1]["tasks"]
    logger.info(f"Evaluating {len(all_tasks)} tasks, {args.num_episodes} episodes each")

    # Dataset path from base Hydra config
    abs_conf_dir = str(conf_dir.resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=abs_conf_dir, version_base=None):
        base_cfg = compose(config_name="config")
    dataset_path = Path(base_cfg.env.dataset_path)
    split = base_cfg.env.split

    # Pre-sample starting conditions (shared across all conditions)
    starting_conditions, sampled_episode_ids = presample_starting_conditions(
        dataset_path, split, all_tasks, args.num_episodes, args.seed,
    )

    # P4: load manually verified valid starting indices. These were picked against
    # the same (seed, num_episodes) used here, so the indices map 1:1 into the
    # presampled lists. Mismatches are fatal — the filter would otherwise pick the
    # wrong starts.
    p4_valid_indices = None
    if args.perturbation_axis == "P4":
        p4_path = conf_dir / "evaluation" / "p4_valid_indices.json"
        if not p4_path.exists():
            logger.error(
                f"--perturbation-axis P4 requires {p4_path}. "
                f"Run tmp/p4_validation/render_starting_states.py and "
                f"tmp/p4_validation/label_starting_states.py first."
            )
            sys.exit(1)
        with open(p4_path) as f:
            p4_data = json.load(f)
        if p4_data.get("seed") != args.seed or p4_data.get("num_episodes") != args.num_episodes:
            logger.error(
                f"P4 valid-indices JSON was built with seed={p4_data.get('seed')}, "
                f"num_episodes={p4_data.get('num_episodes')}, but this run uses "
                f"seed={args.seed}, num_episodes={args.num_episodes}. Indices would "
                f"point at different starting conditions. Re-run the labeler or "
                f"match the seed/num-episodes."
            )
            sys.exit(1)
        p4_valid_indices = p4_data["valid_indices"]
        logger.info(f"Loaded P4 valid indices for {len(p4_valid_indices)} tasks")

    # Save/update sampled episode IDs
    ep_ids_path = output_dir / "sampled_episodes.json"
    with open(ep_ids_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "num_episodes": args.num_episodes,
            "episode_ids": {k: [int(x) for x in v] for k, v in sampled_episode_ids.items()},
        }, f, indent=2)

    # Run each condition
    for config_name, eval_cfg in eval_configs:
        overrides = [
            f"steering={eval_cfg['steering']}",
            "env.provide_pcd_images=true",
            "env.use_gui=false",
            f"env.max_steps={args.max_steps}",
        ]
        if "policy_config" in eval_cfg:
            overrides.insert(0, f"policy={eval_cfg['policy_config']}")
        for key, value in eval_cfg["policy"].items():
            overrides.append(f"policy.{key}={value}")
        if args.perturbation_axis is not None:
            overrides += [
                "env.perturbed_ann_path=perturbed_language_annotations.json",
                f"env.perturbation_axis={args.perturbation_axis}",
            ]

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=abs_conf_dir, version_base=None):
            hydra_cfg = compose(config_name="config", overrides=overrides)

        # Each condition gets its own results file (resumable independently)
        results_path = output_dir / f"{config_name}.json"

        run_condition(
            eval_cfg=eval_cfg,
            hydra_cfg=hydra_cfg,
            starting_conditions=starting_conditions,
            sampled_episode_ids=sampled_episode_ids,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            base_seed=args.seed,
            results_path=results_path,
            perturbation_axis=args.perturbation_axis,
            p4_valid_indices=p4_valid_indices,
        )

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
