"""Main evaluation loop for running experiments."""

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any
import logging
from pathlib import Path
import numpy as np

from core.types import Observation, Action
from core.env import BaseEnvironment
from core.policy import BasePolicy
from core.steering import BaseSteering
from utils.rollout import EpisodeRunner
from visualization import VisualizationManager, VisualizationConfig

logger = logging.getLogger(__name__)


def instantiate_env(cfg: DictConfig) -> BaseEnvironment:
    """Factory function to instantiate environment based on config."""
    env_name = cfg.env.name
    if env_name == "calvin":
        from envs.calvin import CalvinEnvironment
        return CalvinEnvironment(OmegaConf.to_container(cfg.env, resolve=True))
    elif env_name == "rlbench":
        from envs.rlbench import RLBenchEnvironment
        return RLBenchEnvironment(OmegaConf.to_container(cfg.env, resolve=True))
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def instantiate_policy(cfg: DictConfig) -> BasePolicy:
    """Factory function to instantiate policy based on config."""
    policy_name = cfg.policy.name
    if policy_name == "dp3":
        from policies.dp3 import DP3Policy
        policy = DP3Policy(OmegaConf.to_container(cfg.policy, resolve=True))
    elif policy_name == "diffuser_actor":
        from policies.diffuser_actor import DiffuserActorPolicy
        policy = DiffuserActorPolicy(OmegaConf.to_container(cfg.policy, resolve=True))
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    # Load checkpoint if specified
    if hasattr(cfg.policy, "ckpt_path") and cfg.policy.ckpt_path:
        policy.load_checkpoint(cfg.policy.ckpt_path)
    
    return policy


def instantiate_steering(cfg: DictConfig) -> BaseSteering | None:
    """Factory function to instantiate steering module based on config."""
    steering_name = cfg.steering.name
    if steering_name == "none" or steering_name is None:
        return None
    elif steering_name == "dynaguide":
        from steering.dynaguide import DynaGuideSteering
        return DynaGuideSteering(OmegaConf.to_container(cfg.steering, resolve=True))
    elif steering_name == "tweedie":
        from steering.tweedie import TweedieSteering
        return TweedieSteering(OmegaConf.to_container(cfg.steering, resolve=True))
    else:
        raise ValueError(f"Unknown steering: {steering_name}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation loop with visualization and logging.

    1. Instantiate Env and Policy using factory functions
    2. Run loop: obs = env.reset(), act = policy(obs), env.step(act)
    3. Log success rates and trajectory data
    """
    logger.info(f"Starting experiment with config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    import random
    import torch
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Instantiate components
    env = instantiate_env(cfg)
    policy = instantiate_policy(cfg)
    steering = instantiate_steering(cfg)

    # Initialize visualization manager if enabled
    viz_manager = None
    if 'visualization' in cfg and OmegaConf.to_container(cfg.visualization, resolve=True):
        viz_config = VisualizationConfig.from_dict(OmegaConf.to_container(cfg.visualization, resolve=True))
        if viz_config.is_any_enabled():
            viz_manager = VisualizationManager(viz_config)
            logger.info(f"Initialized {viz_manager}")

    # Initialize steering with policy's scheduler and trajectory loader
    if steering is not None:
        if hasattr(steering, 'set_scheduler'):
            steering.set_scheduler(policy._dp3_model.noise_scheduler)
            logger.info("Set scheduler reference for steering")

        if hasattr(steering, 'set_trajectory_loader'):
            from utils.reference_trajectory_loader import ReferenceTrajectoryLoader
            loader = ReferenceTrajectoryLoader(
                dataset_path=cfg.env.dataset_path,
                split=cfg.env.split,
                lang_ann_path=cfg.env.lang_ann_path
            )
            steering.set_trajectory_loader(loader)
            logger.info("Initialized reference trajectory loader for steering")

    # Setup logging
    enable_gui = cfg.get("enable_gui", False)
    log_trajectory = cfg.get("log_trajectory", False)
    log_dir = Path(cfg.get("log_dir", "outputs/trajectories"))
    if log_trajectory:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Trajectory data will be saved to: {log_dir}")

    # Create episode runner
    runner = EpisodeRunner(
        env=env,
        policy=policy,
        steering=steering,
        max_steps=cfg.get("max_steps", 1000),
        collect_data=log_trajectory
    )

    # Run evaluation loop
    num_episodes = cfg.get("num_episodes", 10)
    success_count = 0

    for episode in range(num_episodes):
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        logger.info(f"Task: {env.task_description}")
        logger.info(f"{'='*60}")

        # Setup episode-specific steering and/or reference initial conditions
        obs = None
        use_reference_init = cfg.get('use_reference_init', False)

        if steering is not None and hasattr(steering, 'setup_episode'):
            # Steering is active - use reference trajectory initial state
            robot_obs, scene_obs = steering.setup_episode(env._task_name)

            if robot_obs is not None and scene_obs is not None:
                obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                logger.info("Reset environment to reference trajectory starting state (steering active)")
            else:
                obs = env.reset()
        elif use_reference_init:
            # Steering is NOT active, but user wants to use reference initial conditions
            # Load reference trajectory just for initial state
            from utils.reference_trajectory_loader import ReferenceTrajectoryLoader
            loader = ReferenceTrajectoryLoader(
                dataset_path=cfg.env.dataset_path,
                split=cfg.env.split,
                lang_ann_path=cfg.env.lang_ann_path
            )
            traj_data = loader.load_trajectory_for_task(env._task_name)

            if traj_data is not None:
                robot_obs = traj_data['robot_obs_init']
                scene_obs = traj_data['scene_obs_init']
                obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                logger.info("Reset environment to reference trajectory starting state (no steering, for comparison)")
            else:
                logger.warning(f"No reference trajectory found for task {env._task_name}, using default reset")
                obs = env.reset()
        else:
            # Normal reset with task-specific or random initial conditions
            obs = env.reset()

        # Define step callback for logging and steering step tracking
        def step_callback(timestep, obs, action, reward, done, info):
            # Increment steering step counter for sliding window
            if steering is not None and hasattr(steering, 'increment_step'):
                steering.increment_step()

            # Log step info
            logger.info(f"Step {timestep:3d} | Action: {action.trajectory[0][:3]} (pos) {action.trajectory[0][6]:.2f} (grip)")
            if reward > 0:
                logger.info(f"        | Reward: {reward:.2f} ✓")

        # Reset visualization for new episode
        if viz_manager:
            viz_manager.reset()

        # Run episode using shared runner
        result = runner.run_episode(
            initial_obs=obs,
            reset_env=False,  # Already reset above
            reset_policy=True,
            step_callback=step_callback,
            render=enable_gui
        )

        # Save trajectory if needed
        if log_trajectory:
            traj_path = log_dir / f"episode_{episode:04d}.npz"
            result.trajectory_collector.save_to_npz(str(traj_path))
            logger.info(f"Saved trajectory to {traj_path}")

        # Visualize episode if visualization is enabled
        # Note: Camera and PyBullet rendering happen during the episode via callbacks
        # This is for post-episode visualizations like reference plots
        if viz_manager and viz_manager.config.reference_plot:
            # If steering is active and has reference trajectory, visualize it
            if steering is not None and hasattr(steering, 'reference_trajectory'):
                if steering.reference_trajectory is not None:
                    viz_manager.visualize_reference_trajectory(
                        steering.reference_trajectory,
                        task_name=env._task_name,
                        horizon=cfg.policy.get('pred_horizon', 16)
                    )

        # Update counters and log results
        if result.success:
            success_count += 1
            logger.info(f"✓ Episode {episode + 1} SUCCEEDED (reward: {result.episode_reward:.2f}, steps: {result.episode_length})")
        else:
            logger.info(f"✗ Episode {episode + 1} FAILED (reward: {result.episode_reward:.2f}, steps: {result.episode_length})")

    # Log final results
    success_rate = success_count / num_episodes
    logger.info(f"\n{'='*60}")
    logger.info(f"Final Results: {success_rate:.2%} ({success_count}/{num_episodes})")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
