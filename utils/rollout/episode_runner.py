"""
Episode/rollout execution utilities.

Provides a standardized way to run policy rollouts in environments,
shared between run_experiment.py and visualize_trajectories.py.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Dict

from core.env import BaseEnvironment
from core.policy import BasePolicy
from core.steering import BaseSteering
from core.types import Observation
from utils.rollout.data_collector import TrajectoryDataCollector

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result of a single episode rollout."""

    success: bool                    # Whether episode succeeded
    episode_length: int              # Number of steps taken
    episode_reward: float            # Total reward
    trajectory_collector: TrajectoryDataCollector  # Full trajectory data
    info: Dict                       # Final info dict from environment


class EpisodeRunner:
    """
    Runs policy rollouts in an environment.

    Provides a clean, reusable interface for executing episodes
    with optional data collection, callbacks, and steering.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        policy: BasePolicy,
        steering: Optional[BaseSteering] = None,
        max_steps: int = 1000,
        collect_data: bool = True
    ):
        """
        Initialize episode runner.

        Args:
            env: Environment instance
            policy: Policy instance
            steering: Optional steering module
            max_steps: Maximum steps per episode
            collect_data: Whether to collect trajectory data
        """
        self.env = env
        self.policy = policy
        self.steering = steering
        self.max_steps = max_steps
        self.collect_data = collect_data

    def run_episode(
        self,
        initial_obs: Optional[Observation] = None,
        reset_env: bool = True,
        reset_policy: bool = True,
        step_callback: Optional[Callable] = None,
        render: bool = False
    ) -> EpisodeResult:
        """
        Run a single episode.

        Args:
            initial_obs: Optional pre-reset observation (if None, calls env.reset())
            reset_env: Whether to reset environment (False if using snapshot restore)
            reset_policy: Whether to reset policy observation buffer
            step_callback: Optional callback called after each step: callback(timestep, obs, action, reward, done, info)
            render: Whether to render environment

        Returns:
            EpisodeResult with trajectory data and statistics
        """
        # Initialize trajectory collector
        collector = TrajectoryDataCollector() if self.collect_data else None

        # Reset environment
        if reset_env:
            obs = self.env.reset()
        else:
            obs = initial_obs
            if obs is None:
                raise ValueError("Must provide initial_obs if reset_env=False")

        # Reset policy
        if reset_policy:
            self.policy.reset()

        # Run episode
        done = False
        timestep = 0
        episode_reward = 0.0
        final_info = {}

        while not done and timestep < self.max_steps:
            # Get action from policy
            action = self.policy.forward(obs, steering=self.steering)

            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward

            # Collect data
            if self.collect_data:
                collector.add_step(
                    timestep=timestep,
                    observation=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info
                )

            # Render if requested
            if render:
                self.env.render(mode='human')

            # Call step callback
            if step_callback is not None:
                step_callback(timestep, obs, action, reward, done, info)

            # Update state
            obs = next_obs
            timestep += 1
            final_info = info

        # Check success
        success = final_info.get('success', False)

        # Log episode summary
        status = "SUCCESS" if success else "FAILED"
        logger.debug(
            f"Episode complete: {status} | "
            f"Steps: {timestep} | "
            f"Reward: {episode_reward:.2f}"
        )

        return EpisodeResult(
            success=success,
            episode_length=timestep,
            episode_reward=episode_reward,
            trajectory_collector=collector,
            info=final_info
        )

    def run_multiple_episodes(
        self,
        num_episodes: int,
        episode_callback: Optional[Callable] = None,
        **episode_kwargs
    ) -> list[EpisodeResult]:
        """
        Run multiple episodes.

        Args:
            num_episodes: Number of episodes to run
            episode_callback: Optional callback after each episode: callback(episode_id, result)
            **episode_kwargs: Additional arguments passed to run_episode()

        Returns:
            List of EpisodeResult objects
        """
        results = []

        for episode_id in range(num_episodes):
            logger.info(f"Running episode {episode_id + 1}/{num_episodes}")

            result = self.run_episode(**episode_kwargs)
            results.append(result)

            if episode_callback is not None:
                episode_callback(episode_id, result)

        return results

    def get_success_rate(self, results: list[EpisodeResult]) -> float:
        """
        Calculate success rate from episode results.

        Args:
            results: List of EpisodeResult objects

        Returns:
            Success rate (0.0 to 1.0)
        """
        if len(results) == 0:
            return 0.0

        success_count = sum(1 for r in results if r.success)
        return success_count / len(results)

    def log_summary(self, results: list[EpisodeResult]):
        """
        Log summary statistics for multiple episodes.

        Args:
            results: List of EpisodeResult objects
        """
        success_rate = self.get_success_rate(results)
        avg_length = np.mean([r.episode_length for r in results])
        avg_reward = np.mean([r.episode_reward for r in results])

        logger.info("=" * 70)
        logger.info("EPISODE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total episodes: {len(results)}")
        logger.info(f"Success rate: {success_rate:.1%}")
        logger.info(f"Average length: {avg_length:.1f} steps")
        logger.info(f"Average reward: {avg_reward:.2f}")
        logger.info("=" * 70)
