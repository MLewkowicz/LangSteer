"""Main evaluation loop for running experiments."""

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any
import logging

from core.types import Observation, Action
from core.env import BaseEnvironment
from core.policy import BasePolicy
from core.steering import BaseSteering

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
    Main evaluation loop.
    
    1. Instantiate Env and Policy using factory functions
    2. Run loop: obs = env.reset(), act = policy(obs), env.step(act)
    3. Log success rates
    """
    logger.info(f"Starting experiment with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    import random
    import numpy as np
    import torch
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Instantiate components
    env = instantiate_env(cfg)
    policy = instantiate_policy(cfg)
    steering = instantiate_steering(cfg)
    
    # Run evaluation loop
    num_episodes = cfg.get("num_episodes", 10)
    success_count = 0
    
    for episode in range(num_episodes):
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        obs = env.reset()
        policy.reset()
        
        episode_reward = 0.0
        step_count = 0
        done = False
        
        while not done:
            # Get action from policy (with optional steering)
            action = policy.forward(obs, steering=steering)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Check for episode termination
            if step_count > cfg.get("max_steps", 1000):
                done = True
        
        # Check success
        if info.get("success", False):
            success_count += 1
            logger.info(f"Episode {episode + 1} succeeded!")
        else:
            logger.info(f"Episode {episode + 1} failed. Reward: {episode_reward:.2f}")
    
    # Log final results
    success_rate = success_count / num_episodes
    logger.info(f"Final success rate: {success_rate:.2%} ({success_count}/{num_episodes})")


if __name__ == "__main__":
    main()
