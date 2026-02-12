"""Rendering script for visualizing policy behavior."""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from scripts.run_experiment import instantiate_env, instantiate_policy, instantiate_steering

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Visualization script for rendering policy behavior.
    Similar to run_experiment but with rendering enabled.
    """
    logger.info(f"Starting visualization with config:\n{OmegaConf.to_yaml(cfg)}")
    
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
    
    # Enable rendering in environment config
    env.cfg["render_mode"] = "human"  # Override for visualization
    
    # Run visualization loop
    num_episodes = cfg.get("num_episodes", 1)
    
    for episode in range(num_episodes):
        logger.info(f"Visualizing episode {episode + 1}/{num_episodes}")
        obs = env.reset()
        policy.reset()
        
        step_count = 0
        done = False
        
        while not done:
            # Get action from policy (with optional steering)
            action = policy.forward(obs, steering=steering)
            
            # Step environment (with rendering)
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            # Check for episode termination
            if step_count > cfg.get("max_steps", 1000):
                done = True
        
        logger.info(f"Episode {episode + 1} completed. Success: {info.get('success', False)}")


if __name__ == "__main__":
    main()
