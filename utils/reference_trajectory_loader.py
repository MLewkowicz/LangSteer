"""Reference trajectory loader for CALVIN dataset.

Loads reference trajectories from CALVIN episodes for steering guidance.
Maps tasks to episodes using language annotations and caches loaded data.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ReferenceTrajectoryLoader:
    """Loads reference trajectories from CALVIN dataset for steering."""

    def __init__(self, dataset_path: str, split: str, lang_ann_path: str):
        """
        Initialize reference trajectory loader.

        Args:
            dataset_path: Path to CALVIN dataset root directory
            split: Dataset split - "training" or "validation"
            lang_ann_path: Path to auto_lang_ann.npy annotation file
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.lang_ann_path = Path(lang_ann_path)

        # Load language annotations to map tasks to episodes
        if not self.lang_ann_path.exists():
            logger.warning(f"Language annotation file not found: {lang_ann_path}")
            self.annotations = None
            self.task_to_episodes = {}
        else:
            self.annotations = np.load(lang_ann_path, allow_pickle=True).item()
            self.task_to_episodes = self._build_task_index()
            logger.info(f"Loaded annotations for {len(self.task_to_episodes)} tasks")

        # Cache for loaded trajectories
        self.cache = {}

    def _build_task_index(self) -> Dict[str, List[int]]:
        """
        Build mapping from task name to list of episode IDs.

        Returns:
            Dictionary mapping task names to lists of episode IDs
        """
        if self.annotations is None:
            return {}

        task_index = {}

        # Extract episode ranges and task names from annotations
        for i, (start_id, end_id) in enumerate(self.annotations['info']['indx']):
            task_name = self.annotations['language']['task'][i]
            episode_ids = list(range(start_id, end_id + 1))

            if task_name not in task_index:
                task_index[task_name] = []
            task_index[task_name].extend(episode_ids)

        logger.debug(f"Built task index: {len(task_index)} unique tasks")
        return task_index

    def load_trajectory_for_task(self, task_name: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load the first reference trajectory for the given task.

        Args:
            task_name: Name of the task to load trajectory for

        Returns:
            Dictionary with keys:
                - 'actions': (T, 7) relative end-effector actions
                - 'robot_obs': (T, 15) robot proprioception
                - 'scene_obs': (T, 24) scene state
                - 'robot_obs_init': (15,) initial robot state
                - 'scene_obs_init': (24,) initial scene state
            Returns None if task not found or episode file missing
        """
        # Check cache first
        if task_name in self.cache:
            logger.debug(f"Returning cached trajectory for task: {task_name}")
            return self.cache[task_name]

        # Check if task exists in annotations
        if task_name not in self.task_to_episodes:
            logger.warning(f"No episodes found for task: {task_name}")
            return None

        # Load sequence of episodes for this task
        # CALVIN stores one frame per NPZ file, so we need to load multiple files
        episode_ids = self.task_to_episodes[task_name]
        first_episode_id = episode_ids[0]

        # Load up to 50 frames to build a reference trajectory
        max_frames = min(50, len(episode_ids))
        frames_to_load = episode_ids[:max_frames]

        try:
            # Load all frames
            actions_list = []
            robot_obs_list = []
            scene_obs_list = []

            for ep_id in frames_to_load:
                ep_path = self.dataset_path / self.split / f"episode_{ep_id:07d}.npz"

                if not ep_path.exists():
                    logger.warning(f"Episode file not found: {ep_path}, stopping trajectory loading")
                    break

                data = np.load(ep_path)
                actions_list.append(data['rel_actions'].astype(np.float32))
                robot_obs_list.append(data['robot_obs'].astype(np.float32))
                scene_obs_list.append(data['scene_obs'].astype(np.float32))

            if len(actions_list) == 0:
                logger.warning(f"No frames loaded for task: {task_name}")
                return None

            # Stack frames into trajectory arrays
            trajectory = {
                'actions': np.stack(actions_list, axis=0),  # (T, 7)
                'robot_obs': np.stack(robot_obs_list, axis=0),  # (T, 15)
                'scene_obs': np.stack(scene_obs_list, axis=0),  # (T, 24)
                'robot_obs_init': robot_obs_list[0],  # First frame
                'scene_obs_init': scene_obs_list[0]
            }

            # Cache the loaded trajectory
            self.cache[task_name] = trajectory

            logger.info(
                f"Loaded reference trajectory for '{task_name}' "
                f"(episodes {first_episode_id}-{frames_to_load[-1]}, {len(actions_list)} frames)"
            )

            return trajectory

        except Exception as e:
            logger.error(f"Error loading episodes for task '{task_name}': {e}")
            return None

    def get_available_tasks(self) -> List[str]:
        """
        Get list of all available task names.

        Returns:
            List of task names that have reference trajectories
        """
        return list(self.task_to_episodes.keys())

    def clear_cache(self):
        """Clear the trajectory cache to free memory."""
        self.cache.clear()
        logger.info("Cleared trajectory cache")
