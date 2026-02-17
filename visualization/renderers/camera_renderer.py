"""Camera view renderer - displays RGB images from CALVIN cameras.

Extracted from scripts/visualize_cameras.py
"""

import numpy as np
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CameraRenderer:
    """Renders camera views from CALVIN environment (static overhead + gripper cameras)."""

    def __init__(self, config):
        """
        Initialize camera renderer.

        Args:
            config: CameraVisualizationConfig instance
        """
        self.config = config
        self.output_dir = Path(config.save_dir)
        self.step_counter = 0

        if config.save_images:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_step(self, calvin_obs: Dict, step: int = None) -> Dict[str, np.ndarray]:
        """
        Render camera views for a single step.

        Args:
            calvin_obs: Raw CALVIN observation dict with 'rgb_static' and 'rgb_gripper' keys
            step: Step number for filename (uses internal counter if None)

        Returns:
            Dictionary with camera names as keys and RGB images as values
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        images = {}

        # Extract and save static camera
        rgb_static = calvin_obs.get('rgb_static')
        if rgb_static is not None:
            rgb_static = self._normalize_image(rgb_static)
            images['static'] = rgb_static

            if self.config.save_images:
                img_static = Image.fromarray(rgb_static)
                static_path = self.output_dir / f"step_{step:04d}_static.png"
                img_static.save(static_path)
                logger.debug(f"Saved static camera: {static_path}")

        # Extract and save gripper camera
        rgb_gripper = calvin_obs.get('rgb_gripper')
        if rgb_gripper is not None:
            rgb_gripper = self._normalize_image(rgb_gripper)
            images['gripper'] = rgb_gripper

            if self.config.save_images:
                img_gripper = Image.fromarray(rgb_gripper)
                gripper_path = self.output_dir / f"step_{step:04d}_gripper.png"
                img_gripper.save(gripper_path)
                logger.debug(f"Saved gripper camera: {gripper_path}")

        # Display live if requested
        if self.config.show_live and images:
            self.display_cameras(images)

        return images

    def display_cameras(self, images: Dict[str, np.ndarray]):
        """Display camera views side-by-side using matplotlib."""
        n_cameras = len(images)
        if n_cameras == 0:
            return

        if self.config.display_mode == "side_by_side":
            fig, axes = plt.subplots(1, n_cameras, figsize=(6 * n_cameras, 6))
            if n_cameras == 1:
                axes = [axes]

            for ax, (name, img) in zip(axes, images.items()):
                ax.imshow(img)
                ax.set_title(f"{name.capitalize()} Camera", fontsize=14)
                ax.axis('off')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)  # Non-blocking update

        elif self.config.display_mode == "overlay":
            # Simple overlay mode - just show most recent frame
            for name, img in images.items():
                plt.clf()
                plt.imshow(img)
                plt.title(f"{name.capitalize()} Camera")
                plt.axis('off')
                plt.draw()
                plt.pause(0.001)

    @staticmethod
    def _normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 [0, 255] range."""
        if img.max() <= 1.0:
            return (img * 255).astype(np.uint8)
        return img.astype(np.uint8)

    def reset(self):
        """Reset step counter for new episode."""
        self.step_counter = 0
