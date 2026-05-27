"""VLS (Value-based Learned Steering) — VLM-generated reward guidance.

Integrates the VLS framework into LangSteer as a BaseSteering implementation.

At episode start:
  1. Keypoints are detected from the Isaac Sim observation (world-frame 3-D
     positions via the pre-computed depth-XYZ map).
  2. A vision-language model generates per-stage Python reward functions:
     ``f(keypoints: (K,3), trajectory: (B,T,3)) → scalar``.
  3. Stages are loaded into a lightweight ``VLSStageManager`` that advances
     via proximity-based transitions (same pattern as VoxPoser).

During each denoising step:
  - Tweedie's formula recovers x₀ from x_t and ε.
  - The position slice is denormalized to world frame (via ``PositionTransform``).
  - Autograd backpropagates the reward gradient back through the position tensor.
  - The gradient is converted to ε-space (via ``epsilon_coeff``) and returned.
  - The rotation slice is left unchanged (zero delta).

All math reuses existing LangSteer utilities (``diffusion_utils``,
``coordinates``); no VLS repo imports.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

from core.steering import BaseSteering
from steering.coordinates import PositionTransform
from steering.diffusion_utils import epsilon_coeff, get_alpha_bar, predict_x0
from steering.vls.guidance_generator import VLSGuidanceGenerator
from steering.vls.stage_manager import VLSStageManager

logger = logging.getLogger(__name__)


class VLSSteering(BaseSteering):
    """Steer diffusion policies using VLM-generated reward functions.

    Guidance functions are Python callables produced by GPT-4o at episode
    start.  They operate on the Tweedie-predicted clean trajectory in world
    frame, so no policy retraining is needed.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.guidance_strength: float = cfg.get("guidance_strength", 15.0)
        self.prediction_type: str = cfg.get("prediction_type", "epsilon")
        self.guidance_mode: str = cfg.get("guidance_mode", "epsilon")
        self.start_guidance_timestep: int = cfg.get("start_guidance_timestep", 50)
        self.device: str = cfg.get("device", "cuda")
        self.horizon: int = cfg.get("horizon", 16)

        workspace_min = np.array(
            cfg.get("workspace_bounds_min", [0.0, -0.5, 0.7]), dtype=np.float32
        )
        workspace_max = np.array(
            cfg.get("workspace_bounds_max", [1.0, 0.5, 1.3]), dtype=np.float32
        )

        self._coords = PositionTransform(
            gripper_loc_bounds=cfg.get("gripper_loc_bounds", None),
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            is_relative=cfg.get("relative", True),
            device=self.device,
        )

        self._stage_manager = VLSStageManager(
            proximity_threshold=cfg.get("stage_proximity_threshold", 0.08),
            use_grasp_gate=cfg.get("use_grasp_gate", False),
            grasp_min_width=cfg.get("grasp_min_width", 0.01),
            grasp_max_width=cfg.get("grasp_max_width", 0.04),
            grasp_stability_window=cfg.get("grasp_stability_window", 3),
        )

        self._generator = VLSGuidanceGenerator(
            llm_model=cfg.get("llm_model", "gpt-4o"),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 3000),
            cache_dir=cfg.get("cache_dir", "outputs/vls_cache"),
            api_key=cfg.get("api_key", None),
        )

        # Schedulers — wired by the policy wrapper before inference.
        self.position_scheduler: Any = None
        self.rotation_scheduler: Any = None

        # Episode state
        self._episode_step: int = 0
        self._keypoints: Optional[torch.Tensor] = None  # (K, 3) on device
        self._keypoint_names: list[str] = []

        logger.info(
            f"VLSSteering: strength={self.guidance_strength}, "
            f"mode={self.guidance_mode}, start_t={self.start_guidance_timestep}"
        )

    # ------------------------------------------------------------------
    # Scheduler setters (called by policy wrapper)
    # ------------------------------------------------------------------

    def set_position_scheduler(self, scheduler: Any) -> None:
        self.position_scheduler = scheduler

    def set_rotation_scheduler(self, scheduler: Any) -> None:
        self.rotation_scheduler = scheduler

    # ------------------------------------------------------------------
    # Per-step state setters
    # ------------------------------------------------------------------

    def set_current_gripper_pos(self, gripper_pos: np.ndarray) -> None:
        """Update the gripper position used to denormalize x₀ positions."""
        self._coords.set_gripper_pos(gripper_pos)

    def set_current_gripper_rotation(self, gripper_rot: Any) -> None:
        """No-op — VLS currently provides position-only guidance."""

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def setup_episode(self, obs: Any, task_name: str) -> tuple[None, None]:
        """Detect keypoints and generate guidance functions for a new episode.

        Args:
            obs: ``Observation`` from the Isaac environment.
            task_name: Task name / instruction string.

        Returns:
            (None, None) for compatibility with the run_experiment callback pattern.
        """
        self._episode_step = 0
        self._keypoints = None
        self._keypoint_names = []
        self._stage_manager.setup([])

        instruction = getattr(obs, "instruction", task_name) or task_name

        # -- Keypoint detection --
        from openai import OpenAI
        import os

        api_key = self.cfg.get("api_key", None) or os.environ.get("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=api_key)
        llm_model = self.cfg.get("llm_model", "gpt-4o")

        from steering.vls import keypoint_handler

        names, positions_world = keypoint_handler.detect(
            obs, instruction, openai_client, model=llm_model
        )

        if len(names) == 0:
            logger.warning("VLS: no keypoints detected, steering will be inactive")
            return None, None

        self._keypoint_names = names
        self._keypoints = torch.tensor(
            positions_world, dtype=torch.float32, device=self.device
        )

        # Build an annotated image for the guidance-generation prompt
        image_rgb = obs.rgb.get("static")
        if image_rgb is not None:
            annotated = keypoint_handler.overlay_keypoints(
                image_rgb, names, np.zeros((len(names), 2), dtype=np.float32)
            )
        else:
            annotated = np.zeros((200, 200, 3), dtype=np.uint8)

        # -- Guidance generation --
        stages = self._generator.generate(
            instruction=instruction,
            image_rgb=annotated
            if image_rgb is not None
            else np.zeros((200, 200, 3), dtype=np.uint8),
            keypoint_names=names,
            keypoint_positions=positions_world,
            task_name=task_name,
        )

        if not stages:
            logger.warning("VLS: no guidance stages generated, steering inactive")
            return None, None

        self._stage_manager.setup(stages)
        logger.info(
            f"VLS: episode setup complete — {len(names)} keypoints, "
            f"{len(stages)} stage(s)"
        )
        return None, None

    def check_stage_transition(self, ee_pos: np.ndarray, gripper_width: float) -> bool:
        return self._stage_manager.check_transition(ee_pos, gripper_width)

    def increment_step(self) -> None:
        self._episode_step += 1
        self._stage_manager.increment_step()

    # ------------------------------------------------------------------
    # Core guidance computation
    # ------------------------------------------------------------------

    def get_guidance(
        self,
        current_sample: torch.Tensor,
        timestep: int,
        obs_embedding: Any,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VLM-reward gradient as an ε-space correction.

        Steps:
          1. Gate on ``start_guidance_timestep`` (skip high-noise steps).
          2. Retrieve the active stage's guidance function.
          3. Tweedie-predict x₀ from (x_t, ε).
          4. Denormalize position slice to world frame.
          5. Autograd through ``guidance_fn(keypoints, pos_world)`` → gradient.
          6. Chain-rule gradient back to ε space.
        """
        guidance = torch.zeros_like(model_output)

        t = (
            int(timestep.item())
            if isinstance(timestep, torch.Tensor)
            else int(timestep)
        )
        if t > self.start_guidance_timestep:
            return guidance

        stage = self._stage_manager.current()
        if stage.guidance_fn is None or self._keypoints is None:
            return guidance

        alpha_bar = get_alpha_bar(self.position_scheduler, timestep, device=self.device)

        # Predict clean trajectory x₀ — shape (B, H, 9)
        x0_pred = predict_x0(
            current_sample,
            model_output,
            alpha_bar,
            prediction_type=self.prediction_type,
        )

        # Position slice: (B, H, 3) in model [-1, 1] space
        pos_model = x0_pred[..., :3].detach()

        # Denormalize to world frame
        pos_world = self._coords.model_to_world(pos_model)  # (B, H, 3)
        pos_world = pos_world.requires_grad_(True)

        # Evaluate reward and backprop
        kp = self._keypoints.to(pos_world.device)
        try:
            reward = stage.guidance_fn(kp, pos_world)
            if reward is None or not isinstance(reward, torch.Tensor):
                return guidance
            if not reward.requires_grad:
                reward = reward + 0.0 * pos_world.sum()
            reward.backward()
        except Exception as e:
            logger.warning(f"VLS: guidance_fn forward/backward failed: {e}")
            return guidance

        if pos_world.grad is None:
            return guidance

        grad_world = pos_world.grad  # (B, H, 3)

        # Chain-rule: world → model space
        grad_model = self._coords.world_gradient_to_model(grad_world)

        # ε-space correction: Δε = strength · (√ᾱ / √(1−ᾱ)) · Δx₀
        coeff = epsilon_coeff(alpha_bar)
        delta_eps_pos = self.guidance_strength * coeff * grad_model  # (B, H, 3)

        guidance[..., :3] = delta_eps_pos.detach()
        return guidance
