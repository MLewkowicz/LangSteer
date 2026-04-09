"""Analytical/Tweedie guidance steering implementation."""

import logging
from typing import Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from core.steering import BaseSteering

logger = logging.getLogger(__name__)


class TweedieSteering(BaseSteering):
    """
    Tweedie guidance for steering diffusion policies.
    Uses Tweedie's formula to predict clean trajectories and compute
    analytical gradient guidance toward reference trajectories.

    Operates in dual-scheduler mode (position/rotation) with epsilon prediction.
    Reference trajectories are in normalized pos(3) + rot_6d(6) space.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.guidance_strength = cfg.get("guidance_strength", 1.0)
        self.horizon = cfg.get("horizon", 16)
        self.device = cfg.get("device", "cuda")

        # Timestep scaling
        self.use_timestep_scaling = cfg.get("use_timestep_scaling", True)
        self.min_timestep_scale = cfg.get("min_timestep_scale", 0.1)

        # Sliding window tracking
        self.reference_trajectory = None  # (T, 9): abs_pos(3) + rot_6d(6)
        self.current_episode_step = 0
        self.position_scheduler = None
        self.rotation_scheduler = None
        self.trajectory_loader = None

        # DiffuserActor coordinate conversion state
        self._gripper_loc_bounds = None  # (2, 3) tensor for position normalization
        self._current_gripper_pos = None  # (3,) absolute gripper position
        self._is_relative = cfg.get("relative", True)

        glb = cfg.get("gripper_loc_bounds", None)
        if glb is not None:
            self._gripper_loc_bounds = torch.tensor(glb, dtype=torch.float32)

        logger.info(
            f"Initialized TweedieSteering: "
            f"strength={self.guidance_strength}, "
            f"horizon={self.horizon}, "
            f"timestep_scaling={self.use_timestep_scaling}"
        )

    def set_position_scheduler(self, scheduler):
        """Store reference to position noise scheduler."""
        self.position_scheduler = scheduler
        logger.debug("Set position scheduler for Tweedie steering")

    def set_rotation_scheduler(self, scheduler):
        """Store reference to rotation noise scheduler."""
        self.rotation_scheduler = scheduler
        logger.debug("Set rotation scheduler for Tweedie steering")

    def set_trajectory_loader(self, loader):
        """Store reference trajectory loader."""
        self.trajectory_loader = loader
        logger.debug("Set trajectory loader for Tweedie steering")

    def set_current_gripper_pos(self, gripper_pos: np.ndarray):
        """
        Set current absolute gripper position for relative coordinate conversion.
        Called by policy wrapper before each forward pass.
        """
        self._current_gripper_pos = torch.tensor(
            gripper_pos, dtype=torch.float32, device=self.device
        )

    def setup_episode(self, task_name: str):
        """
        Load reference trajectory for new episode and reset step counter.
        Converts reference to normalized-6D space.
        """
        self.current_episode_step = 0

        if self.trajectory_loader is None:
            logger.warning("No trajectory loader set, steering disabled for this episode")
            self.reference_trajectory = None
            return None, None

        trajectory_data = self.trajectory_loader.load_trajectory_for_task(task_name)

        if trajectory_data is None:
            logger.warning(f"No reference trajectory found for task: {task_name}")
            self.reference_trajectory = None
            return None, None

        self._setup_reference(trajectory_data)

        logger.info(
            f"Loaded reference trajectory for '{task_name}': "
            f"{self.reference_trajectory.shape[0]} frames, "
            f"shape={self.reference_trajectory.shape}"
        )

        return trajectory_data['robot_obs_init'], trajectory_data['scene_obs_init']

    def _setup_reference(self, trajectory_data):
        """
        Setup reference from CALVIN robot_obs (absolute poses).

        Converts to model-internal space:
        - Position: absolute XYZ (NOT normalized/relative yet - done per guidance call)
        - Rotation: euler XYZ -> quaternion wxyz -> rotation matrix -> 6D

        Stored as (T, 9): abs_pos(3) + rot_6d(6)
        """
        from policies.diffuser_actor_components.rotation_utils import (
            quaternion_to_matrix,
            get_ortho6d_from_rotation_matrix,
            normalise_quat,
        )
        from training.policies.diffuser_actor.preprocessing.calvin_utils import (
            convert_rotation,
        )

        robot_obs = trajectory_data['robot_obs']  # (T, 15)
        abs_pos = robot_obs[:, :3]  # (T, 3) absolute XYZ
        euler_xyz = robot_obs[:, 3:6]  # (T, 3) euler XYZ

        quats = []
        for i in range(len(euler_xyz)):
            q = convert_rotation(euler_xyz[i])  # (4,) wxyz
            quats.append(q)
        quats = np.stack(quats)  # (T, 4)

        quats_t = torch.from_numpy(quats).float()
        quats_t = normalise_quat(quats_t)
        rot_matrices = quaternion_to_matrix(quats_t)  # (T, 3, 3)
        rot_6d = get_ortho6d_from_rotation_matrix(rot_matrices)  # (T, 6)

        abs_pos_t = torch.from_numpy(abs_pos).float()
        self.reference_trajectory = torch.cat(
            [abs_pos_t, rot_6d], dim=-1
        ).to(self.device)

        logger.debug(
            f"Reference: pos range "
            f"[{abs_pos_t.min(0).values.numpy()}, {abs_pos_t.max(0).values.numpy()}]"
        )

    def increment_step(self):
        """Advance the sliding window by one step."""
        self.current_episode_step += 1

    def get_guidance(self, current_sample: torch.Tensor, timestep: int,
                     obs_embedding: Any, model_output: torch.Tensor) -> torch.Tensor:
        """
        Compute Tweedie guidance gradient toward reference trajectory.

        Uses Tweedie's formula to predict x_0 from (x_t, epsilon), then computes
        the analytical gradient of MSE(x_0_pred, ref) w.r.t. epsilon.

        For epsilon prediction: x_0 = (x_t - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)
        Modifying eps by d_eps shifts x_0 by: -sqrt(1-alpha_bar)/sqrt(alpha_bar) * d_eps
        So d_eps = strength * sqrt(1-alpha_bar)/sqrt(alpha_bar) * (x_0_pred - ref) steers x_0 toward ref.
        """
        if self.reference_trajectory is None:
            return torch.zeros_like(model_output)

        ref_window = self._get_reference_window()  # (horizon, 9): abs_pos + rot_6d
        if ref_window is None:
            return torch.zeros_like(model_output)

        # Convert reference position to model-internal space (relative + normalized)
        ref_pos = ref_window[:, :3].clone()  # (horizon, 3)
        ref_rot = ref_window[:, 3:9].clone()  # (horizon, 6)

        if self._is_relative and self._current_gripper_pos is not None:
            ref_pos = ref_pos - self._current_gripper_pos.unsqueeze(0)

        if self._gripper_loc_bounds is not None:
            bounds = self._gripper_loc_bounds.to(ref_pos.device)
            pos_min = bounds[0]
            pos_max = bounds[1]
            ref_pos = (ref_pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

        # Build reference in model space: (1, horizon, 9)
        ref_model_space = torch.cat([ref_pos, ref_rot], dim=-1).unsqueeze(0)

        scale = self._compute_timestep_scale(timestep)

        if isinstance(timestep, torch.Tensor):
            t_idx = timestep.long()
        else:
            t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)

        alpha_bar_pos = torch.clamp(
            self.position_scheduler.alphas_cumprod[t_idx], min=1e-6
        )
        alpha_bar_rot = torch.clamp(
            self.rotation_scheduler.alphas_cumprod[t_idx], min=1e-6
        )

        # Tweedie: predict x_0 from x_t and epsilon
        eps_pos = model_output[:, :self.horizon, :3]
        eps_rot = model_output[:, :self.horizon, 3:9]
        x_t_pos = current_sample[:, :self.horizon, :3]
        x_t_rot = current_sample[:, :self.horizon, 3:9]

        x_0_pos = (x_t_pos - torch.sqrt(1 - alpha_bar_pos) * eps_pos) / torch.sqrt(alpha_bar_pos)
        x_0_rot = (x_t_rot - torch.sqrt(1 - alpha_bar_rot) * eps_rot) / torch.sqrt(alpha_bar_rot)

        # Analytical gradient in epsilon space
        coeff_pos = torch.sqrt(1 - alpha_bar_pos) / torch.sqrt(alpha_bar_pos)
        coeff_rot = torch.sqrt(1 - alpha_bar_rot) / torch.sqrt(alpha_bar_rot)

        delta_eps_pos = self.guidance_strength * scale * coeff_pos * (x_0_pos - ref_model_space[..., :3])
        delta_eps_rot = self.guidance_strength * scale * coeff_rot * (x_0_rot - ref_model_space[..., 3:9])

        # Build full guidance tensor matching model_output shape
        B, L, D = model_output.shape
        guidance = torch.zeros_like(model_output)
        h = min(self.horizon, L)
        guidance[:, :h, :3] = delta_eps_pos[:, :h]
        guidance[:, :h, 3:9] = delta_eps_rot[:, :h]

        # Logging
        mse_pos = F.mse_loss(x_0_pos, ref_model_space[..., :3], reduction='mean')
        mse_rot = F.mse_loss(x_0_rot, ref_model_space[..., 3:9], reduction='mean')

        if self.current_episode_step % 10 == 0:
            logger.debug(
                f"Step {self.current_episode_step}, t={timestep}: "
                f"MSE_pos={mse_pos.item():.4f}, MSE_rot={mse_rot.item():.4f}, "
                f"guidance_norm={torch.norm(guidance).item():.4f}, "
                f"scale={scale:.4f}, coeff_pos={coeff_pos.item():.4f}"
            )

        return guidance

    def _get_reference_window(self) -> Optional[torch.Tensor]:
        """Get current window from reference trajectory using sliding window."""
        if self.reference_trajectory is None:
            return None

        start_idx = self.current_episode_step
        end_idx = start_idx + self.horizon

        if start_idx >= len(self.reference_trajectory):
            logger.warning(
                f"Episode step {start_idx} beyond reference trajectory "
                f"length {len(self.reference_trajectory)}"
            )
            return None

        if end_idx <= len(self.reference_trajectory):
            window = self.reference_trajectory[start_idx:end_idx]
        else:
            available = self.reference_trajectory[start_idx:]
            pad_length = self.horizon - len(available)

            if available.dim() == 1:
                available = available.unsqueeze(0)

            last_frame_row = self.reference_trajectory[-1]
            if last_frame_row.dim() != 1:
                last_frame_row = last_frame_row.squeeze()

            last_frame = last_frame_row.unsqueeze(0).expand(pad_length, -1)

            assert available.shape[1] == last_frame.shape[1], \
                f"Dimension mismatch: available={available.shape}, last_frame={last_frame.shape}"

            window = torch.cat([available, last_frame], dim=0)

            logger.debug(
                f"Padded reference window: {len(available)} frames + "
                f"{pad_length} repeated frames"
            )

        return window

    def _compute_timestep_scale(self, timestep: int) -> float:
        """Compute guidance scale based on diffusion timestep."""
        if not self.use_timestep_scaling:
            return 1.0

        if isinstance(timestep, torch.Tensor):
            t = timestep.item()
        else:
            t = timestep

        if self.position_scheduler is None:
            return 1.0

        max_timestep = self.position_scheduler.config.num_train_timesteps
        normalized_t = t / max_timestep

        scale = self.min_timestep_scale + (1.0 - self.min_timestep_scale) * normalized_t
        return scale
