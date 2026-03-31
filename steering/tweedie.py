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
    gradient guidance toward reference trajectories.

    Supports two trajectory formats:
    - "dp3": Single scheduler, (B, H, 7) trajectories, prediction_type from config
    - "diffuser_actor": Dual schedulers (position/rotation), (B, L, 9) trajectories,
      epsilon prediction, reference in normalized pos(3) + rot_6d(6) space
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Guidance parameters
        self.guidance_strength = cfg.get("guidance_strength", 1.0)
        self.horizon = cfg.get("horizon", 16)
        self.prediction_type = cfg.get("prediction_type", "sample")
        self.device = cfg.get("device", "cuda")
        self.trajectory_format = cfg.get("trajectory_format", "dp3")

        # Timestep scaling configuration
        self.use_timestep_scaling = cfg.get("use_timestep_scaling", True)
        self.min_timestep_scale = cfg.get("min_timestep_scale", 0.1)

        # Sliding window tracking
        self.reference_trajectory = None  # (T, D) where D=7 for dp3, D=9 for diffuser_actor
        self.current_episode_step = 0
        self.scheduler = None  # DP3: single scheduler
        self.position_scheduler = None  # DiffuserActor: position scheduler
        self.rotation_scheduler = None  # DiffuserActor: rotation scheduler
        self.trajectory_loader = None

        # DiffuserActor-specific state
        self._gripper_loc_bounds = None  # (2, 3) tensor for position normalization
        self._current_gripper_pos = None  # (3,) absolute gripper position for relative conversion
        self._is_relative = cfg.get("relative", True)

        # Store gripper_loc_bounds from config if provided
        glb = cfg.get("gripper_loc_bounds", None)
        if glb is not None:
            self._gripper_loc_bounds = torch.tensor(glb, dtype=torch.float32)

        logger.info(
            f"Initialized TweedieSteering: "
            f"strength={self.guidance_strength}, "
            f"horizon={self.horizon}, "
            f"format={self.trajectory_format}, "
            f"prediction_type={self.prediction_type}, "
            f"timestep_scaling={self.use_timestep_scaling}"
        )

    def set_scheduler(self, scheduler):
        """Store reference to noise scheduler (DP3 single-scheduler mode)."""
        self.scheduler = scheduler
        logger.debug("Set scheduler reference for Tweedie steering")

    def set_position_scheduler(self, scheduler):
        """Store reference to position noise scheduler (DiffuserActor)."""
        self.position_scheduler = scheduler
        # Also set as default scheduler for timestep scaling
        self.scheduler = scheduler
        logger.debug("Set position scheduler for Tweedie steering")

    def set_rotation_scheduler(self, scheduler):
        """Store reference to rotation noise scheduler (DiffuserActor)."""
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

        Args:
            gripper_pos: (3,) absolute gripper XYZ position
        """
        self._current_gripper_pos = torch.tensor(
            gripper_pos, dtype=torch.float32, device=self.device
        )

    def setup_episode(self, task_name: str):
        """
        Load reference trajectory for new episode and reset step counter.

        For diffuser_actor format, converts reference to normalized-6D space.
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

        if self.trajectory_format == "diffuser_actor":
            self._setup_diffuser_actor_reference(trajectory_data)
        else:
            self._setup_dp3_reference(trajectory_data)

        logger.info(
            f"Loaded reference trajectory for '{task_name}': "
            f"{self.reference_trajectory.shape[0]} frames, "
            f"shape={self.reference_trajectory.shape}, "
            f"format={self.trajectory_format}"
        )

        return trajectory_data['robot_obs_init'], trajectory_data['scene_obs_init']

    def _setup_dp3_reference(self, trajectory_data):
        """Setup reference from DP3-format actions (relative deltas)."""
        actions = trajectory_data['actions']  # (T, 7) numpy
        self.reference_trajectory = torch.from_numpy(actions).float().to(self.device)
        if self.reference_trajectory.dim() == 1:
            self.reference_trajectory = self.reference_trajectory.unsqueeze(0)

    def _setup_diffuser_actor_reference(self, trajectory_data):
        """
        Setup reference from CALVIN robot_obs (absolute poses).

        Converts to model-internal space:
        - Position: absolute XYZ (NOT normalized/relative yet - done per guidance call)
        - Rotation: euler XYZ → quaternion wxyz → rotation matrix → 6D

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

        # Convert euler → quaternion (wxyz) → 6D rotation
        # convert_rotation does euler → wxyz quat via pytorch3d
        quats = []
        for i in range(len(euler_xyz)):
            q = convert_rotation(euler_xyz[i])  # (4,) wxyz
            quats.append(q)
        quats = np.stack(quats)  # (T, 4)

        quats_t = torch.from_numpy(quats).float()
        quats_t = normalise_quat(quats_t)
        rot_matrices = quaternion_to_matrix(quats_t)  # (T, 3, 3)
        rot_6d = get_ortho6d_from_rotation_matrix(rot_matrices)  # (T, 6)

        # Store as (T, 9): abs_pos(3) + rot_6d(6)
        abs_pos_t = torch.from_numpy(abs_pos).float()
        self.reference_trajectory = torch.cat(
            [abs_pos_t, rot_6d], dim=-1
        ).to(self.device)

        logger.debug(
            f"DiffuserActor reference: pos range "
            f"[{abs_pos_t.min(0).values.numpy()}, {abs_pos_t.max(0).values.numpy()}]"
        )

    def increment_step(self):
        """Advance the sliding window by one step."""
        self.current_episode_step += 1

    def get_guidance(self, current_sample: torch.Tensor, timestep: int,
                     obs_embedding: Any, model_output: torch.Tensor) -> torch.Tensor:
        """
        Compute Tweedie guidance gradient toward reference trajectory.

        Args:
            current_sample: Noisy trajectory x_t, shape (B, H, D)
            timestep: Current diffusion timestep (int or Tensor)
            obs_embedding: Observation features (unused for dp3; fixed_inputs for diffuser_actor)
            model_output: Model's prediction (epsilon or x_0 depending on type)

        Returns:
            Guidance gradient to add to model output, shape matching model_output
        """
        if self.reference_trajectory is None:
            return torch.zeros_like(model_output)

        if self.trajectory_format == "diffuser_actor":
            return self._get_guidance_diffuser_actor(
                current_sample, timestep, obs_embedding, model_output
            )
        else:
            return self._get_guidance_dp3(
                current_sample, timestep, obs_embedding, model_output
            )

    def _get_guidance_dp3(self, current_sample, timestep, obs_embedding,
                          model_output):
        """Original DP3 guidance path."""
        ref_window = self._get_reference_window()
        if ref_window is None:
            return torch.zeros_like(model_output)

        ref_batch = ref_window.unsqueeze(0).detach()  # (1, horizon, D)
        scale = self._compute_timestep_scale(timestep)

        if self.prediction_type == "sample":
            pred_traj = model_output[:, :self.horizon, :]
            mse_loss = F.mse_loss(pred_traj, ref_batch, reduction='mean')

            guidance = -self.guidance_strength * scale * (pred_traj - ref_batch)

            if model_output.shape[1] > self.horizon:
                padding = torch.zeros(
                    (model_output.shape[0], model_output.shape[1] - self.horizon, model_output.shape[2]),
                    device=model_output.device
                )
                guidance = torch.cat([guidance, padding], dim=1)
        else:
            current_sample = current_sample.clone().requires_grad_(True)
            x_0_pred = self._predict_x0(current_sample, timestep, model_output)
            pred_traj = x_0_pred[:, :self.horizon, :]

            mse_loss = F.mse_loss(pred_traj, ref_batch, reduction='mean')

            gradient = torch.autograd.grad(
                mse_loss, current_sample, create_graph=False
            )[0]
            guidance = -self.guidance_strength * scale * gradient

        if self.current_episode_step % 10 == 0:
            logger.debug(
                f"Step {self.current_episode_step}, timestep {timestep}: "
                f"MSE={mse_loss.item():.4f}, "
                f"guidance_norm={torch.norm(guidance).item():.4f}, "
                f"scale={scale:.4f}"
            )

        return guidance.detach()

    def _get_guidance_diffuser_actor(self, current_sample, timestep,
                                     fixed_inputs, model_output):
        """
        DiffuserActor guidance: analytical epsilon-space guidance with dual schedulers.

        Uses Tweedie's formula to predict x_0 from (x_t, epsilon), then computes
        the analytical gradient of MSE(x_0_pred, ref) w.r.t. epsilon. This avoids
        autograd and works inside torch.no_grad() context.

        For epsilon prediction: x_0 = (x_t - sqrt(1-ᾱ)·ε) / sqrt(ᾱ)
        Modifying ε by δε shifts x_0 by: -sqrt(1-ᾱ)/sqrt(ᾱ) · δε
        So δε = strength · sqrt(1-ᾱ)/sqrt(ᾱ) · (x_0_pred - ref) steers x_0 toward ref.
        """
        ref_window = self._get_reference_window()  # (horizon, 9): abs_pos + rot_6d
        if ref_window is None:
            return torch.zeros_like(model_output)

        # Convert reference position to model-internal space (relative + normalized)
        ref_pos = ref_window[:, :3].clone()  # (horizon, 3)
        ref_rot = ref_window[:, 3:9].clone()  # (horizon, 6)

        # Relative conversion: subtract current gripper position
        if self._is_relative and self._current_gripper_pos is not None:
            ref_pos = ref_pos - self._current_gripper_pos.unsqueeze(0)

        # Normalize position to [-1, 1] using gripper_loc_bounds
        if self._gripper_loc_bounds is not None:
            bounds = self._gripper_loc_bounds.to(ref_pos.device)
            pos_min = bounds[0]
            pos_max = bounds[1]
            ref_pos = (ref_pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

        # Build reference in model space: (1, horizon, 9)
        ref_model_space = torch.cat([ref_pos, ref_rot], dim=-1).unsqueeze(0)

        scale = self._compute_timestep_scale(timestep)

        # Get alpha_bar values for Tweedie prediction
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

        # Analytical gradient in epsilon space:
        # δε = strength · scale · sqrt(1-ᾱ)/sqrt(ᾱ) · (x_0_pred - ref)
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
        # dims 9+ (openness) get zero guidance

        # Compute MSE for logging
        mse_pos = F.mse_loss(x_0_pos, ref_model_space[..., :3], reduction='mean')
        mse_rot = F.mse_loss(x_0_rot, ref_model_space[..., 3:9], reduction='mean')

        if self.current_episode_step % 10 == 0:
            logger.debug(
                f"[DA] Step {self.current_episode_step}, t={timestep}: "
                f"MSE_pos={mse_pos.item():.4f}, MSE_rot={mse_rot.item():.4f}, "
                f"guidance_norm={torch.norm(guidance).item():.4f}, "
                f"scale={scale:.4f}, coeff_pos={coeff_pos.item():.4f}"
            )

        return guidance

    def _get_reference_window(self) -> Optional[torch.Tensor]:
        """
        Get current window from reference trajectory using sliding window.

        Returns:
            Tensor of shape (horizon, D) or None if out of bounds
        """
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

    def _predict_x0(self, x_t: torch.Tensor, timestep: int,
                    model_output: torch.Tensor) -> torch.Tensor:
        """
        Apply Tweedie's formula (single scheduler, DP3 path).
        """
        if self.prediction_type == "sample":
            return model_output

        elif self.prediction_type == "epsilon":
            if isinstance(timestep, torch.Tensor):
                t_idx = timestep.long()
            else:
                t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)

            alpha_bar = self.scheduler.alphas_cumprod[t_idx]
            alpha_bar = torch.clamp(alpha_bar, min=1e-6)
            alpha_bar = alpha_bar.view(-1, 1, 1)

            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * model_output) / torch.sqrt(alpha_bar)
            return x_0_pred

        elif self.prediction_type == "v_prediction":
            if isinstance(timestep, torch.Tensor):
                t_idx = timestep.long()
            else:
                t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)

            alpha_bar = self.scheduler.alphas_cumprod[t_idx]
            alpha_bar = torch.clamp(alpha_bar, min=1e-6)
            alpha_bar = alpha_bar.view(-1, 1, 1)

            x_0_pred = torch.sqrt(alpha_bar) * x_t - torch.sqrt(1 - alpha_bar) * model_output
            return x_0_pred

        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def _compute_timestep_scale(self, timestep: int) -> float:
        """Compute guidance scale based on diffusion timestep."""
        if not self.use_timestep_scaling:
            return 1.0

        if isinstance(timestep, torch.Tensor):
            t = timestep.item()
        else:
            t = timestep

        # Use position scheduler (or single scheduler) for max timestep
        sched = self.position_scheduler or self.scheduler
        if sched is None:
            return 1.0

        max_timestep = sched.config.num_train_timesteps
        normalized_t = t / max_timestep

        scale = self.min_timestep_scale + (1.0 - self.min_timestep_scale) * normalized_t
        return scale
