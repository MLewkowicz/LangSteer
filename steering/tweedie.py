"""Analytical/Tweedie guidance steering implementation."""

import logging
from typing import Any, Optional
import torch
import torch.nn.functional as F
from core.steering import BaseSteering

logger = logging.getLogger(__name__)


class TweedieSteering(BaseSteering):
    """
    Tweedie guidance for steering diffusion policies.
    Uses Tweedie's formula to predict clean trajectories and compute
    gradient guidance toward reference trajectories.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Guidance parameters
        self.guidance_strength = cfg.get("guidance_strength", 1.0)
        self.horizon = cfg.get("horizon", 16)
        self.prediction_type = cfg.get("prediction_type", "sample")
        self.device = cfg.get("device", "cuda")

        # Timestep scaling configuration
        self.use_timestep_scaling = cfg.get("use_timestep_scaling", True)
        self.min_timestep_scale = cfg.get("min_timestep_scale", 0.1)

        # Sliding window tracking
        self.reference_trajectory = None  # Full trajectory (T, 7)
        self.current_episode_step = 0  # Track episode progress
        self.scheduler = None  # Set externally via set_scheduler()
        self.trajectory_loader = None  # Set externally via set_trajectory_loader()

        logger.info(
            f"Initialized TweedieSteering: "
            f"strength={self.guidance_strength}, "
            f"horizon={self.horizon}, "
            f"prediction_type={self.prediction_type}, "
            f"timestep_scaling={self.use_timestep_scaling}"
        )

    def set_scheduler(self, scheduler):
        """
        Store reference to noise scheduler for alpha values.

        Args:
            scheduler: The diffusion noise scheduler (e.g., DDIMScheduler)
        """
        self.scheduler = scheduler
        logger.debug("Set scheduler reference for Tweedie steering")

    def set_trajectory_loader(self, loader):
        """
        Store reference trajectory loader.

        Args:
            loader: ReferenceTrajectoryLoader instance
        """
        self.trajectory_loader = loader
        logger.debug("Set trajectory loader for Tweedie steering")

    def setup_episode(self, task_name: str):
        """
        Load reference trajectory for new episode and reset step counter.

        Args:
            task_name: Name of the task to load reference trajectory for

        Returns:
            Tuple of (robot_obs, scene_obs) for environment reset,
            or (None, None) if no trajectory found
        """
        # Reset step counter for new episode
        self.current_episode_step = 0

        if self.trajectory_loader is None:
            logger.warning("No trajectory loader set, steering disabled for this episode")
            self.reference_trajectory = None
            return None, None

        # Load trajectory data from dataset
        trajectory_data = self.trajectory_loader.load_trajectory_for_task(task_name)

        if trajectory_data is None:
            logger.warning(f"No reference trajectory found for task: {task_name}")
            self.reference_trajectory = None
            return None, None

        # Store full trajectory as torch tensor
        actions = trajectory_data['actions']  # (T, 7) numpy array
        logger.debug(f"Actions from loader: type={type(actions)}, shape={actions.shape}, dtype={actions.dtype}")

        self.reference_trajectory = torch.from_numpy(actions).float().to(self.device)
        logger.debug(f"After torch conversion: shape={self.reference_trajectory.shape}, dim={self.reference_trajectory.dim()}")

        # Ensure 2D shape (T, 7)
        if self.reference_trajectory.dim() == 1:
            logger.warning(f"Reference trajectory is 1D with shape {self.reference_trajectory.shape}, converting to 2D")
            self.reference_trajectory = self.reference_trajectory.unsqueeze(0)

        logger.info(
            f"Loaded reference trajectory for '{task_name}': "
            f"{self.reference_trajectory.shape[0]} frames, shape={self.reference_trajectory.shape}"
        )

        # Return initial states for environment reset
        return trajectory_data['robot_obs_init'], trajectory_data['scene_obs_init']

    def increment_step(self):
        """
        Advance the sliding window by one step.
        Should be called after each environment step.
        """
        self.current_episode_step += 1

    def get_guidance(self, current_sample: torch.Tensor, timestep: int,
                     obs_embedding: Any, model_output: torch.Tensor) -> torch.Tensor:
        """
        Compute Tweedie guidance gradient toward reference trajectory.

        Args:
            current_sample: Noisy trajectory x_t, shape (B, H, D)
            timestep: Current diffusion timestep (int or Tensor)
            obs_embedding: Observation features (unused)
            model_output: Model's prediction (epsilon, x_0, or v depending on type)

        Returns:
            Guidance gradient to add to model output, shape (B, H, D)
        """
        # No guidance if reference not loaded
        if self.reference_trajectory is None:
            return torch.zeros_like(current_sample)

        # Extract sliding window from reference trajectory
        ref_window = self._get_reference_window()  # (horizon, 7)

        if ref_window is None:
            return torch.zeros_like(current_sample)

        # Compute MSE loss and guidance
        ref_batch = ref_window.unsqueeze(0).detach()  # (1, horizon, 7)

        # Apply timestep-dependent scaling
        scale = self._compute_timestep_scale(timestep)

        if self.prediction_type == "sample":
            # For sample prediction, model_output IS x_0 directly
            # Use direct MSE-based guidance without gradients
            pred_traj = model_output[:, :self.horizon, :]  # (B, horizon, 7)
            mse_loss = F.mse_loss(pred_traj, ref_batch, reduction='mean')

            # Direct guidance: push predicted trajectory toward reference
            # guidance = -(pred - ref) scaled by guidance strength and timestep
            guidance = -self.guidance_strength * scale * (pred_traj - ref_batch)

            # Pad to match full model output shape if needed
            if model_output.shape[1] > self.horizon:
                padding = torch.zeros(
                    (model_output.shape[0], model_output.shape[1] - self.horizon, model_output.shape[2]),
                    device=model_output.device
                )
                guidance = torch.cat([guidance, padding], dim=1)

        else:
            # For epsilon/v_prediction, use Tweedie's formula and compute gradients
            current_sample = current_sample.clone().requires_grad_(True)
            x_0_pred = self._predict_x0(current_sample, timestep, model_output)
            pred_traj = x_0_pred[:, :self.horizon, :]

            mse_loss = F.mse_loss(pred_traj, ref_batch, reduction='mean')

            # Compute gradient w.r.t. current_sample
            gradient = torch.autograd.grad(
                mse_loss, current_sample, create_graph=False
            )[0]

            guidance = -self.guidance_strength * scale * gradient

        # Log guidance statistics periodically
        if self.current_episode_step % 10 == 0:
            logger.debug(
                f"Step {self.current_episode_step}, timestep {timestep}: "
                f"MSE={mse_loss.item():.4f}, "
                f"guidance_norm={torch.norm(guidance).item():.4f}, "
                f"scale={scale:.4f}"
            )

        return guidance.detach()

    def _get_reference_window(self) -> Optional[torch.Tensor]:
        """
        Get current window from reference trajectory using sliding window.

        Returns:
            Tensor of shape (horizon, 7) containing reference actions,
            or None if window is out of bounds
        """
        if self.reference_trajectory is None:
            return None

        start_idx = self.current_episode_step
        end_idx = start_idx + self.horizon

        # Check if we're beyond the reference trajectory
        if start_idx >= len(self.reference_trajectory):
            logger.warning(
                f"Episode step {start_idx} beyond reference trajectory "
                f"length {len(self.reference_trajectory)}"
            )
            return None

        # Extract window with padding if needed
        if end_idx <= len(self.reference_trajectory):
            # Full window available
            window = self.reference_trajectory[start_idx:end_idx]
        else:
            # Need padding - repeat last frame
            available = self.reference_trajectory[start_idx:]
            pad_length = self.horizon - len(available)

            # Ensure available is 2D
            if available.dim() == 1:
                available = available.unsqueeze(0)

            # Get last frame (row) from reference trajectory
            last_frame_row = self.reference_trajectory[-1]  # Shape: (7,)

            # Ensure it's 1D before expanding
            if last_frame_row.dim() != 1:
                last_frame_row = last_frame_row.squeeze()

            # Expand to 2D with proper shape: (pad_length, 7)
            last_frame = last_frame_row.unsqueeze(0).expand(pad_length, -1)  # Use expand instead of repeat

            # Verify shapes before concatenation
            assert available.shape[1] == last_frame.shape[1], \
                f"Dimension mismatch: available={available.shape}, last_frame={last_frame.shape}"

            window = torch.cat([available, last_frame], dim=0)

            logger.debug(
                f"Padded reference window: {len(available)} frames + "
                f"{pad_length} repeated frames"
            )

        return window  # (horizon, 7)

    def _predict_x0(self, x_t: torch.Tensor, timestep: int,
                    model_output: torch.Tensor) -> torch.Tensor:
        """
        Apply Tweedie's formula to predict clean sample from noisy sample.

        Args:
            x_t: Noisy sample at timestep t, shape (B, H, D)
            timestep: Current diffusion timestep
            model_output: Model's prediction (type depends on prediction_type)

        Returns:
            Predicted clean sample x_0, shape (B, H, D)
        """
        if self.prediction_type == "sample":
            # Model directly predicts x_0
            return model_output

        elif self.prediction_type == "epsilon":
            # Model predicts noise, apply Tweedie formula:
            # x_0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)

            # Get alpha values from scheduler
            if isinstance(timestep, torch.Tensor):
                t_idx = timestep.long()
            else:
                t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)

            alpha_bar = self.scheduler.alphas_cumprod[t_idx]

            # Clamp to avoid numerical instability
            alpha_bar = torch.clamp(alpha_bar, min=1e-6)

            # Reshape for broadcasting
            alpha_bar = alpha_bar.view(-1, 1, 1)  # (B, 1, 1)

            # Apply Tweedie formula
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * model_output) / torch.sqrt(alpha_bar)
            return x_0_pred

        elif self.prediction_type == "v_prediction":
            # V-prediction: v = sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x_0
            # Solve for x_0: x_0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v

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
        """
        Compute guidance scale based on diffusion timestep.

        Args:
            timestep: Current diffusion timestep

        Returns:
            Scale factor for guidance (0.0 to 1.0)
        """
        if not self.use_timestep_scaling:
            return 1.0

        # Convert to scalar if tensor
        if isinstance(timestep, torch.Tensor):
            t = timestep.item()
        else:
            t = timestep

        # Normalize to [0, 1] based on scheduler's timestep range
        max_timestep = self.scheduler.config.num_train_timesteps
        normalized_t = t / max_timestep

        # Linear annealing: stronger guidance at later timesteps (cleaner samples)
        scale = self.min_timestep_scale + (1.0 - self.min_timestep_scale) * normalized_t
        return scale
