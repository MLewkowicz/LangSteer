"""DP3 (Diffusion Policy with 3D Point Clouds) - Core Policy Logic.

Extracted from:
- diffusion_policy_3d/policy/dp3.py
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint

from policies.dp3_components.utils import sample_farthest_points as torch3d_ops_sample_farthest_points
from policies.dp3_components.normalizer import LinearNormalizer
from policies.dp3_components.unet import ConditionalUnet1D
from policies.dp3_components.encoder import DP3Encoder
from policies.dp3_components.utils import dict_apply

# Create a mock module for compatibility
class MockOps:
    sample_farthest_points = staticmethod(torch3d_ops_sample_farthest_points)
torch3d_ops = MockOps()


class DP3BasePolicy(nn.Module):
    """
    Base policy class for DP3 components.

    Note: Renamed from 'BasePolicy' to avoid conflict with LangSteer's core.policy.BasePolicy.
    Provides device and dtype properties for convenience.
    """
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class DP3(DP3BasePolicy):
    """
    DP3: Diffusion Policy with 3D Point Clouds.
    Main policy class integrating observation encoder, diffusion model, and normalizer.
    """

    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # parameters passed to step
            **kwargs):
        """
        Args:
            shape_meta: Dictionary defining observation and action shapes
            noise_scheduler: Diffusion noise scheduler (DDPM/DDIM)
            horizon: Prediction horizon
            n_action_steps: Number of action steps to execute
            n_obs_steps: Number of observation steps to condition on
            num_inference_steps: Number of diffusion denoising steps (default: same as training)
            obs_as_global_cond: Whether to use observations as global conditioning
            diffusion_step_embed_dim: Timestep embedding dimension
            down_dims: Dimensions for U-Net downsampling path
            kernel_size: Convolution kernel size
            n_groups: Number of groups for GroupNorm
            condition_type: Conditioning type ('film', 'add', 'cross_attention')
            use_down_condition: Whether to condition downsampling path
            use_mid_condition: Whether to condition middle blocks
            use_up_condition: Whether to condition upsampling path
            encoder_output_dim: Output dimension of observation encoder
            crop_shape: Image crop shape (unused for point clouds)
            use_pc_color: Whether to use RGB colors in point clouds
            pointnet_type: Type of PointNet encoder
            pointcloud_encoder_cfg: Configuration for PointNet encoder
        """
        super().__init__()

        self.condition_type = condition_type

        # Parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = {key: val['shape'] for key, val in obs_shape_meta.items()}

        # Initialize observation encoder
        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                 img_crop_shape=crop_shape,
                                 out_channel=encoder_output_dim,
                                 pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                 use_pc_color=use_pc_color,
                                 pointnet_type=pointnet_type,
                                 )

        # Create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        # Normalizer starts as identity and must be set via:
        # 1. set_normalizer() method, OR
        # 2. Loaded from checkpoint in load_checkpoint()
        # If neither happens, model uses identity normalization (no-op).
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        cprint(f"[DP3] Initialized with {sum(p.numel() for p in self.parameters())} parameters", "green")

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            guidance_fn=None,  # STEERING INJECTION POINT
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        Conditional diffusion sampling with optional guidance.

        Args:
            condition_data: Conditioning data tensor
            condition_mask: Mask for conditioning
            local_cond: Local conditioning (optional)
            global_cond: Global conditioning (optional)
            generator: Random generator for sampling
            guidance_fn: Optional guidance function for steering
                        Signature: fn(trajectory, timestep, obs_embedding) -> guidance_gradient
            **kwargs: Additional arguments for scheduler.step

        Returns:
            Sampled trajectory tensor
        """
        model = self.model
        scheduler = self.noise_scheduler

        # Initialize with random noise
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # Set diffusion timesteps
        scheduler.set_timesteps(self.num_inference_steps)

        # Denoising loop
        for t in scheduler.timesteps:
            # 1. Apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. Predict noise/sample
            model_output = model(sample=trajectory,
                                timestep=t,
                                local_cond=local_cond,
                                global_cond=global_cond)

            # 2.5 Apply steering guidance (if provided)
            if guidance_fn is not None:
                guidance = guidance_fn(trajectory, t, global_cond, model_output)
                model_output = model_output + guidance

            # 3. Compute previous sample: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory).prev_sample

        # Finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], guidance_fn=None) -> Dict[str, torch.Tensor]:
        """
        Predict action from observations.

        Args:
            obs_dict: Dictionary of observations
                - 'point_cloud': (B, T_obs, N, C) point cloud
                - 'agent_pos': (B, T_obs, D) agent state
            guidance_fn: Optional guidance function for steering

        Returns:
            Dictionary containing:
                - 'action': (B, n_action_steps, action_dim) predicted actions
                - 'action_pred': (B, horizon, action_dim) full horizon predictions
        """
        # Normalize input
        nobs = self.normalizer.normalize(obs_dict)

        # Handle point cloud color
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # Build input
        device = self.device
        dtype = self.dtype

        # Handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # Condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # Treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # Reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # Empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # Condition through inpainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # Reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # Run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            guidance_fn=guidance_fn,  # Pass guidance function
            **self.kwargs)

        # Unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer from external source (e.g., from checkpoint)."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        Compute training loss (for completeness, not used in inference).

        Args:
            batch: Dictionary containing 'obs' and 'action' keys

        Returns:
            loss: Scalar loss value
            loss_dict: Dictionary of loss components
        """
        # Note: Mask generator is removed from minimal version
        # This is kept for checkpoint compatibility but may need mask_generator
        # for actual training

        # Normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # Handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # Reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # Treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # Reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # Reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # Reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # Sample noise
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # Sample random timestep
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()

        # Add noise (forward diffusion)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # Predict the noise residual/sample
        pred = self.model(sample=noisy_trajectory,
                        timestep=timesteps,
                        local_cond=local_cond,
                        global_cond=global_cond)

        # Determine target based on prediction type
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # V-prediction: velocity prediction
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='mean')

        loss_dict = {
            'bc_loss': loss.item(),
        }

        return loss, loss_dict
