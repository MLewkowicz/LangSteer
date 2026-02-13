"""DP3 policy wrapper."""

from typing import Optional, Any, Dict
from collections import deque
import logging
import numpy as np
import torch
from core.policy import BasePolicy
from core.types import Observation, Action
from core.steering import BaseSteering

# Import DP3 components at module level
try:
    from policies.dp3_components.dp3_policy import DP3
    from diffusers import DDIMScheduler
except ImportError as e:
    # Will be caught during initialization
    _DP3_IMPORT_ERROR = e
    DP3 = None
    DDIMScheduler = None
else:
    _DP3_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class DP3Policy(BasePolicy):
    """
    Wrapper for DP3 diffusion policy.
    Implements the diffusion denoising loop with optional steering guidance.
    Bridges LangSteer Observation/Action DTOs with DP3 internal format.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Check for import errors
        if _DP3_IMPORT_ERROR is not None:
            logger.error(f"Failed to import DP3 components: {_DP3_IMPORT_ERROR}")
            raise _DP3_IMPORT_ERROR

        # Store config
        self._device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_horizon = cfg.get("obs_horizon", 2)
        self.pred_horizon = cfg.get("pred_horizon", 16)
        self.action_horizon = cfg.get("action_horizon", 8)
        self.num_points = cfg.get("num_points", 2048)

        # Build shape_meta from config (with CALVIN defaults)
        default_shape_meta = {
            'obs': {
                'point_cloud': {'shape': [self.num_points, 3], 'type': 'point_cloud'},
                'agent_pos': {'shape': [15], 'type': 'low_dim'}  # CALVIN robot state dimension
            },
            'action': {'shape': [7]}  # CALVIN action dimension (x, y, z, euler_x, euler_y, euler_z, gripper)
        }
        shape_meta = cfg.get("shape_meta", default_shape_meta)

        # Initialize noise scheduler
        scheduler_cfg = cfg.get("scheduler", {})
        scheduler = DDIMScheduler(
            num_train_timesteps=scheduler_cfg.get("num_train_timesteps", 100),
            beta_start=scheduler_cfg.get("beta_start", 0.0001),
            beta_end=scheduler_cfg.get("beta_end", 0.02),
            beta_schedule=scheduler_cfg.get("beta_schedule", "squaredcos_cap_v2"),
            clip_sample=scheduler_cfg.get("clip_sample", True),
            prediction_type=scheduler_cfg.get("prediction_type", "sample"),
        )

        # Encoder config
        encoder_cfg = cfg.get("encoder", {})
        pointcloud_encoder_cfg = {
            'out_channels': encoder_cfg.get("output_dim", 64),
            'use_layernorm': True,
            'final_norm': 'layernorm',
        }

        # Diffusion config
        diffusion_cfg = cfg.get("diffusion", {})

        # Initialize DP3 model
        self._dp3_model = DP3(
            shape_meta=shape_meta,
            noise_scheduler=scheduler,
            horizon=self.pred_horizon,
            n_action_steps=self.action_horizon,
            n_obs_steps=self.obs_horizon,
            num_inference_steps=scheduler_cfg.get("num_inference_steps", 10),
            obs_as_global_cond=True,
            diffusion_step_embed_dim=diffusion_cfg.get("diffusion_step_embed_dim", 256),
            down_dims=diffusion_cfg.get("down_dims", [512, 1024, 2048]),
            kernel_size=diffusion_cfg.get("kernel_size", 5),
            n_groups=diffusion_cfg.get("n_groups", 8),
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=encoder_cfg.get("output_dim", 64),
            use_pc_color=encoder_cfg.get("use_pc_color", False),
            pointnet_type=encoder_cfg.get("pointnet_type", "pointnet"),
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        )

        self._dp3_model.to(self._device)
        self._dp3_model.eval()

        # Observation history buffer for temporal conditioning (using deque for efficiency)
        self._obs_buffer = deque(maxlen=self.obs_horizon)

        logger.info(f"DP3Policy initialized on device: {self._device}")

    def load_checkpoint(self, path: str) -> None:
        """
        Loads model weights from checkpoint.

        Args:
            path: Path to checkpoint file (.pth or .ckpt)
        """
        logger.info(f"Loading DP3 checkpoint from: {path}")

        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle PyTorch Lightning prefix 'model.'
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                cleaned_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                cleaned_state_dict[k] = v

        # Load model state
        missing_keys, unexpected_keys = self._dp3_model.load_state_dict(cleaned_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        # Load normalizer if present
        strict_normalizer = self.cfg.get('strict_normalizer', False)
        if 'normalizer' in checkpoint:
            self._dp3_model.normalizer.load_state_dict(checkpoint['normalizer'])
            logger.info("Loaded normalizer from checkpoint")
        elif any('normalizer' in k for k in cleaned_state_dict.keys()):
            logger.info("Normalizer loaded from state_dict")
        else:
            if strict_normalizer:
                raise ValueError(
                    "Checkpoint missing normalizer and strict_normalizer=True. "
                    "Set strict_normalizer=False in config to use identity normalization."
                )
            logger.warning("No normalizer found in checkpoint - using identity normalization")
            # Initialize identity normalization params for all observation and action keys
            self._initialize_identity_normalizer()

        self._dp3_model.eval()
        logger.info("DP3 checkpoint loaded successfully")

    def _initialize_identity_normalizer(self) -> None:
        """Initialize identity (no-op) normalization parameters for all keys."""
        import torch.nn as nn

        # Get observation and action shapes from the DP3 model
        obs_keys = ['point_cloud', 'agent_pos']  # CALVIN observation keys
        action_key = 'action'

        # Create identity params for each key
        for key in obs_keys:
            if key == 'point_cloud':
                dim = self.num_points * 3  # (N, 3) flattened
            elif key == 'agent_pos':
                dim = 15  # CALVIN robot state dimension
            else:
                continue

            # Identity normalization: scale=1, offset=0
            self._dp3_model.normalizer.params_dict[key] = nn.ParameterDict({
                'scale': nn.Parameter(torch.ones(dim, dtype=torch.float32, device=self._device), requires_grad=False),
                'offset': nn.Parameter(torch.zeros(dim, dtype=torch.float32, device=self._device), requires_grad=False),
                'input_stats': nn.ParameterDict({
                    'min': nn.Parameter(torch.zeros(dim, dtype=torch.float32, device=self._device), requires_grad=False),
                    'max': nn.Parameter(torch.ones(dim, dtype=torch.float32, device=self._device), requires_grad=False),
                    'mean': nn.Parameter(torch.zeros(dim, dtype=torch.float32, device=self._device), requires_grad=False),
                    'std': nn.Parameter(torch.ones(dim, dtype=torch.float32, device=self._device), requires_grad=False),
                })
            })

        # Action normalization
        action_dim = 7  # CALVIN action dimension
        self._dp3_model.normalizer.params_dict[action_key] = nn.ParameterDict({
            'scale': nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=self._device), requires_grad=False),
            'offset': nn.Parameter(torch.zeros(action_dim, dtype=torch.float32, device=self._device), requires_grad=False),
            'input_stats': nn.ParameterDict({
                'min': nn.Parameter(torch.zeros(action_dim, dtype=torch.float32, device=self._device), requires_grad=False),
                'max': nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=self._device), requires_grad=False),
                'mean': nn.Parameter(torch.zeros(action_dim, dtype=torch.float32, device=self._device), requires_grad=False),
                'std': nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=self._device), requires_grad=False),
            })
        })

        logger.info("Initialized identity normalizer with no-op transformation")

    def reset(self) -> None:
        """Clears internal temporal buffers (observation history)."""
        self._obs_buffer.clear()

    def _obs_to_dp3_dict(self, obs: Observation) -> Dict[str, torch.Tensor]:
        """
        Convert LangSteer Observation DTO to DP3 internal format.

        Args:
            obs: Observation DTO with point cloud and proprioception

        Returns:
            Dictionary with 'point_cloud' and 'agent_pos' tensors
        """
        # Extract point cloud: (N, 3) -> (B=1, T=1, N, 3)
        pcd = torch.from_numpy(obs.pcd).float().to(self._device)
        if len(pcd.shape) == 2:
            pcd = pcd.unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions

        # Extract agent state: (15,) -> (B=1, T=1, 15)
        agent_pos = torch.from_numpy(obs.proprio).float().to(self._device)
        if len(agent_pos.shape) == 1:
            agent_pos = agent_pos.unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions

        # Add to observation buffer (deque automatically limits size)
        obs_dict = {
            'point_cloud': pcd,
            'agent_pos': agent_pos,
        }
        self._obs_buffer.append(obs_dict)

        # Stack observations along time dimension
        # If we don't have enough history, zero-pad at the beginning
        obs_list = list(self._obs_buffer)
        while len(obs_list) < self.obs_horizon:
            zero_obs = {
                'point_cloud': torch.zeros_like(pcd),
                'agent_pos': torch.zeros_like(agent_pos)
            }
            obs_list.insert(0, zero_obs)

        stacked_obs = {
            'point_cloud': torch.cat([o['point_cloud'] for o in obs_list], dim=1),  # (B, T_obs, N, 3)
            'agent_pos': torch.cat([o['agent_pos'] for o in obs_list], dim=1),  # (B, T_obs, 15)
        }

        return stacked_obs

    def _dp3_to_action(self, action_dict: Dict[str, torch.Tensor]) -> Action:
        """
        Convert DP3 output to LangSteer Action DTO.

        Args:
            action_dict: Dictionary with 'action' key containing (B, H, 7) tensor

        Returns:
            Action DTO with trajectory and gripper state
        """
        # Extract action tensor: (B, H, 7) -> (H, 7)
        action_tensor = action_dict['action'].squeeze(0).cpu().numpy()  # Remove batch dimension

        # Split into trajectory (position + orientation) and gripper
        # Action format: [x, y, z, euler_x, euler_y, euler_z, gripper]
        trajectory = action_tensor  # Keep all 7 dimensions
        gripper = float(action_tensor[0, 6])  # Gripper state from first action

        return Action(
            trajectory=trajectory,
            gripper=gripper
        )

    def forward(self, obs: Observation, steering: Optional[BaseSteering] = None) -> Action:
        """
        Predicts an action given an observation, optionally guided by a steering module.

        Args:
            obs: Observation DTO containing point cloud, proprioception, etc.
            steering: Optional steering module for guidance

        Returns:
            Action DTO with predicted trajectory and gripper state
        """
        with torch.no_grad():
            # Convert observation to DP3 format
            obs_dict = self._obs_to_dp3_dict(obs)

            # Create guidance function if steering is provided
            guidance_fn = None
            if steering is not None:
                def guidance_fn(trajectory, timestep, obs_embedding):
                    """Wrapper to call steering.get_guidance()."""
                    return steering.get_guidance(trajectory, timestep, obs_embedding)

            # Run DP3 inference with optional guidance
            action_dict = self._dp3_model.predict_action(obs_dict, guidance_fn=guidance_fn)

            # Convert to Action DTO
            action = self._dp3_to_action(action_dict)

        return action
