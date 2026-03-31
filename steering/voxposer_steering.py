"""VoxPoser value-map-based steering for diffusion policies.

Uses LLM-synthesized 3D value maps to guide the diffusion denoising
process via spatial gradients. At each denoising step, predicts the
clean trajectory x_0 via Tweedie's formula, queries the precomputed
gradient field of the value map at each waypoint, and produces a
guidance signal that pushes the trajectory toward high-affordance regions.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch

from core.steering import BaseSteering
from voxposer.calvin_interface import voxel2pc
from voxposer.lmp import setup_lmp, set_lmp_objects
from voxposer.value_map import ValueMap
from voxposer.visualizer import ValueMapVisualizer

logger = logging.getLogger(__name__)


class VoxPoserSteering(BaseSteering):
    """Steer diffusion policies using LLM-generated spatial value maps.

    At episode start, the LLM composer decomposes the task instruction into
    affordance/avoidance/gripper maps over a 3D voxel grid. During denoising,
    the precomputed gradient of the cost map guides predicted waypoints toward
    target regions and away from obstacles.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.guidance_strength = cfg.get('guidance_strength', 1.0)
        self.horizon = cfg.get('horizon', 16)
        self.prediction_type = cfg.get('prediction_type', 'sample')
        self.device = cfg.get('device', 'cuda')
        self.map_size = cfg.get('map_size', 100)

        # Timestep scaling
        self.use_timestep_scaling = cfg.get('use_timestep_scaling', True)
        self.min_timestep_scale = cfg.get('min_timestep_scale', 0.1)

        # Workspace bounds
        self._workspace_min = np.array(
            cfg.get('workspace_bounds_min', [-0.35, -0.40, 0.40]),
            dtype=np.float32,
        )
        self._workspace_max = np.array(
            cfg.get('workspace_bounds_max', [0.35, 0.15, 0.85]),
            dtype=np.float32,
        )

        # Visualization
        self._visualize = cfg.get('visualize', False)
        self._visualizer: Optional[ValueMapVisualizer] = None

        # Lazy-init LMP system
        self._lmp_config = cfg
        self._lmps = None
        self._lmp_interface = None

        # Per-episode state
        self._value_map: Optional[ValueMap] = None
        self._gradient_field: Optional[torch.Tensor] = None  # (M,M,M,3)
        self.scheduler = None
        self.current_episode_step = 0

        logger.info(
            f"VoxPoserSteering: strength={self.guidance_strength}, "
            f"map_size={self.map_size}, prediction_type={self.prediction_type}"
        )

    def set_scheduler(self, scheduler):
        """Store reference to noise scheduler (for timestep scaling)."""
        self.scheduler = scheduler

    def _init_lmp_system(self):
        """Lazily initialize the LMP hierarchy."""
        if self._lmps is not None:
            return
        self._lmps, self._lmp_interface = setup_lmp(self._lmp_config)
        if self._visualize:
            self._visualizer = ValueMapVisualizer(self._lmp_config)
        logger.info("Initialized VoxPoser LMP system")

    def setup_episode(self, task_name: str, instruction: str = None,
                      robot_obs: np.ndarray = None,
                      scene_obs: np.ndarray = None):
        """Generate value maps for a new episode via LLM composer.

        Args:
            task_name: CALVIN task name
            instruction: Natural language instruction (defaults to task_name)
            robot_obs: (15,) robot state for object detection
            scene_obs: (24,) scene state for object detection

        Returns:
            (None, None) for compatibility with TweedieSteering interface
        """
        self.current_episode_step = 0
        self._init_lmp_system()

        if instruction is None:
            instruction = task_name.replace('_', ' ')

        # Update scene state
        if robot_obs is not None and scene_obs is not None:
            self._lmp_interface.update_state(robot_obs, scene_obs)

        # Set object context for LMPs
        object_names = self._lmp_interface.get_object_names()
        set_lmp_objects(self._lmps, object_names)

        # Run composer to generate value maps
        logger.info(f"Running VoxPoser composer for: '{instruction}'")
        try:
            result = self._lmps['composer'](instruction)
        except Exception as e:
            logger.error(f"Composer failed for '{instruction}': {e}")
            self._value_map = None
            return None, None

        # Parse composer result: (affordance_map, avoidance_map, gripper_map)
        if isinstance(result, tuple) and len(result) == 3:
            aff_fn, avoid_fn, grip_fn = result
        else:
            logger.warning(f"Unexpected composer result type: {type(result)}")
            self._value_map = None
            return None, None

        # Evaluate lazy functions (non-composer LMPs return callables)
        affordance = self._eval_map(aff_fn)
        avoidance = self._eval_map(avoid_fn)
        gripper = self._eval_map(grip_fn)

        if affordance is None:
            logger.warning("No affordance map generated, steering disabled")
            self._value_map = None
            return None, None

        # Build ValueMap
        self._value_map = ValueMap(
            affordance=affordance,
            avoidance=avoidance,
            gripper=gripper,
            workspace_bounds_min=self._workspace_min,
            workspace_bounds_max=self._workspace_max,
            map_size=self.map_size,
            instruction=instruction,
        )
        self._value_map.smooth()
        self._value_map.precompute_gradients()

        # Precompute gradient field as torch tensor for fast per-step lookups
        grad_x, grad_y, grad_z = (
            self._value_map._grad_x,
            self._value_map._grad_y,
            self._value_map._grad_z,
        )
        self._gradient_field = torch.from_numpy(
            np.stack([grad_x, grad_y, grad_z], axis=-1)
        ).float().to(self.device)

        logger.info(
            f"ValueMap ready: affordance max={affordance.max():.2f}, "
            f"non-zero voxels={np.count_nonzero(affordance)}"
        )

        # Visualize if enabled
        if self._visualizer is not None:
            ee_pos = robot_obs[:3] if robot_obs is not None else None
            detections = self._lmp_interface.get_all_detections()
            self._visualizer.visualize(
                self._value_map, ee_pos_world=ee_pos, objects=detections
            )

        return None, None

    def increment_step(self):
        """Advance episode step counter."""
        self.current_episode_step += 1

    def get_guidance(self, current_sample: torch.Tensor, timestep: int,
                     obs_embedding: Any, model_output: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient guidance from value map.

        Args:
            current_sample: Noisy trajectory x_t, shape (B, H, D)
            timestep: Current diffusion timestep
            obs_embedding: Observation features (unused)
            model_output: Model's prediction (epsilon or x_0)

        Returns:
            Guidance gradient, shape matching model_output
        """
        if self._value_map is None or self._gradient_field is None:
            return torch.zeros_like(model_output)

        # Predict clean trajectory x_0
        if self.prediction_type == 'sample':
            x_0_pred = model_output
        else:
            x_0_pred = self._predict_x0(current_sample, timestep, model_output)

        # Extract position components: (B, H, 3)
        B, L, D = x_0_pred.shape
        H = min(self.horizon, L)
        pred_positions = x_0_pred[:, :H, :3]  # world XYZ

        # Convert positions to voxel coordinates for gradient lookup
        scale = self._compute_timestep_scale(timestep)

        # Compute guidance from spatial gradients
        guidance = torch.zeros_like(model_output)
        pos_guidance = self._lookup_gradient(pred_positions)

        # Negate gradient (we want to move toward LOWER cost = higher affordance)
        # Scale by guidance strength and timestep
        pos_guidance = -self.guidance_strength * scale * pos_guidance

        guidance[:, :H, :3] = pos_guidance

        if self.current_episode_step % 10 == 0:
            logger.debug(
                f"[VoxPoser] Step {self.current_episode_step}, t={timestep}: "
                f"guidance_norm={torch.norm(guidance).item():.4f}, "
                f"scale={scale:.4f}"
            )

        return guidance.detach()

    def _lookup_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """Look up precomputed cost gradient at trajectory positions.

        Args:
            positions: (B, H, 3) world-frame XYZ positions

        Returns:
            (B, H, 3) gradient vectors (direction of increasing cost)
        """
        B, H, _ = positions.shape
        M = self.map_size

        # Convert world positions to voxel indices
        ws_min = torch.tensor(self._workspace_min, device=positions.device, dtype=positions.dtype)
        ws_max = torch.tensor(self._workspace_max, device=positions.device, dtype=positions.dtype)

        # Normalize to [0, M-1]
        clamped = torch.clamp(positions, ws_min, ws_max)
        voxel_float = (clamped - ws_min) / (ws_max - ws_min) * (M - 1)
        voxel_idx = torch.clamp(voxel_float.long(), 0, M - 1)

        # Flatten batch for indexing
        flat_idx = voxel_idx.reshape(-1, 3)  # (B*H, 3)
        ix = flat_idx[:, 0]
        iy = flat_idx[:, 1]
        iz = flat_idx[:, 2]

        # Index into precomputed gradient field: (M, M, M, 3)
        grad_flat = self._gradient_field[ix, iy, iz]  # (B*H, 3)

        # Scale gradient from voxel space to world space
        # Gradient is per-voxel, so multiply by voxels-per-meter to get world-space gradient
        resolution = torch.tensor(
            (self._workspace_max - self._workspace_min) / M,
            device=positions.device, dtype=positions.dtype,
        )
        grad_world = grad_flat / resolution.unsqueeze(0)

        return grad_world.reshape(B, H, 3)

    def _predict_x0(self, x_t: torch.Tensor, timestep: int,
                    model_output: torch.Tensor) -> torch.Tensor:
        """Apply Tweedie's formula to predict clean sample x_0."""
        if self.prediction_type == 'sample':
            return model_output

        if isinstance(timestep, torch.Tensor):
            t_idx = timestep.long()
        else:
            t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)

        if self.scheduler is None:
            logger.warning("No scheduler set, returning model_output as x_0")
            return model_output

        alpha_bar = self.scheduler.alphas_cumprod[t_idx]
        alpha_bar = torch.clamp(alpha_bar, min=1e-6).view(-1, 1, 1)

        if self.prediction_type == 'epsilon':
            x_0 = (x_t - torch.sqrt(1 - alpha_bar) * model_output) / torch.sqrt(alpha_bar)
        elif self.prediction_type == 'v_prediction':
            x_0 = torch.sqrt(alpha_bar) * x_t - torch.sqrt(1 - alpha_bar) * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return x_0

    def _compute_timestep_scale(self, timestep: int) -> float:
        """Compute guidance scale based on diffusion timestep."""
        if not self.use_timestep_scaling:
            return 1.0

        if isinstance(timestep, torch.Tensor):
            t = timestep.item()
        else:
            t = timestep

        if self.scheduler is None:
            return 1.0

        max_timestep = self.scheduler.config.num_train_timesteps
        normalized_t = t / max_timestep
        return self.min_timestep_scale + (1.0 - self.min_timestep_scale) * normalized_t

    @staticmethod
    def _eval_map(map_fn) -> Optional[np.ndarray]:
        """Evaluate a voxel map, handling callables and VoxelIndexingWrapper."""
        if map_fn is None:
            return None
        try:
            if callable(map_fn):
                result = map_fn()
            else:
                result = map_fn
            # Unwrap VoxelIndexingWrapper
            if hasattr(result, 'array'):
                return result.array
            return np.asarray(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate map: {e}")
            return None
