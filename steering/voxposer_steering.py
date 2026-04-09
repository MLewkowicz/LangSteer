"""VoxPoser value-map-based steering for diffusion policies.

Uses LLM-synthesized 3D value maps to guide the diffusion denoising
process via spatial gradients. At each guided denoising step, predicts the
clean trajectory x_0 via Tweedie's formula, denormalizes predicted positions
to world space, queries the precomputed gradient field of the value map, then
applies the gradient as guidance in either epsilon space or DPS (post-step)
style.

Two guidance modes are supported (toggled via guidance_mode config):
  - 'epsilon': Convert world-space gradient → epsilon space via Jacobian and
    add to ε before the scheduler step. No denoising loop changes required.
  - 'dps': Return gradient correction in model trajectory space; the
    denoising loop applies it to x_{t-1} after the scheduler step.

Steering is skipped entirely for timesteps above start_guidance_timestep,
preventing guidance from acting on pure-noise early steps where Tweedie
x_0 predictions are meaningless.
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
    Tweedie's formula predicts x_0 from the current noisy trajectory, positions
    are denormalized to world space, the precomputed gradient field is queried,
    and the result is converted to either epsilon-space or DPS-style guidance.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.guidance_strength = cfg.get('guidance_strength', 1.0)
        self.horizon = cfg.get('horizon', 16)
        # Must be 'epsilon' for DiffuserActor (which uses epsilon prediction)
        self.prediction_type = cfg.get('prediction_type', 'epsilon')
        self.device = cfg.get('device', 'cuda')
        self.map_size = cfg.get('map_size', 100)

        # Guidance mode: 'epsilon' (modify ε before scheduler) or
        # 'dps' (correct x_{t-1} after scheduler step)
        self.guidance_mode = cfg.get('guidance_mode', 'epsilon')

        # Only apply guidance for timesteps <= this value (out of max_timesteps).
        # Steps above the threshold are pure noise — Tweedie predictions are
        # unreliable there.
        self.start_guidance_timestep = cfg.get('start_guidance_timestep', 50)

        # Timestep scaling (soft ramp within the guided window)
        self.use_timestep_scaling = cfg.get('use_timestep_scaling', True)
        self.min_timestep_scale = cfg.get('min_timestep_scale', 0.1)

        # Workspace bounds (world coords, meters) — for value-map lookup
        self._workspace_min = np.array(
            cfg.get('workspace_bounds_min', [-0.35, -0.40, 0.40]),
            dtype=np.float32,
        )
        self._workspace_max = np.array(
            cfg.get('workspace_bounds_max', [0.35, 0.15, 0.85]),
            dtype=np.float32,
        )

        # DiffuserActor position normalization bounds (gripper_loc_bounds).
        # These define the normalization from gripper-relative world coords
        # to model-internal [-1, 1] space.
        glb = cfg.get('gripper_loc_bounds', None)
        self._gripper_loc_bounds: Optional[torch.Tensor] = (
            torch.tensor(glb, dtype=torch.float32) if glb is not None else None
        )
        self._is_relative = cfg.get('relative', True)

        # Current absolute gripper position (set per-step by policy wrapper)
        self._current_gripper_pos: Optional[torch.Tensor] = None

        # Visualization
        self._visualize = cfg.get('visualize', False)
        self._visualizer: Optional[ValueMapVisualizer] = None

        # Lazy-init LMP system
        self._lmp_config = cfg
        self._lmps = None
        self._lmp_interface = None

        # Per-episode state (current active stage)
        self._value_map: Optional[ValueMap] = None
        self._gradient_field: Optional[torch.Tensor] = None  # (M,M,M,3)

        # Multi-stage state
        self._stages: list = []  # [(aff, avoid), ...] raw from composer
        self._current_stage_idx: int = 0
        self._current_stage_target: Optional[np.ndarray] = None  # (3,) world pos
        self._stage_proximity_threshold: float = cfg.get(
            'stage_proximity_threshold', 0.05
        )

        # Schedulers — set by the policy wrapper before inference
        self.position_scheduler = None
        self.rotation_scheduler = None

        self.current_episode_step = 0
        self._robot_obs: Optional[np.ndarray] = None  # cached for visualization

        logger.info(
            f"VoxPoserSteering: mode={self.guidance_mode}, "
            f"strength={self.guidance_strength}, "
            f"start_t={self.start_guidance_timestep}, "
            f"map_size={self.map_size}, prediction_type={self.prediction_type}"
        )

    # ------------------------------------------------------------------
    # Scheduler setters
    # ------------------------------------------------------------------

    def set_position_scheduler(self, scheduler):
        """Store reference to position noise scheduler."""
        self.position_scheduler = scheduler

    def set_rotation_scheduler(self, scheduler):
        """Store reference to rotation noise scheduler."""
        self.rotation_scheduler = scheduler

    def set_current_gripper_pos(self, gripper_pos: np.ndarray):
        """Set current absolute gripper position for relative coordinate conversion.

        Called by policy wrapper before each forward pass.

        Args:
            gripper_pos: (3,) absolute gripper XYZ position
        """
        self._current_gripper_pos = torch.tensor(
            gripper_pos, dtype=torch.float32, device=self.device
        )

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------

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

        The composer returns either:
          - A list of stage tuples: [(aff, avoid), ...]
          - A single tuple: (aff, avoid)

        Each stage gets its own ValueMap + gradient field. Stage transitions
        happen at runtime via check_stage_transition().

        Args:
            task_name: CALVIN task name
            instruction: Natural language instruction (defaults to task_name)
            robot_obs: (15,) robot state for object detection
            scene_obs: (24,) scene state for object detection

        Returns:
            (None, None) for compatibility with TweedieSteering interface
        """
        self.current_episode_step = 0
        self._stages = []
        self._current_stage_idx = 0
        self._current_stage_target = None
        self._robot_obs = robot_obs
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

        # Parse composer result into list of stage tuples
        if isinstance(result, list):
            self._stages = result
        elif isinstance(result, tuple) and len(result) == 2:
            # Single-stage format: wrap as 1-element list
            self._stages = [result]
        else:
            logger.warning(f"Unexpected composer result type: {type(result)}")
            self._value_map = None
            return None, None

        logger.info(f"Composer returned {len(self._stages)} stage(s)")

        # Activate the first stage
        self._activate_stage(0)

        return None, None

    def _activate_stage(self, idx: int):
        """Build ValueMap and gradient field for stage `idx`.

        Evaluates the lazy map functions, creates a ValueMap, smooths it,
        precomputes gradients, and computes the stage's target position
        (centroid of raw affordance voxels in world space).
        """
        if idx >= len(self._stages):
            logger.warning(f"Stage {idx} out of range (have {len(self._stages)})")
            return

        stage = self._stages[idx]
        if not (isinstance(stage, (tuple, list)) and len(stage) == 2):
            logger.warning(f"Stage {idx}: expected 2-tuple, got {type(stage)}")
            self._value_map = None
            self._gradient_field = None
            return

        aff_fn, avoid_fn = stage

        affordance = self._eval_map(aff_fn)
        avoidance = self._eval_map(avoid_fn)

        if affordance is None:
            logger.warning(f"Stage {idx}: no affordance map, steering disabled")
            self._value_map = None
            self._gradient_field = None
            self._current_stage_target = None
            self._current_stage_idx = idx
            return

        self._value_map = ValueMap(
            affordance=affordance,
            avoidance=avoidance,
            workspace_bounds_min=self._workspace_min,
            workspace_bounds_max=self._workspace_max,
            map_size=self.map_size,
            instruction=f"stage_{idx}",
        )
        self._value_map.smooth()
        self._value_map.precompute_gradients()

        # Precompute gradient field as torch tensor
        self._gradient_field = torch.from_numpy(
            np.stack([
                self._value_map._grad_x,
                self._value_map._grad_y,
                self._value_map._grad_z,
            ], axis=-1)
        ).float().to(self.device)

        # Compute stage target: centroid of raw affordance voxels → world coords
        raw_aff = self._value_map._raw_affordance
        if raw_aff is not None and raw_aff.max() > 0:
            target_voxels = np.argwhere(raw_aff > 0)
            centroid_voxel = target_voxels.mean(axis=0).astype(int)
            self._current_stage_target = voxel2pc(
                centroid_voxel[np.newaxis],
                self._workspace_min, self._workspace_max, self.map_size,
            )[0]
        else:
            self._current_stage_target = None

        self._current_stage_idx = idx

        logger.info(
            f"Activated stage {idx}/{len(self._stages) - 1}: "
            f"affordance max={affordance.max():.2f}, "
            f"non-zero={np.count_nonzero(affordance)}, "
            f"target={self._current_stage_target}"
        )

        # Visualize if enabled
        if self._visualizer is not None:
            ee_pos = (
                self._robot_obs[:3] if self._robot_obs is not None else None
            )
            detections = self._lmp_interface.get_all_detections()
            self._visualizer.visualize(
                self._value_map, ee_pos_world=ee_pos, objects=detections
            )

    def increment_step(self):
        """Advance episode step counter."""
        self.current_episode_step += 1

    def check_stage_transition(self, ee_pos: np.ndarray) -> bool:
        """Check if EE is close enough to current target to advance stage.

        Called at each environment step from the step callback.

        Args:
            ee_pos: (3,) absolute world-frame end-effector position

        Returns:
            True if stage was advanced
        """
        if self._current_stage_target is None:
            return False
        if self._current_stage_idx >= len(self._stages) - 1:
            return False  # already on last stage

        dist = np.linalg.norm(ee_pos - self._current_stage_target)
        if dist < self._stage_proximity_threshold:
            next_idx = self._current_stage_idx + 1
            logger.info(
                f"Stage transition: {self._current_stage_idx} → {next_idx} "
                f"(dist={dist:.3f}m < threshold={self._stage_proximity_threshold}m)"
            )
            self._activate_stage(next_idx)
            return True
        return False

    # ------------------------------------------------------------------
    # Core guidance computation
    # ------------------------------------------------------------------

    def get_guidance(self, current_sample: torch.Tensor, timestep: int,
                     obs_embedding: Any, model_output: torch.Tensor) -> torch.Tensor:
        """Compute value-map gradient guidance.

        Pipeline:
          1. Skip if timestep > start_guidance_timestep (high-noise regime).
          2. Apply Tweedie's formula to predict x_0 in model-internal space.
          3. Denormalize position dims to world space.
          4. Look up value-map gradient at world positions.
          5. Convert gradient to model (x_0) space via chain rule.
          6. Convert to the output space required by guidance_mode:
               'epsilon' → δε = sqrt(ᾱ)/sqrt(1-ᾱ) · grad_model
               'dps'     → δx = (1/sqrt(ᾱ)) · grad_model  (applied to x_{t-1})

        Args:
            current_sample: Noisy trajectory x_t, shape (B, L, D)
            timestep: Current diffusion timestep
            obs_embedding: Observation features (unused; fixed_inputs for DA)
            model_output: Model's epsilon prediction ε, shape (B, L, D)

        Returns:
            Guidance tensor, shape matching model_output.
            In 'epsilon' mode: added to ε before scheduler step.
            In 'dps' mode: added to x_{t-1} after scheduler step.
        """
        if self._value_map is None or self._gradient_field is None:
            return torch.zeros_like(model_output)

        # Hard threshold — skip noisy timesteps
        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        if t > self.start_guidance_timestep:
            return torch.zeros_like(model_output)

        B, L, D = model_output.shape
        H = min(self.horizon, L)

        # Predict x_0 in model-internal space via Tweedie
        x_0_pred = self._predict_x0(current_sample, timestep, model_output)

        # Denormalize position dims to world space: (B, H, 3)
        model_pos = x_0_pred[:, :H, :3]
        world_pos = self._model_to_world(model_pos)

        # Value-map gradient at world positions: (B, H, 3) — world space
        grad_world = self._lookup_gradient(world_pos)

        # Convert world-space gradient → model (x_0) space via chain rule.
        # normalize_pos: model = (world_rel - pos_min) / (pos_max - pos_min) * 2 - 1
        # Jacobian: dmodel/dworld = 2 / (pos_max - pos_min)
        # Inverse: dworld/dmodel = (pos_max - pos_min) / 2
        # For gradient: grad_model = grad_world * (pos_max - pos_min) / 2
        if self._gripper_loc_bounds is not None:
            bounds = self._gripper_loc_bounds.to(grad_world.device)
            scale_factor = (bounds[1] - bounds[0]) / 2.0  # (3,)
            grad_model = grad_world * scale_factor.unsqueeze(0).unsqueeze(0)
        else:
            grad_model = grad_world

        scale = self._compute_timestep_scale(timestep)
        alpha_bar = self._get_alpha_bar(timestep)  # scalar tensor

        if self.guidance_mode == 'epsilon':
            # To shift x_0 by δx_0, adjust ε by:
            # δε = sqrt(ᾱ)/sqrt(1-ᾱ) · δx_0_model
            # grad_model points toward increasing affordance (positive direction)
            coeff = torch.sqrt(alpha_bar) / torch.sqrt(
                torch.clamp(1.0 - alpha_bar, min=1e-6)
            )
            delta = self.guidance_strength * scale * coeff * grad_model
        else:
            # DPS: chain rule through Tweedie.
            # ∂x_0_hat/∂x_t = 1/sqrt(ᾱ), so the correction to x_{t-1} is:
            # δx_{t-1} = (1/sqrt(ᾱ)) · guidance_strength · scale · grad_model
            coeff = 1.0 / torch.sqrt(torch.clamp(alpha_bar, min=1e-6))
            delta = self.guidance_strength * scale * coeff * grad_model

        guidance = torch.zeros_like(model_output)
        guidance[:, :H, :3] = delta

        if self.current_episode_step % 10 == 0:
            logger.debug(
                f"[VoxPoser/{self.guidance_mode}] step={self.current_episode_step}, "
                f"t={t}: guidance_norm={torch.norm(guidance).item():.4f}, "
                f"coeff={coeff.item():.4f}, scale={scale:.4f}, "
                f"alpha_bar={alpha_bar.item():.4f}"
            )

        return guidance.detach()

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------

    def _model_to_world(self, model_pos: torch.Tensor) -> torch.Tensor:
        """Convert model-internal normalized position → absolute world position.

        Reverses DiffuserActor's two-stage normalization:
          1. normalize_pos: model = (world_rel - pos_min) / (pos_max - pos_min) * 2 - 1
          2. convert2rel:   world_rel = world_abs - gripper_pos

        Args:
            model_pos: (B, H, 3) positions in model [-1, 1] space

        Returns:
            (B, H, 3) absolute world positions (meters)
        """
        if self._gripper_loc_bounds is not None:
            bounds = self._gripper_loc_bounds.to(model_pos.device)
            pos_min = bounds[0]
            pos_max = bounds[1]
            # Undo normalize_pos
            world_rel = (model_pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min
        else:
            world_rel = model_pos

        # Undo gripper-relative conversion
        if self._is_relative and self._current_gripper_pos is not None:
            gripper = self._current_gripper_pos.to(model_pos.device)
            world_abs = world_rel + gripper.view(1, 1, 3)
        else:
            world_abs = world_rel

        return world_abs

    def _lookup_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """Look up precomputed value-map gradient at world positions.

        Args:
            positions: (B, H, 3) absolute world-frame XYZ positions (meters)

        Returns:
            (B, H, 3) gradient vectors pointing toward increasing affordance
        """
        B, H, _ = positions.shape
        M = self.map_size

        ws_min = torch.tensor(self._workspace_min, device=positions.device, dtype=positions.dtype)
        ws_max = torch.tensor(self._workspace_max, device=positions.device, dtype=positions.dtype)

        # Normalize to [0, M-1] voxel indices
        clamped = torch.clamp(positions, ws_min, ws_max)
        voxel_float = (clamped - ws_min) / (ws_max - ws_min) * (M - 1)
        voxel_idx = torch.clamp(voxel_float.long(), 0, M - 1)

        flat_idx = voxel_idx.reshape(-1, 3)
        ix = flat_idx[:, 0]
        iy = flat_idx[:, 1]
        iz = flat_idx[:, 2]

        # Gradient field: (M, M, M, 3) — gradients of the value map in voxel space
        grad_flat = self._gradient_field[ix, iy, iz]  # (B*H, 3)

        # Scale from voxel-space gradient → world-space gradient (1/m)
        resolution = torch.tensor(
            (self._workspace_max - self._workspace_min) / M,
            device=positions.device, dtype=positions.dtype,
        )
        grad_world = grad_flat / resolution.unsqueeze(0)

        return grad_world.reshape(B, H, 3)

    # ------------------------------------------------------------------
    # Diffusion helpers
    # ------------------------------------------------------------------

    def _get_alpha_bar(self, timestep) -> torch.Tensor:
        """Get cumulative noise schedule value ᾱ_t for the position scheduler."""
        sched = self.position_scheduler
        if sched is None:
            return torch.tensor(0.5, device=self.device)

        if isinstance(timestep, torch.Tensor):
            t_idx = timestep.long()
        else:
            t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)

        alpha_bar = sched.alphas_cumprod[t_idx]
        return torch.clamp(alpha_bar, min=1e-6, max=1.0 - 1e-6)

    def _predict_x0(self, x_t: torch.Tensor, timestep,
                    model_output: torch.Tensor) -> torch.Tensor:
        """Apply Tweedie's formula to predict clean sample x_0 from x_t and ε.

        For epsilon prediction (DiffuserActor):
            x_0 = (x_t - sqrt(1-ᾱ) · ε) / sqrt(ᾱ)

        Uses position_scheduler for alpha_bar if available.
        """
        if self.prediction_type == 'sample':
            return model_output

        sched = self.position_scheduler
        if sched is None:
            logger.warning("No scheduler set, returning model_output as x_0")
            return model_output

        if isinstance(timestep, torch.Tensor):
            t_idx = timestep.long()
        else:
            t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)

        alpha_bar = torch.clamp(sched.alphas_cumprod[t_idx], min=1e-6)
        alpha_bar = alpha_bar.view(-1, 1, 1)

        # model_output may have more dims than x_t (e.g. openness appended)
        D = x_t.shape[-1]
        eps = model_output[..., :D]

        if self.prediction_type == 'epsilon':
            x_0 = (x_t - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        elif self.prediction_type == 'v_prediction':
            x_0 = torch.sqrt(alpha_bar) * x_t - torch.sqrt(1 - alpha_bar) * eps
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return x_0

    def _compute_timestep_scale(self, timestep) -> float:
        """Compute soft guidance scale based on diffusion timestep.

        Ramps from min_timestep_scale at t=0 to 1.0 at t=max_timesteps.
        Only applied within the guided window (t <= start_guidance_timestep).
        """
        if not self.use_timestep_scaling:
            return 1.0

        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep

        if self.position_scheduler is None:
            return 1.0

        max_timestep = self.position_scheduler.config.num_train_timesteps
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
            if hasattr(result, 'array'):
                return result.array
            return np.asarray(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate map: {e}")
            return None
