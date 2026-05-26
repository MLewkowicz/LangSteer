"""Shared plumbing for the four DiffuserActor policy variants.

`DiffuserActorBasePolicy` owns everything that does not depend on the
conditioning mode:

  * model build + checkpoint load + gripper history buffer,
  * RGB / PCD / gripper-history tensor prep,
  * model-output → CALVIN action conversion (quaternion → Euler with PyBullet
    gimbal-lock fallback, openness → binary gripper),
  * the top-level `forward()` template with two variant hooks,
  * the steering hookup (`_build_guidance_fns`),
  * `set_primitive` / `set_object` default RuntimeError raisers — subclasses
    that actually support these override them.

Variant subclasses (`Language`, `Nolang`, `Primitive`, `PrimitiveObject`) only
override the two hooks and, where applicable, the `set_*` methods. Class-level
constants (`_use_instruction`, `_use_primitive_id`, `_use_object_id`) flag each
variant's mode for `wire_steering` introspection in the runners.

Checkpoint compatibility is preserved: every variant builds the same
`DiffuserActor` nn.Module, so state-dict keys are unchanged across the split.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

import numpy as np
import torch

from core.policy import BasePolicy
from core.steering import BaseSteering
from core.types import Action, Observation

try:
    from policies.diffuser_actor_components import DiffuserActor
except ImportError as e:  # pragma: no cover — surfaced at construction time
    _IMPORT_ERROR: Optional[BaseException] = e
    DiffuserActor = None  # type: ignore[misc,assignment]
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class DiffuserActorBasePolicy(BasePolicy):
    """Shared plumbing for the four DiffuserActor variants.

    Subclasses implement two hooks:
        _build_instruction(obs)               -> Tensor | None
        _log_conditioning_diag(instr_emb, obs)
    """

    # Class-level conditioning constants. Subclasses override.
    _use_instruction: bool = True
    _use_primitive_id: bool = False
    _use_object_id: bool = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        if _IMPORT_ERROR is not None:
            raise _IMPORT_ERROR

        self._device = torch.device(
            cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model hyperparameters (variant-independent).
        backbone = cfg.get("backbone", "clip")
        image_size = tuple(cfg.get("image_size", [256, 256]))
        embedding_dim = cfg.get("embedding_dim", 120)
        num_vis_ins_attn_layers = cfg.get("num_vis_ins_attn_layers", 2)
        fps_subsampling_factor = cfg.get("fps_subsampling_factor", 5)
        gripper_loc_bounds = cfg.get("gripper_loc_bounds", [[-2, -2, -2], [2, 2, 2]])
        rotation_parametrization = cfg.get("rotation_parametrization", "6D")
        quaternion_format = cfg.get("quaternion_format", "wxyz")
        diffusion_timesteps = cfg.get("diffusion_timesteps", 100)
        nhist = cfg.get("nhist", 1)
        relative = cfg.get("relative", True)
        lang_enhanced = cfg.get("lang_enhanced", True)

        # Conditioning flags — variant subclasses override the class-level
        # constants above, but the model needs them as constructor args too.
        num_primitives = cfg.get("num_primitives", 4)
        num_objects = cfg.get("num_objects", 8)

        self.nhist = nhist
        self.image_size = image_size
        self.camera_names = cfg.get("cameras", ["static", "gripper"])
        self.pred_horizon = cfg.get("pred_horizon", 1)
        self.crop_images = cfg.get("crop_images", True)
        self._relative = relative
        self._quaternion_format = quaternion_format
        self._gripper_loc_bounds_np = np.array(gripper_loc_bounds)
        self._mask_language = cfg.get("mask_language", False)
        self._corrector_steps = cfg.get("corrector_steps", 0)
        self._corrector_step_size = cfg.get("corrector_step_size", 1e-3)

        # Build model — every variant uses the same nn.Module class so the
        # checkpoint key set is identical regardless of which Python subclass
        # is doing the wrapping.
        self._model = DiffuserActor(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            use_instruction=self._use_instruction,
            fps_subsampling_factor=fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=rotation_parametrization,
            quaternion_format=quaternion_format,
            diffusion_timesteps=diffusion_timesteps,
            nhist=nhist,
            relative=relative,
            lang_enhanced=lang_enhanced,
            use_primitive_id=self._use_primitive_id,
            num_primitives=num_primitives,
            use_object_id=self._use_object_id,
            num_objects=num_objects,
        )
        self._model.to(self._device)
        self._model.eval()

        # Gripper history buffer: stores [pos(3), quat_wxyz(4)] per step.
        self._gripper_history: deque = deque(maxlen=nhist)

        # Diagnostic logging counter — log first N forward passes only.
        self._log_count = 0

        logger.info(f"{type(self).__name__} initialized on {self._device}")

    # ------------------------------------------------------------------
    # Checkpoint + episode lifecycle
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from a 3D Diffuser Actor checkpoint."""
        logger.info(f"Loading DiffuserActor checkpoint from: {path}")

        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        # 3DA checkpoint format: {"weight": state_dict, ...}.
        if "weight" in checkpoint:
            state_dict = checkpoint["weight"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip DDP "module." prefix if present.
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned[k[7:]] = v
            else:
                cleaned[k] = v

        missing, unexpected = self._model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.error(
                f"Checkpoint has {len(missing)} MISSING keys (model will produce garbage): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            logger.warning(
                f"Checkpoint has {len(unexpected)} unexpected keys: {unexpected[:5]}"
            )

        self._model.eval()
        logger.info("DiffuserActor checkpoint loaded successfully")

    def reset(self) -> None:
        """Clear gripper history buffer."""
        self._gripper_history.clear()

    # ------------------------------------------------------------------
    # Conditioning setters — default raisers, overridden where supported
    # ------------------------------------------------------------------

    def set_primitive(self, primitive_id: int) -> None:
        """Default raise; subclasses that support primitive conditioning override."""
        raise RuntimeError(
            "set_primitive() called but policy is not primitive-conditioned. "
            "Use policy=diffuser_actor_primitive or policy=diffuser_actor_primitive_object."
        )

    def set_object(self, object_id: int) -> None:
        """Default raise; only PrimitiveObject overrides."""
        raise RuntimeError(
            "set_object() called but policy is not object-conditioned. "
            "Use policy=diffuser_actor_primitive_object."
        )

    # ------------------------------------------------------------------
    # Variant hooks
    # ------------------------------------------------------------------

    def _build_instruction(self, obs: Observation) -> Optional[torch.Tensor]:
        """Build the model's `instruction` arg from the observation.

        Returns:
            * (1, seq_len, 512) float for the language variant,
            * (1, 1) long for primitive,
            * (1, 2) long for primitive+object,
            * None for nolang.
        """
        raise NotImplementedError

    def _log_conditioning_diag(
        self, instr_emb: Optional[torch.Tensor], obs: Observation
    ) -> None:
        """Append the variant-specific diagnostic line to the diag block."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Observation prep helpers
    # ------------------------------------------------------------------

    def _prepare_rgb(self, obs: Observation) -> torch.Tensor:
        """RGB dict → (1, ncam, 3, H, W) tensor in [0, 1]. CALVIN 200→160 crop."""
        rgb_list = []
        for cam_name in self.camera_names:
            if cam_name not in obs.rgb:
                raise ValueError(
                    f"Camera '{cam_name}' not found in observation. "
                    f"Available: {list(obs.rgb.keys())}"
                )
            img = obs.rgb[cam_name]
            if img.dtype == np.uint8:
                t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            else:
                t = torch.from_numpy(img).float().permute(2, 0, 1)
            if self.crop_images and t.shape[1] >= 200 and t.shape[2] >= 200:
                t = t[:, 20:180, 20:180]
            rgb_list.append(t)
        return torch.stack(rgb_list).unsqueeze(0).to(self._device)

    def _prepare_pcd(self, obs: Observation) -> torch.Tensor:
        """Per-pixel PCD images → (1, ncam, 3, H, W) tensor. CALVIN 200→160 crop."""
        if obs.depth is None:
            raise ValueError(
                "DiffuserActorPolicy requires per-pixel PCD images in obs.depth. "
                "Set env.provide_pcd_images=true."
            )
        pcd_list = []
        for cam_name in self.camera_names:
            if cam_name not in obs.depth:
                raise ValueError(
                    f"PCD for camera '{cam_name}' not found. "
                    f"Available: {list(obs.depth.keys())}"
                )
            pcd_img = obs.depth[cam_name]  # (H, W, 3) world XYZ
            t = torch.from_numpy(pcd_img).float().permute(2, 0, 1)
            if self.crop_images and t.shape[1] >= 200 and t.shape[2] >= 200:
                t = t[:, 20:180, 20:180]
            pcd_list.append(t)
        return torch.stack(pcd_list).unsqueeze(0).to(self._device)

    def _prepare_gripper(self, obs: Observation) -> torch.Tensor:
        """Build (1, nhist, 7) gripper history = [pos(3), quat_wxyz(4)].

        Converts CALVIN ee_pose [pos(3), euler_XYZ(3), gripper_width(1)] to a
        wxyz quaternion via the trainer's `convert_rotation` helper.
        """
        from training.policies.diffuser_actor.preprocessing.calvin_utils import (
            convert_rotation,
        )

        ee_pose = obs.ee_pose.copy()
        pos = ee_pose[:3]
        euler = ee_pose[3:6]
        quat = convert_rotation(euler)  # euler → wxyz quaternion (4,)
        gripper_pose = np.concatenate([pos, quat])

        self._gripper_history.append(gripper_pose)

        # Pad history by repeating the oldest observation.
        history = list(self._gripper_history)
        while len(history) < self.nhist:
            history.insert(0, history[0].copy())

        gripper = np.stack(history)
        return torch.from_numpy(gripper).float().unsqueeze(0).to(self._device)

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_quat_to_euler(quat: Any) -> np.ndarray:
        """Convert quaternion (wxyz) → Euler XYZ. PyBullet fallback for gimbal lock."""
        from training.policies.diffuser_actor.preprocessing.pytorch3d_transforms import (
            matrix_to_euler_angles,
            quaternion_to_matrix,
        )

        quat_t = torch.as_tensor(quat)
        mat = quaternion_to_matrix(quat_t)
        euler = matrix_to_euler_angles(mat, "XYZ")
        euler_np = euler.data.cpu().numpy()

        if np.isnan(euler_np).any():
            import pybullet

            flat_shape = list(quat.shape)[:-1] + [3]
            flat_quat = quat.reshape(-1, 4)
            if isinstance(flat_quat, torch.Tensor):
                flat_quat = flat_quat.data.cpu().numpy()
            # PyBullet uses xyzw convention.
            euler_np = np.array(
                [
                    pybullet.getEulerFromQuaternion([q[1], q[2], q[3], q[0]])
                    for q in flat_quat
                ]
            ).reshape(flat_shape)

        return euler_np

    def _convert_action(self, trajectory: torch.Tensor) -> np.ndarray:
        """Model output (B, L, 8) → CALVIN action (L, 7) = [pos, euler, gripper]."""
        traj = trajectory.clone()
        pos = traj[..., :3].cpu().numpy()
        quat = traj[..., 3:7]  # quaternion in model's format
        openness = traj[..., 7:]

        if self._quaternion_format == "xyzw":
            quat = quat[..., [3, 0, 1, 2]]  # xyzw → wxyz

        euler = self._convert_quat_to_euler(quat)
        gripper = (2 * (openness >= 0.5).long() - 1).cpu().numpy()

        return np.concatenate([pos, euler, gripper], axis=-1)

    def _convert_action_relative(
        self, trajectory: torch.Tensor, curr_gripper: torch.Tensor
    ) -> np.ndarray:
        """Relative model output (B, L, 8) -> absolute action (B, L, 7).

        Composes rotations as proper SO(3) elements (quaternion product) rather
        than adding Euler components. Matches the training-side relativisation
        R_rel = R_base.T @ R_target so that R_world = R_base @ R_rel. Position
        is still an additive world-frame delta, which IS correct.
        """
        from training.policies.diffuser_actor.preprocessing.pytorch3d_transforms import (
            quaternion_multiply,
        )

        traj = trajectory.clone()
        pos_rel = traj[..., :3]
        quat_rel = traj[..., 3:7]
        openness = traj[..., 7:]

        if self._quaternion_format == "xyzw":
            quat_rel = quat_rel[..., [3, 0, 1, 2]]  # xyzw -> wxyz

        base = curr_gripper[:, [-1], :]              # (B, 1, 7) = [pos, quat_wxyz]
        base_pos = base[..., :3]
        base_quat = base[..., 3:7]

        pos_world = (pos_rel + base_pos).cpu().numpy()
        quat_world = quaternion_multiply(base_quat, quat_rel)  # body-frame compose
        euler_world = self._convert_quat_to_euler(quat_world)
        gripper = (2 * (openness >= 0.5).long() - 1).cpu().numpy()

        return np.concatenate([pos_world, euler_world, gripper], axis=-1)

    # ------------------------------------------------------------------
    # Steering hookup
    # ------------------------------------------------------------------

    def _build_guidance_fns(
        self, steering: Optional[BaseSteering], obs: Observation
    ) -> tuple[Optional[Any], Optional[Any]]:
        """Wire the steering module into the model's denoising loop.

        Returns a pair `(guidance_fn, dps_guidance_fn)`. Exactly one is non-None
        depending on `steering.guidance_mode` (default 'epsilon'). Both are
        None when no steering is provided.
        """
        if steering is None:
            return None, None

        if hasattr(steering, "set_current_gripper_pos"):
            steering.set_current_gripper_pos(obs.ee_pose[:3])
        # Forward-compat plumbing for rotation-aware steering; ee_pose encodes
        # orientation as Euler XYZ at indices [3:6].
        if hasattr(steering, "set_current_gripper_rotation"):
            steering.set_current_gripper_rotation(obs.ee_pose[3:6])

        def _steering_fn(trajectory, timestep, fixed_inputs, model_output):
            return steering.get_guidance(
                trajectory, timestep, fixed_inputs, model_output
            )

        if getattr(steering, "guidance_mode", "epsilon") == "dps":
            return None, _steering_fn
        return _steering_fn, None

    # ------------------------------------------------------------------
    # Forward template
    # ------------------------------------------------------------------

    def forward(
        self, obs: Observation, steering: Optional[BaseSteering] = None
    ) -> Action:
        """Predict one action. Variant differences enter via two hooks."""
        with torch.no_grad():
            rgb_obs = self._prepare_rgb(obs)
            pcd_obs = self._prepare_pcd(obs)
            curr_gripper = self._prepare_gripper(obs)

            instr_emb = self._build_instruction(obs)

            # Diagnostic logging on the first 2 forward passes — shared frame
            # plus a per-variant hook for the conditioning-specific line.
            _diag = self._log_count < 2
            if _diag:
                logger.info(
                    f"[Diag] rgb: shape={rgb_obs.shape}, "
                    f"range=[{rgb_obs.min():.3f}, {rgb_obs.max():.3f}]"
                )
                logger.info(
                    f"[Diag] pcd: shape={pcd_obs.shape}, "
                    f"range=[{pcd_obs.min():.3f}, {pcd_obs.max():.3f}]"
                )
                g = curr_gripper[0, -1, :]
                logger.info(
                    f"[Diag] gripper[-1]: pos={g[:3].cpu().numpy()}, "
                    f"quat={g[3:7].cpu().numpy()}, "
                    f"quat_norm={g[3:7].norm().item():.4f}"
                )
                self._log_conditioning_diag(instr_emb, obs)

            trajectory_mask = torch.ones(1, self.pred_horizon, device=self._device)

            guidance_fn, dps_guidance_fn = self._build_guidance_fns(steering, obs)

            trajectory = self._model(
                gt_trajectory=None,
                trajectory_mask=trajectory_mask,
                rgb_obs=rgb_obs,
                pcd_obs=pcd_obs,
                curr_gripper=curr_gripper,
                instruction=instr_emb,
                run_inference=True,
                mask_language=self._mask_language,
                guidance_fn=guidance_fn,
                dps_guidance_fn=dps_guidance_fn,
                corrector_steps=self._corrector_steps,
                corrector_step_size=self._corrector_step_size,
            )

            if _diag:
                t = trajectory[0, 0]
                logger.info(
                    f"[Diag] raw_traj[0]: pos={t[:3].cpu().numpy()}, "
                    f"quat={t[3:7].cpu().numpy()}, "
                    f"openness={t[7].item():.3f}"
                )

            if self._relative:
                action_np = self._convert_action_relative(trajectory, curr_gripper)
                if _diag:
                    logger.info(f"[Diag] after_rel2abs[0]: {action_np[0, 0]}")
            else:
                action_np = self._convert_action(trajectory)
                if _diag:
                    logger.info(f"[Diag] after_convert[0]: {action_np[0, 0]}")

            if _diag:
                self._log_count += 1

            action_np = action_np.squeeze(0)

        gripper = float(action_np[0, 6])
        return Action(trajectory=action_np, gripper=gripper)
