"""DiffuserActor policy wrapper for LangSteer inference."""

from typing import Optional, Any, Dict
from collections import deque
import logging
import numpy as np
import torch
from core.policy import BasePolicy
from core.types import Observation, Action
from core.steering import BaseSteering

try:
    from policies.diffuser_actor_components import DiffuserActor
except ImportError as e:
    _IMPORT_ERROR = e
    DiffuserActor = None
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class DiffuserActorPolicy(BasePolicy):
    """
    Wrapper for DiffuserActor diffusion policy.

    The model expects:
        - rgb_obs: (B, ncam, 3, H, W) float [0, 1]
        - pcd_obs: (B, ncam, 3, H, W) per-pixel world-space XYZ
        - instruction: (B, seq_len, 512) CLIP text embeddings
        - curr_gripper: (B, nhist, 7) ee_pose history [pos(3) + quat_wxyz(4)]
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        if _IMPORT_ERROR is not None:
            raise _IMPORT_ERROR

        self._device = torch.device(
            cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model hyperparameters
        backbone = cfg.get("backbone", "clip")
        image_size = tuple(cfg.get("image_size", [256, 256]))
        embedding_dim = cfg.get("embedding_dim", 120)
        num_vis_ins_attn_layers = cfg.get("num_vis_ins_attn_layers", 2)
        use_instruction = cfg.get("use_instruction", True)
        fps_subsampling_factor = cfg.get("fps_subsampling_factor", 5)
        gripper_loc_bounds = cfg.get(
            "gripper_loc_bounds", [[-2, -2, -2], [2, 2, 2]]
        )
        rotation_parametrization = cfg.get("rotation_parametrization", "6D")
        quaternion_format = cfg.get("quaternion_format", "wxyz")
        diffusion_timesteps = cfg.get("diffusion_timesteps", 100)
        nhist = cfg.get("nhist", 1)
        relative = cfg.get("relative", True)
        lang_enhanced = cfg.get("lang_enhanced", True)

        self.nhist = nhist
        self.image_size = image_size
        self._use_instruction = use_instruction
        self.camera_names = cfg.get("cameras", ["static", "gripper"])
        self.pred_horizon = cfg.get("pred_horizon", 1)
        self.crop_images = cfg.get("crop_images", True)
        self.text_max_length = cfg.get("text_max_length", 16)
        self._relative = relative
        self._quaternion_format = quaternion_format
        self._gripper_loc_bounds_np = np.array(gripper_loc_bounds)
        self._mask_language = cfg.get("mask_language", False)
        self._corrector_steps = cfg.get("corrector_steps", 0)
        self._corrector_step_size = cfg.get("corrector_step_size", 1e-3)

        # Build model
        self._model = DiffuserActor(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            use_instruction=use_instruction,
            fps_subsampling_factor=fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=rotation_parametrization,
            quaternion_format=quaternion_format,
            diffusion_timesteps=diffusion_timesteps,
            nhist=nhist,
            relative=relative,
            lang_enhanced=lang_enhanced,
        )
        self._model.to(self._device)
        self._model.eval()

        # Gripper history buffer: stores [pos(3), quat_wxyz(4)] per step
        self._gripper_history: deque = deque(maxlen=nhist)

        # Instruction embedding cache
        self._instruction_cache: Dict[str, torch.Tensor] = {}
        self._clip_text_model = None
        self._clip_tokenizer = None

        # Diagnostic logging counter (log first N forward passes)
        self._log_count = 0

        logger.info(f"DiffuserActorPolicy initialized on {self._device}")

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from a 3D Diffuser Actor checkpoint."""
        logger.info(f"Loading DiffuserActor checkpoint from: {path}")

        checkpoint = torch.load(
            path, map_location=self._device, weights_only=False
        )

        # Handle 3DA checkpoint format: {"weight": state_dict, ...}
        if "weight" in checkpoint:
            state_dict = checkpoint["weight"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip DDP "module." prefix if present
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
            logger.warning(f"Checkpoint has {len(unexpected)} unexpected keys: {unexpected[:5]}")

        self._model.eval()
        logger.info("DiffuserActor checkpoint loaded successfully")

    def reset(self) -> None:
        """Clear gripper history buffer."""
        self._gripper_history.clear()

    def _get_instruction_embedding(self, instruction_text: str) -> torch.Tensor:
        """
        Get CLIP token-level embeddings using HuggingFace CLIPTextModel.

        Returns:
            (seq_len, 512) tensor of CLIP text features.
        """
        if instruction_text in self._instruction_cache:
            return self._instruction_cache[instruction_text]

        # Lazy-load HuggingFace CLIP text model
        if self._clip_text_model is None:
            import transformers
            self._clip_tokenizer = transformers.CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_tokenizer.model_max_length = self.text_max_length
            self._clip_text_model = transformers.CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self._device).eval()
            logger.info(
                f"Loaded HuggingFace CLIPTextModel (max_length={self.text_max_length})"
            )

        instr = instruction_text + '.'
        tokens = self._clip_tokenizer(instr, padding="max_length")["input_ids"]
        tokens = torch.tensor(tokens).to(self._device).view(1, -1)
        with torch.no_grad():
            pred = self._clip_text_model(tokens).last_hidden_state  # (1, seq_len, 512)

        emb = pred.squeeze(0)  # (seq_len, 512)
        self._instruction_cache[instruction_text] = emb
        return emb

    def _prepare_rgb(self, obs: Observation) -> torch.Tensor:
        """
        Convert RGB dict to (1, ncam, 3, H, W) tensor in [0, 1].
        Crops CALVIN 200x200 → 160x160 (no further resize — matches training).
        """
        rgb_list = []
        for cam_name in self.camera_names:
            if cam_name not in obs.rgb:
                raise ValueError(
                    f"Camera '{cam_name}' not found in observation. "
                    f"Available: {list(obs.rgb.keys())}"
                )
            img = obs.rgb[cam_name]
            # Handle both uint8 and float formats
            if img.dtype == np.uint8:
                t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            else:
                t = torch.from_numpy(img).float().permute(2, 0, 1)
            # Crop CALVIN images (200x200 → 160x160)
            if self.crop_images and t.shape[1] >= 200 and t.shape[2] >= 200:
                t = t[:, 20:180, 20:180]
            rgb_list.append(t)

        return torch.stack(rgb_list).unsqueeze(0).to(self._device)

    def _prepare_pcd(self, obs: Observation) -> torch.Tensor:
        """
        Convert pre-deprojected per-pixel PCD images to (1, ncam, 3, H, W) tensor.
        The depth dict contains (H, W, 3) world-space XYZ arrays.
        Crops CALVIN 200x200 → 160x160 (no further resize — matches training).
        """
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
            t = torch.from_numpy(pcd_img).float().permute(2, 0, 1)  # (3, H, W)
            # Crop CALVIN images (200x200 → 160x160)
            if self.crop_images and t.shape[1] >= 200 and t.shape[2] >= 200:
                t = t[:, 20:180, 20:180]
            pcd_list.append(t)

        return torch.stack(pcd_list).unsqueeze(0).to(self._device)

    def _prepare_gripper(self, obs: Observation) -> torch.Tensor:
        """
        Build gripper history tensor: (1, nhist, 7) = [pos(3), quat_wxyz(4)].
        Converts CALVIN ee_pose [pos(3), euler_XYZ(3), gripper_width(1)] to quaternion.
        """
        from training.policies.diffuser_actor.preprocessing.calvin_utils import (
            convert_rotation,
        )

        ee_pose = obs.ee_pose.copy()
        pos = ee_pose[:3]
        euler = ee_pose[3:6]
        quat = convert_rotation(euler)  # euler → wxyz quaternion (4,)
        gripper_pose = np.concatenate([pos, quat])  # (7,)

        self._gripper_history.append(gripper_pose)

        # Pad history by repeating the oldest observation
        history = list(self._gripper_history)
        while len(history) < self.nhist:
            history.insert(0, history[0].copy())

        gripper = np.stack(history)  # (nhist, 7)
        return (
            torch.from_numpy(gripper)
            .float()
            .unsqueeze(0)
            .to(self._device)
        )  # (1, nhist, 7)

    @staticmethod
    def _convert_quat_to_euler(quat):
        """Convert quaternion (wxyz) to euler angles (XYZ).

        Uses pytorch3d transforms with pybullet fallback for gimbal lock.
        """
        from training.policies.diffuser_actor.preprocessing.pytorch3d_transforms import (
            quaternion_to_matrix,
            matrix_to_euler_angles,
        )

        quat_t = torch.as_tensor(quat)
        mat = quaternion_to_matrix(quat_t)
        euler = matrix_to_euler_angles(mat, "XYZ")
        euler_np = euler.data.cpu().numpy()

        # Fallback for NaN (gimbal lock)
        if np.isnan(euler_np).any():
            import pybullet
            flat_shape = list(quat.shape)[:-1] + [3]
            flat_quat = quat.reshape(-1, 4)
            if isinstance(flat_quat, torch.Tensor):
                flat_quat = flat_quat.data.cpu().numpy()
            # pybullet uses xyzw convention
            euler_np = np.array([
                pybullet.getEulerFromQuaternion([q[1], q[2], q[3], q[0]])
                for q in flat_quat
            ]).reshape(flat_shape)

        return euler_np

    def _convert_action(self, trajectory):
        """Convert model output to CALVIN action format.

        Input: (B, L, 8) = [pos(3), quat(4), openness_prob(1)]
        Output: (L, 7) numpy = [pos(3), euler_XYZ(3), gripper_binary(1)]

        Quaternion format depends on self._quaternion_format.
        """
        traj = trajectory.clone()
        pos = traj[..., :3].cpu().numpy()
        quat = traj[..., 3:7]  # quaternion in model's format
        openness = traj[..., 7:]

        # Reorder to wxyz if model uses xyzw
        if self._quaternion_format == 'xyzw':
            quat = quat[..., [3, 0, 1, 2]]  # xyzw → wxyz

        euler = self._convert_quat_to_euler(quat)
        gripper = (2 * (openness >= 0.5).long() - 1).cpu().numpy()

        return np.concatenate([pos, euler, gripper], axis=-1)

    def forward(
        self, obs: Observation, steering: Optional[BaseSteering] = None
    ) -> Action:
        """
        Predict an action given an observation.

        Args:
            obs: Observation with rgb dict, depth dict (per-pixel PCD), ee_pose, instruction
            steering: Optional steering module (injected into denoising loop)

        Returns:
            Action with predicted trajectory and gripper state
        """
        with torch.no_grad():
            rgb_obs = self._prepare_rgb(obs)
            pcd_obs = self._prepare_pcd(obs)
            curr_gripper = self._prepare_gripper(obs)

            instr_emb = None
            if self._use_instruction:
                instr_emb = self._get_instruction_embedding(obs.instruction)
                instr_emb = instr_emb.unsqueeze(0)  # (1, seq_len, 512)

            # Diagnostic logging on first 2 forward passes
            _diag = self._log_count < 2
            if _diag:
                logger.info(f"[Diag] rgb: shape={rgb_obs.shape}, range=[{rgb_obs.min():.3f}, {rgb_obs.max():.3f}]")
                logger.info(f"[Diag] pcd: shape={pcd_obs.shape}, range=[{pcd_obs.min():.3f}, {pcd_obs.max():.3f}]")
                g = curr_gripper[0, -1, :]
                logger.info(f"[Diag] gripper[-1]: pos={g[:3].cpu().numpy()}, quat={g[3:7].cpu().numpy()}, "
                            f"quat_norm={g[3:7].norm().item():.4f}")
                if self._use_instruction:
                    logger.info(f"[Diag] instr: shape={instr_emb.shape}, norm={instr_emb.norm().item():.3f}")
                    logger.info(f"[Diag] instruction: '{obs.instruction}'"
                                f"{' (masked)' if self._mask_language else ''}")
                else:
                    logger.info("[Diag] no language conditioning (use_instruction=False)")

            trajectory_mask = torch.ones(
                1, self.pred_horizon, device=self._device
            )

            # Build guidance function from steering module
            guidance_fn = None
            if steering is not None:
                # Provide current gripper position for relative coordinate conversion
                if hasattr(steering, 'set_current_gripper_pos'):
                    steering.set_current_gripper_pos(obs.ee_pose[:3])

                def guidance_fn(trajectory, timestep, fixed_inputs, model_output):
                    return steering.get_guidance(
                        trajectory, timestep, fixed_inputs, model_output
                    )

            # Run inference
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
                corrector_steps=self._corrector_steps,
                corrector_step_size=self._corrector_step_size,
            )
            # trajectory: (1, L, 8) = [pos(3), quat(4), openness_prob(1)]

            if _diag:
                t = trajectory[0, 0]
                logger.info(f"[Diag] raw_traj[0]: pos={t[:3].cpu().numpy()}, "
                            f"quat={t[3:7].cpu().numpy()}, openness={t[7].item():.3f}")

            # Convert to CALVIN format: [pos(3), euler_XYZ(3), gripper_binary(1)]
            action_np = self._convert_action(trajectory)  # (1, L, 7)

            if _diag:
                logger.info(f"[Diag] after_convert[0]: {action_np[0, 0]}")

            # Convert relative → absolute using current gripper pose
            if self._relative:
                # curr_gripper is (1, nhist, 7) = [pos(3), quat_wxyz(4)]
                # Pad with dummy openness to match _convert_action input format
                gripper_last = curr_gripper[:, [-1], :]  # (1, 1, 7)
                gripper_padded = torch.cat([
                    gripper_last,
                    torch.zeros_like(gripper_last[..., :1])
                ], dim=-1)  # (1, 1, 8)
                gripper_euler = self._convert_action(gripper_padded)  # (1, 1, 7)
                if _diag:
                    logger.info(f"[Diag] gripper_euler (base pose): {gripper_euler[0, 0]}")
                action_np[..., :3] += gripper_euler[..., :3]
                action_np[..., 3:6] += gripper_euler[..., 3:6]

            if _diag:
                logger.info(f"[Diag] after_rel2abs[0]: {action_np[0, 0]}")
                self._log_count += 1

            action_np = action_np.squeeze(0)  # (L, 7)

        gripper = float(action_np[0, 6])
        return Action(trajectory=action_np, gripper=gripper)
