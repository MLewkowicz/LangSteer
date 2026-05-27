"""Target-orientation rotation steering for the diffusion policy.

Biases the denoising loop toward a specific end-effector orientation by pulling
the predicted rotation toward a target. Designed for the bimodal place task: the
base policy *completes* the wrist inversion reliably once it starts but rarely
*initiates* it from the upright start, so a gentle rotational nudge tips the
sample into the inverted (mode-B) basin while the model fills in the rest.

Critical convention: with `relative=True` the model's denoising-space rotation is
a BODY-FRAME relative rotation R_rel, where R_world = R_base @ R_rel and R_base is
the current EE orientation (see policies/diffuser_actor_base._convert_action_relative).
So the absolute target orientation is converted to the relative target
    R_rel_target = R_base.T @ R_target_world
each forward pass, against the live EE pose. (This differs from steering/tweedie.py,
which uses the absolute orientation as its rotation reference — that is only
consistent with relative=False; do not copy it here.)

Operates in the model's 6D rotation subspace (trajectory[..., 3:9]) using the
shared get_ortho6d_from_rotation_matrix convention. Epsilon mode adds a Tweedie
ε-space delta to the noise prediction; DPS mode adds an x-space correction to
x_{t-1}. Gate to lower timesteps via `start_guidance_timestep` to focus the nudge
where the two basins separate.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch

from core.steering import BaseSteering
from policies.diffuser_actor_components.rotation_utils import (
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    quaternion_to_matrix,
)
from training.policies.diffuser_actor.preprocessing.calvin_utils import (
    convert_rotation,
)

logger = logging.getLogger(__name__)


def _euler_to_matrix(euler_xyz, device) -> torch.Tensor:
    """Euler XYZ (3,) -> rotation matrix (3, 3), via the model's wxyz-quat path."""
    quat = convert_rotation(np.asarray(euler_xyz, dtype=np.float32))  # (4,) wxyz
    quat_t = normalise_quat(torch.from_numpy(quat).float())
    return quaternion_to_matrix(quat_t).to(device)  # (3, 3)


class TargetRotationSteering(BaseSteering):
    """Pull the predicted rotation toward a target absolute orientation."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        target_euler = cfg.get("target_euler", None)
        if target_euler is None:
            raise ValueError(
                "TargetRotationSteering requires `target_euler` (absolute target "
                "orientation as Euler XYZ radians, length 3)."
            )
        self.device = cfg.get("device", "cuda")
        self.guidance_strength = float(cfg.get("guidance_strength", 1.0))
        self.horizon = int(cfg.get("horizon", 20))
        # Apply guidance only at timesteps <= this (the low-t band where the two
        # rotation basins separate). Default permissive (always on).
        self.start_guidance_timestep = int(cfg.get("start_guidance_timestep", 10_000))
        # epsilon: add Tweedie ε-delta to the noise pred; dps: add x-space
        # correction to x_{t-1}. Read by policy._build_guidance_fns to route.
        self.guidance_mode = str(cfg.get("guidance_mode", "epsilon"))
        # Per-horizon quadratic ramp: early waypoints (still upright) get
        # `ramp_floor` of the pull, late waypoints get full pull — matches the
        # data, where the inversion happens late in the window.
        self.ramp_floor = float(cfg.get("ramp_floor", 0.0))

        # Target absolute orientation as a rotation matrix (3, 3).
        self._R_target_world = _euler_to_matrix(target_euler, self.device)
        # Relative 6D target, recomputed per forward from the live EE pose.
        self._rotation_target_6d: Optional[torch.Tensor] = None

        self.rotation_scheduler = None
        self.position_scheduler = None

        logger.info(
            f"Initialized TargetRotationSteering: target_euler={list(target_euler)}, "
            f"strength={self.guidance_strength}, mode={self.guidance_mode}, "
            f"start_t={self.start_guidance_timestep}, horizon={self.horizon}"
        )

    # ------------------------------------------------------------------
    # Wiring hooks (called by run_experiment / the policy)
    # ------------------------------------------------------------------

    def set_rotation_scheduler(self, scheduler) -> None:
        self.rotation_scheduler = scheduler

    def set_position_scheduler(self, scheduler) -> None:
        # Unused (rotation-only steering); stored for API parity.
        self.position_scheduler = scheduler

    def set_current_gripper_rotation(self, ee_euler_xyz: np.ndarray) -> None:
        """Recompute the relative 6D target from the live EE orientation.

        Called by policy._build_guidance_fns with obs.ee_pose[3:6] before each
        forward pass. R_rel_target = R_base.T @ R_target_world.
        """
        R_base = _euler_to_matrix(ee_euler_xyz, self.device)  # (3, 3)
        R_rel = R_base.transpose(0, 1) @ self._R_target_world  # (3, 3)
        self._rotation_target_6d = get_ortho6d_from_rotation_matrix(
            R_rel.unsqueeze(0)
        )[0]  # (6,)

    # Lifecycle no-ops for run_experiment compatibility.
    def setup_episode(self, task_name: str):
        return None, None

    def increment_step(self) -> None:
        pass

    def reset(self) -> None:
        self._rotation_target_6d = None

    # ------------------------------------------------------------------
    # Guidance
    # ------------------------------------------------------------------

    def get_guidance(
        self,
        current_sample: torch.Tensor,
        timestep: int,
        obs_embedding: Any,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Return a guidance tensor pulling rotation toward the relative target.

        Epsilon mode: shape matches model_output (added to ε before the scheduler
        step). DPS mode: shape matches current_sample = x_t (added to x_{t-1}).
        Both carry the delta only in the rotation slice [..., 3:9].
        """
        # Container must match the tensor the model adds the result to.
        container = model_output if self.guidance_mode != "dps" else current_sample
        zero = torch.zeros_like(container)

        if self._rotation_target_6d is None or self.rotation_scheduler is None:
            return zero

        t = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
        if t > self.start_guidance_timestep:
            return zero

        B, L, _ = model_output.shape
        H = min(self.horizon, L)

        abar = float(self.rotation_scheduler.alphas_cumprod[t])
        abar = max(abar, 1e-6)
        sqrt_abar = abar ** 0.5
        sqrt_1m = (1.0 - abar) ** 0.5

        eps_rot = model_output[:, :H, 3:9]
        x_t_rot = current_sample[:, :H, 3:9]
        # Tweedie x0 estimate for rotation (squaredcos rotation schedule).
        x0_rot = (x_t_rot - sqrt_1m * eps_rot) / sqrt_abar  # (B, H, 6)

        target = self._rotation_target_6d.to(container.device).view(1, 1, 6)
        h_idx = torch.arange(H, device=container.device, dtype=x0_rot.dtype)
        ramp = self.ramp_floor + (1.0 - self.ramp_floor) * (h_idx / max(H - 1, 1)) ** 2
        ramp = ramp.view(1, H, 1)

        if self.guidance_mode == "dps":
            # Nudge x_{t-1} directly toward the target rotation.
            delta = self.guidance_strength * (target - x0_rot) * ramp
        else:
            # Tweedie ε-delta: out = out + delta shifts x0 by -coeff*delta, so a
            # +(x0 - target) delta moves x0 toward target.
            coeff = sqrt_1m / sqrt_abar
            delta = self.guidance_strength * coeff * (x0_rot - target) * ramp

        zero[:, :H, 3:9] = delta
        return zero
