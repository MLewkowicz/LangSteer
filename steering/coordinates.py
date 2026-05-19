"""Coordinate-frame transforms between Diffuser Actor model space and world.

Diffuser Actor predicts positions in a gripper-relative, normalized [-1, 1]
frame. The value map lives in absolute world coordinates. `PositionTransform`
owns the bidirectional mapping plus the voxel-grid lookup, so the rest of the
steering code never touches scaling constants directly.

Public surface:
    PositionTransform(*, gripper_loc_bounds, workspace_min, workspace_max,
                      is_relative, device)
    pt.set_gripper_pos(np.ndarray)
    pt.current_gripper_pos -> Optional[torch.Tensor]
    pt.model_to_world(model_pos)
    pt.world_gradient_to_model(grad_world)
    pt.lookup_voxel_gradient(positions_world, gradient_field, map_size)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class PositionTransform:
    """Model ↔ world coordinate conversions + voxel-gradient lookup.

    Holds the per-step gripper position so the orchestrator only has to
    update it once and both `model_to_world` and downstream consumers see
    the same value.
    """

    def __init__(
        self,
        *,
        gripper_loc_bounds: Optional[np.ndarray],
        workspace_min: np.ndarray,
        workspace_max: np.ndarray,
        is_relative: bool,
        device: str,
    ) -> None:
        self._gripper_loc_bounds: Optional[torch.Tensor] = (
            torch.tensor(gripper_loc_bounds, dtype=torch.float32)
            if gripper_loc_bounds is not None
            else None
        )
        self._workspace_min = workspace_min
        self._workspace_max = workspace_max
        self._is_relative = is_relative
        self._device = device
        self._current_gripper_pos: Optional[torch.Tensor] = None

    def set_gripper_pos(self, gripper_pos: np.ndarray) -> None:
        """Update the cached absolute gripper position used for relative→abs."""
        self._current_gripper_pos = torch.tensor(
            gripper_pos,
            dtype=torch.float32,
            device=self._device,
        )

    @property
    def current_gripper_pos(self) -> Optional[torch.Tensor]:
        return self._current_gripper_pos

    def model_to_world(self, model_pos: torch.Tensor) -> torch.Tensor:
        """Reverse Diffuser Actor's two-stage normalization.

          1. normalize_pos: model = (world_rel - pos_min) / (pos_max - pos_min) * 2 - 1
          2. convert2rel:   world_rel = world_abs - gripper_pos

        Args:
            model_pos: (B, H, 3) in model [-1, 1] space.

        Returns:
            (B, H, 3) absolute world positions in meters.
        """
        if self._gripper_loc_bounds is not None:
            bounds = self._gripper_loc_bounds.to(model_pos.device)
            pos_min = bounds[0]
            pos_max = bounds[1]
            world_rel = (model_pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min
        else:
            world_rel = model_pos

        if self._is_relative and self._current_gripper_pos is not None:
            gripper = self._current_gripper_pos.to(model_pos.device)
            world_abs = world_rel + gripper.view(1, 1, 3)
        else:
            world_abs = world_rel
        return world_abs

    def world_gradient_to_model(self, grad_world: torch.Tensor) -> torch.Tensor:
        """Chain-rule conversion of a world-space gradient into model space.

        Jacobian of normalize_pos is constant per-axis: dmodel/dworld = 2 /
        (pos_max - pos_min). The reverse for gradients is the reciprocal,
        applied as `grad_world * (pos_max - pos_min) / 2`.
        """
        if self._gripper_loc_bounds is not None:
            bounds = self._gripper_loc_bounds.to(grad_world.device)
            scale_factor = (bounds[1] - bounds[0]) / 2.0
            return grad_world * scale_factor.unsqueeze(0).unsqueeze(0)
        return grad_world

    def lookup_voxel_gradient(
        self,
        positions_world: torch.Tensor,
        gradient_field: torch.Tensor,
        map_size: int,
    ) -> torch.Tensor:
        """Sample the precomputed value-map gradient at world-frame positions.

        Args:
            positions_world: (B, H, 3) absolute world XYZ (meters).
            gradient_field: (M, M, M, 3) voxel-indexed gradient field.
            map_size: M.

        Returns:
            (B, H, 3) world-frame gradient vectors (1/m).
        """
        B, H, _ = positions_world.shape
        M = map_size

        ws_min = torch.tensor(
            self._workspace_min,
            device=positions_world.device,
            dtype=positions_world.dtype,
        )
        ws_max = torch.tensor(
            self._workspace_max,
            device=positions_world.device,
            dtype=positions_world.dtype,
        )

        clamped = torch.clamp(positions_world, ws_min, ws_max)
        voxel_float = (clamped - ws_min) / (ws_max - ws_min) * (M - 1)
        voxel_idx = torch.clamp(voxel_float.long(), 0, M - 1)

        flat_idx = voxel_idx.reshape(-1, 3)
        ix = flat_idx[:, 0]
        iy = flat_idx[:, 1]
        iz = flat_idx[:, 2]
        grad_flat = gradient_field[ix, iy, iz]  # (B*H, 3)

        # Convert voxel-space gradient into world-space (1/m) units.
        resolution = torch.tensor(
            (self._workspace_max - self._workspace_min) / M,
            device=positions_world.device,
            dtype=positions_world.dtype,
        )
        grad_world = grad_flat / resolution.unsqueeze(0)
        return grad_world.reshape(B, H, 3)
