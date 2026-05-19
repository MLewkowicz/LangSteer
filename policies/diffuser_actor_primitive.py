"""Primitive-id-conditioned DiffuserActor.

Config: `diffuser_actor_primitive.yaml`.

Replaces CLIP text embedding with `nn.Embedding(num_primitives, D)` at the
model level. The wrapper drives the current primitive id via `set_primitive`
(invoked at each stage transition by the steering module), and emits a (1, 1)
long tensor as the model's `instruction` arg.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from core.types import Observation
from policies.diffuser_actor_base import DiffuserActorBasePolicy

logger = logging.getLogger(__name__)


class PrimitiveDiffuserActorPolicy(DiffuserActorBasePolicy):
    """DiffuserActor conditioned on a discrete primitive id."""

    _use_instruction = True
    _use_primitive_id = True
    _use_object_id = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._num_primitives = cfg.get("num_primitives", 4)
        self._current_primitive_id: Optional[int] = None

    def set_primitive(self, primitive_id: int) -> None:
        """Set the active primitive id; called by steering at stage transitions."""
        if not (0 <= primitive_id < self._num_primitives):
            raise ValueError(
                f"primitive_id={primitive_id} out of range [0, {self._num_primitives})"
            )
        self._current_primitive_id = int(primitive_id)

    def _build_instruction(self, obs: Observation) -> Optional[torch.Tensor]:
        if self._current_primitive_id is None:
            raise RuntimeError(
                "Primitive-id mode active but no primitive set. "
                "The policy expects a steering module (e.g. "
                "steering=voxposer) to drive set_primitive(idx) at "
                "every stage transition. Either add steering=voxposer "
                "to your run, or call policy.set_primitive(idx) "
                "manually before each forward()."
            )
        return torch.tensor(
            [[self._current_primitive_id]],
            dtype=torch.long,
            device=self._device,
        )  # (1, 1)

    def _log_conditioning_diag(
        self, instr_emb: Optional[torch.Tensor], obs: Observation
    ) -> None:
        logger.info(f"[Diag] primitive_id={self._current_primitive_id}")
