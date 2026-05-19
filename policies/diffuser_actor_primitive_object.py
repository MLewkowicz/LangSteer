"""Primitive + object-id-conditioned DiffuserActor — production variant.

Config: `diffuser_actor_primitive_object.yaml`.

Subclasses `PrimitiveDiffuserActorPolicy` because object conditioning is
strictly an extension of primitive conditioning (`use_object_id=True`
requires `use_primitive_id=True`). Adds a parallel `nn.Embedding(num_objects,
D)` at the model level, and emits a (1, 2) long tensor as the model's
`instruction` arg.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from core.types import Observation
from policies.diffuser_actor_primitive import PrimitiveDiffuserActorPolicy

logger = logging.getLogger(__name__)


class PrimitiveObjectDiffuserActorPolicy(PrimitiveDiffuserActorPolicy):
    """DiffuserActor conditioned on (primitive id, object id) tuples."""

    _use_object_id = True

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._num_objects = cfg.get("num_objects", 8)
        self._current_object_id: Optional[int] = None

    def set_object(self, object_id: int) -> None:
        """Set the active object id; called by steering at stage transitions."""
        if not (0 <= object_id < self._num_objects):
            raise ValueError(
                f"object_id={object_id} out of range [0, {self._num_objects})"
            )
        self._current_object_id = int(object_id)

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
        if self._current_object_id is None:
            raise RuntimeError(
                "Object-id mode active but no object set. "
                "Add steering=voxposer (whose composer emits "
                "object slots in every stage tuple) or call "
                "policy.set_object(idx) manually before forward()."
            )
        return torch.tensor(
            [[self._current_primitive_id, self._current_object_id]],
            dtype=torch.long,
            device=self._device,
        )  # (1, 2)

    def _log_conditioning_diag(
        self, instr_emb: Optional[torch.Tensor], obs: Observation
    ) -> None:
        logger.info(
            f"[Diag] primitive_id={self._current_primitive_id} "
            f"object_id={self._current_object_id}"
        )
