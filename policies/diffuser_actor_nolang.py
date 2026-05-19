"""No-language DiffuserActor variant.

Configs: `diffuser_actor_nolang.yaml`, `diffuser_actor_nolang_abcd.yaml`.

`instruction=None` is passed straight through to the model; the model's
encoder branch on `if self.use_instruction:` is False, so the language
cross-attention pathway is never built and `mask_language` is unread.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from core.types import Observation
from policies.diffuser_actor_base import DiffuserActorBasePolicy

logger = logging.getLogger(__name__)


class NolangDiffuserActorPolicy(DiffuserActorBasePolicy):
    """DiffuserActor with no language conditioning."""

    _use_instruction = False
    _use_primitive_id = False
    _use_object_id = False

    def _build_instruction(self, obs: Observation) -> Optional[torch.Tensor]:
        return None

    def _log_conditioning_diag(
        self, instr_emb: Optional[torch.Tensor], obs: Observation
    ) -> None:
        logger.info("[Diag] no language conditioning (use_instruction=False)")
