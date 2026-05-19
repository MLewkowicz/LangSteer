"""Language-conditioned DiffuserActor — CLIP text embeddings.

Configs: `diffuser_actor.yaml`, `diffuser_actor_maskedlang.yaml`.

Lazy-loads HuggingFace CLIPTextModel + tokenizer on the first
`_get_instruction_embedding` call, caches embeddings per instruction string.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from core.types import Observation
from policies.diffuser_actor_base import DiffuserActorBasePolicy

logger = logging.getLogger(__name__)


class LanguageDiffuserActorPolicy(DiffuserActorBasePolicy):
    """CLIP text-embedding-conditioned DiffuserActor."""

    _use_instruction = True
    _use_primitive_id = False
    _use_object_id = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.text_max_length = cfg.get("text_max_length", 16)
        self._instruction_cache: dict[str, torch.Tensor] = {}
        self._clip_text_model: Any = None
        self._clip_tokenizer: Any = None

    def _get_instruction_embedding(self, instruction_text: str) -> torch.Tensor:
        """Return (seq_len, 512) CLIP text features for `instruction_text`."""
        if instruction_text in self._instruction_cache:
            return self._instruction_cache[instruction_text]

        if self._clip_text_model is None:
            import transformers

            self._clip_tokenizer = transformers.CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_tokenizer.model_max_length = self.text_max_length
            self._clip_text_model = (
                transformers.CLIPTextModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                .to(self._device)
                .eval()
            )
            logger.info(
                f"Loaded HuggingFace CLIPTextModel (max_length={self.text_max_length})"
            )

        instr = instruction_text + "."
        tokens = self._clip_tokenizer(instr, padding="max_length")["input_ids"]
        tokens = torch.tensor(tokens).to(self._device).view(1, -1)
        with torch.no_grad():
            pred = self._clip_text_model(tokens).last_hidden_state

        emb = pred.squeeze(0)  # (seq_len, 512)
        self._instruction_cache[instruction_text] = emb
        return emb

    def _build_instruction(self, obs: Observation) -> Optional[torch.Tensor]:
        emb = self._get_instruction_embedding(obs.instruction)
        return emb.unsqueeze(0)  # (1, seq_len, 512)

    def _log_conditioning_diag(
        self, instr_emb: Optional[torch.Tensor], obs: Observation
    ) -> None:
        assert instr_emb is not None  # language variant always has an embedding
        logger.info(
            f"[Diag] instr: shape={instr_emb.shape}, norm={instr_emb.norm().item():.3f}"
        )
        logger.info(
            f"[Diag] instruction: '{obs.instruction}'"
            f"{' (masked)' if self._mask_language else ''}"
        )
