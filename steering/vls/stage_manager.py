"""Lightweight stage manager for VLS steering.

Mirrors the proximity-based transition pattern from VoxPoser's StageManager
without any VoxPoser/LMP dependencies. Stages advance when the EE comes within
``proximity_threshold`` metres of the current stage's target keypoint.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VLSStageSpec:
    """A single stage produced by the VLM guidance generator."""

    guidance_fn: Callable
    target_world: Optional[np.ndarray]  # (3,) absolute world position
    primitive: str  # e.g. 'grasp', 'place', 'move'
    description: str


@dataclass
class VLSStageActivation:
    """Read-only snapshot consumed by the guidance computation."""

    guidance_fn: Optional[Callable]
    target_world: Optional[np.ndarray]
    primitive: str
    description: str
    stage_idx: int
    num_stages: int
    steps_in_stage: int


class VLSStageManager:
    """Owns stage list + transition logic for VLS steering.

    Transitions are proximity-based (same pattern as VoxPoser) with an
    optional grasp-completion gate that prevents advancing past a 'grasp'
    stage until the gripper is sufficiently closed and stable.
    """

    def __init__(
        self,
        proximity_threshold: float = 0.08,
        use_grasp_gate: bool = False,
        grasp_min_width: float = 0.01,
        grasp_max_width: float = 0.04,
        grasp_stability_window: int = 3,
    ) -> None:
        self._proximity_threshold = proximity_threshold
        self._use_grasp_gate = use_grasp_gate
        self._grasp_min = grasp_min_width
        self._grasp_max = grasp_max_width
        self._grasp_stability_window = grasp_stability_window

        self._stages: list[VLSStageSpec] = []
        self._current_idx: int = 0
        self._steps_in_stage: int = 0
        self._grasp_history: deque[bool] = deque(maxlen=grasp_stability_window)

    def setup(self, stages: list[VLSStageSpec]) -> None:
        """Load a new stage list for the episode."""
        self._stages = stages
        self._current_idx = 0
        self._steps_in_stage = 0
        self._grasp_history.clear()
        logger.info(f"VLSStageManager: loaded {len(stages)} stage(s)")
        for i, s in enumerate(stages):
            logger.info(f"  Stage {i}: [{s.primitive}] {s.description}")

    def current(self) -> VLSStageActivation:
        if not self._stages:
            return VLSStageActivation(
                guidance_fn=None,
                target_world=None,
                primitive="none",
                description="",
                stage_idx=0,
                num_stages=0,
                steps_in_stage=0,
            )
        spec = self._stages[self._current_idx]
        return VLSStageActivation(
            guidance_fn=spec.guidance_fn,
            target_world=spec.target_world,
            primitive=spec.primitive,
            description=spec.description,
            stage_idx=self._current_idx,
            num_stages=len(self._stages),
            steps_in_stage=self._steps_in_stage,
        )

    def check_transition(self, ee_pos: np.ndarray, gripper_width: float) -> bool:
        """Advance to the next stage if the EE is close enough to the target.

        Returns True when a transition fires.
        """
        if not self._stages or self._current_idx >= len(self._stages):
            return False

        spec = self._stages[self._current_idx]

        if spec.target_world is None:
            return False

        dist = float(np.linalg.norm(ee_pos - spec.target_world))
        close_enough = dist < self._proximity_threshold

        if not close_enough:
            return False

        if self._use_grasp_gate and spec.primitive == "grasp":
            in_range = self._grasp_min <= gripper_width <= self._grasp_max
            self._grasp_history.append(in_range)
            if not (
                len(self._grasp_history) == self._grasp_stability_window
                and all(self._grasp_history)
            ):
                return False

        self._advance()
        return True

    def increment_step(self) -> None:
        self._steps_in_stage += 1

    def _advance(self) -> None:
        if self._current_idx < len(self._stages) - 1:
            self._current_idx += 1
            self._steps_in_stage = 0
            self._grasp_history.clear()
            spec = self._stages[self._current_idx]
            logger.info(
                "VLSStageManager: → stage %d [%s] %s",
                self._current_idx,
                spec.primitive,
                spec.description,
            )
        else:
            logger.info("VLSStageManager: final stage reached, no further transitions")
