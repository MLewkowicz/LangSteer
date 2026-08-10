"""pi0.5 (OpenPI) baseline policy — a thin, numpy-only websocket client.

The pi0.5 JAX/Flax model is far too large to share a process (or an 8GB GPU) with
LangSteer's PyTorch stack, so it runs in a separate OpenPI process behind a
websocket policy server (see ``serve_pi05.py``). This class packages the LangSteer
``Observation`` into the observation dict the server expects, calls ``client.infer``,
and returns a receding-horizon ``Action`` made of RELATIVE rel_actions rows.

Server observation contract (post-repack keys read by OpenPI ``LiberoInputs``):
    observation/image        uint8 (H,W,3)   static camera  (native 200x200)
    observation/wrist_image  uint8 (H,W,3)   gripper camera (native 84x84)
    observation/state        float32 (15,)   CALVIN robot_obs
    prompt                   str             language instruction
``client.infer(obs)["actions"]`` returns shape ``(action_horizon, 7)`` of RELATIVE
rel_actions ``[dx, dy, dz, d_euler_x, d_euler_y, d_euler_z, gripper]``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from core.policy import BasePolicy
from core.steering import BaseSteering
from core.types import Action, Observation

logger = logging.getLogger(__name__)


class Pi05Policy(BasePolicy):
    """Baseline pi0.5 policy that queries a remote OpenPI websocket server."""

    # No language/primitive/object conditioning hooks; steering is unsupported.
    _use_instruction = True
    _use_primitive_id = False
    _use_object_id = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._host = cfg.get("host", "127.0.0.1")
        self._port = int(cfg.get("port", 8000))
        self._action_horizon = int(cfg.get("action_horizon", 10))
        self._replan_steps = int(cfg.get("replan_steps", 5))
        self._static_key = cfg.get("static_key", "static")
        self._gripper_key = cfg.get("gripper_key", "gripper_native")
        self._prompt_override = cfg.get("prompt", None)
        self._client = None
        self._connect()

    def _connect(self) -> None:
        # openpi_client is numpy/msgpack/websockets only (no jax, no torch).
        from openpi_client import websocket_client_policy

        logger.info(
            f"[pi05] connecting to {self._host}:{self._port} "
            f"(blocks/retries until the server is reachable)…"
        )
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=self._host, port=self._port
        )
        logger.info(f"[pi05] connected. server metadata={self._client.get_server_metadata()}")

    def load_checkpoint(self, path: str) -> None:
        # Weights live on the server; conf/policy/pi05.yaml keeps ckpt_path null so
        # instantiate_policy never calls this. Kept for BasePolicy compliance.
        logger.info(f"[pi05] load_checkpoint is a no-op (weights served remotely); path={path!r}")

    def reset(self) -> None:
        # No client-side action cache in the receding-horizon design (see forward()).
        if self._client is not None:
            self._client.reset()  # server-side no-op

    # ------------------------------------------------------------------ obs packing
    @staticmethod
    def _to_uint8_hwc(img: np.ndarray) -> np.ndarray:
        """Coerce an image to contiguous uint8 HWC.

        The eval (PCD) path delivers RGB as float32 in [0, 1]; convert back to uint8.
        Defensively transpose a CHW array to HWC.
        """
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
        if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        return np.ascontiguousarray(a)

    def _build_obs(self, obs: Observation) -> dict:
        if self._static_key not in obs.rgb:
            raise KeyError(
                f"[pi05] missing static camera obs.rgb['{self._static_key}']; "
                f"available keys: {list(obs.rgb)}"
            )
        if self._gripper_key not in obs.rgb:
            raise KeyError(
                f"[pi05] missing native gripper obs.rgb['{self._gripper_key}']; "
                f"available keys: {list(obs.rgb)}. Ensure envs/calvin.py exposes the "
                f"raw 84x84 gripper (rgb['gripper_native'])."
            )
        return {
            "observation/image": self._to_uint8_hwc(obs.rgb[self._static_key]),       # 200x200x3
            "observation/wrist_image": self._to_uint8_hwc(obs.rgb[self._gripper_key]),  # 84x84x3
            "observation/state": np.asarray(obs.proprio, dtype=np.float32),           # (15,)
            "prompt": str(self._prompt_override or obs.instruction),
        }

    # --------------------------------------------------------------------- forward
    def forward(self, obs: Observation, steering: Optional[BaseSteering] = None) -> Action:
        if steering is not None:
            raise ValueError("Pi05Policy is a baseline (no-steering) policy; steering is unsupported.")

        result = self._client.infer(self._build_obs(obs))
        chunk = np.asarray(result["actions"], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise ValueError(
                f"[pi05] expected an (H,7) action chunk from the server, got shape {chunk.shape}."
            )

        # Receding horizon: execute the first `replan_steps` relative rows, then re-infer.
        n = min(self._replan_steps, chunk.shape[0])
        rows = chunk[:n].copy()
        return Action(trajectory=rows, gripper=float(rows[0, 6]), relative=True)
