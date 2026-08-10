#!/usr/bin/env python3
"""Serve the fine-tuned pi0.5 CALVIN checkpoint over the OpenPI websocket protocol.

Run this INSIDE the OpenPI uv environment (NOT the LangSteer env) on the GPU box.
`pi05_calvin_config.py` must be importable from here — it imports openpi modules, so
it must run in the openpi venv. Copy `training/common/pi05_calvin_config.py` and this
file into the openpi repo root (or any dir on PYTHONPATH).

The checkpoint STEP directory must contain `params/` and
`assets/<asset_id>/norm_stats.json` (train_state/ is not needed for serving).

    # in the openpi repo, openpi venv
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        uv run python serve_pi05.py --ckpt-dir ./ckpt/199999 --port 8000

The LangSteer eval connects to this via policies/pi05.py (openpi_client). The obs
dict the client sends (post-repack keys) is:
    observation/image       uint8 (H,W,3)  static 200x200
    observation/wrist_image uint8 (H,W,3)  gripper native 84x84
    observation/state       float32 (15,)  CALVIN robot_obs
    prompt                  str
and `infer()["actions"]` returns (action_horizon=10, 7) relative rel_actions.
"""
from __future__ import annotations

import argparse
import logging
import socket

# pi05_calvin is NOT a built-in openpi config, so we import the TrainConfig object
# directly rather than resolving it by name via scripts/serve_policy.py.
from pi05_calvin_config import PI05_CALVIN_CONFIGS
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve fine-tuned pi0.5 CALVIN over websocket.")
    ap.add_argument("--ckpt-dir", required=True,
                    help="Checkpoint STEP dir containing params/ and assets/ (e.g. ./ckpt/199999).")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--default-prompt", default=None,
                    help="Fallback prompt if a client omits 'prompt' (clients here always send one).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    config = PI05_CALVIN_CONFIGS[0]  # name="pi05_calvin"

    logging.info("Loading pi0.5 policy from %s (config=%s)…", args.ckpt_dir, config.name)
    # create_trained_policy restores JAX params (bf16) from <ckpt>/params and norm
    # stats from <ckpt>/assets/<asset_id>/norm_stats.json, and builds the serving
    # transform stack (LiberoInputs -> Normalize -> resize/tokenize / unnorm -> [:7]).
    policy = policy_config.create_trained_policy(
        config, args.ckpt_dir, default_prompt=args.default_prompt
    )

    logging.info("Serving pi0.5 on %s:%d (%s)", args.host, args.port, socket.gethostname())
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
