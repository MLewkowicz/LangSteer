#!/usr/bin/env python3
"""
SageMaker training entry for LangSteer — 3D Diffuser Actor Isaac mug task.

Invoked by SageMaker after it installs training/sagemaker/requirements.txt.
Remaining setup (repack HDF5 episodes, flash-attn, networkx fix, package
registration) is done here, then torchrun launches train_diffuser_actor.py.

Input channel "isaac_mug" must be mounted at SM_CHANNEL_ISAAC_MUG with:
    training/episode_*.h5
    validation/episode_*.h5

Checkpoints land in /opt/ml/model/checkpoints/ and are auto-packaged into
model.tar.gz by SageMaker.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _argv_to_hps(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            body = a[2:]
            if "=" in body:
                k, v = body.split("=", 1)
                out[k.replace("-", "_")] = v
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[body.replace("-", "_")] = argv[i + 1]
                i += 2
            else:
                out[body.replace("-", "_")] = "1"
                i += 1
        else:
            i += 1
    return out


def _merge_hyperparameters() -> dict[str, str]:
    hps = dict(DEFAULT_HYPERPARAMETERS)
    hps.update(_argv_to_hps(sys.argv[1:]))
    raw = os.environ.get("SM_HPS", "{}")
    try:
        user = json.loads(raw)
        if isinstance(user, dict):
            hps.update({k: str(v) for k, v in user.items()})
    except json.JSONDecodeError:
        pass
    return hps


def _find_data_root(channel: Path) -> Path:
    """Find the directory that directly contains training/ and validation/."""
    for p in [channel, *(channel.iterdir() if channel.is_dir() else [])]:
        if (p / "training").is_dir() and (p / "validation").is_dir():
            print(f"[data] Found dataset root: {p}", flush=True)
            return p
    raise FileNotFoundError(
        f"Could not find training/ and validation/ under {channel}."
    )


def _repack_episodes(data_root: Path, root: Path) -> None:
    """Repack HDF5 episodes for fast per-frame random access."""
    print("[repack] Repacking HDF5 episodes (this may take a few minutes)...", flush=True)
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "repack_isaac_episodes.py"),
         "--data-dir", str(data_root)],
        check=False,
    )
    if result.returncode != 0:
        print(
            "[repack] WARNING: repack returned non-zero — training will proceed "
            "with original chunking (slower data loading).",
            file=sys.stderr, flush=True,
        )
    else:
        print("[repack] Done.", flush=True)


def _log_episode_counts(data_root: Path) -> None:
    for split in ("training", "validation"):
        d = data_root / split
        n = len(list(d.glob("episode_*.h5"))) if d.is_dir() else 0
        print(f"[data] {split}/  {n} episodes", flush=True)


def _maybe_install_flash_attn() -> None:
    print("Installing flash-attn (no-build-isolation; may take several minutes)...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "packaging", "ninja", "flash-attn",
         "--no-build-isolation"],
        check=False,
    )
    if result.returncode != 0:
        print(
            "Warning: flash-attn install failed. Using standard attention.",
            file=sys.stderr,
        )


def _ensure_networkx_compatible() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "--force-reinstall",
         "--no-deps", "--no-cache-dir", "networkx>=2.6,<4"],
        check=True,
    )


def _nproc(hps: dict[str, str]) -> int:
    raw = hps.get("nproc_per_node", "")
    if raw.strip() not in ("", "0"):
        return max(1, int(raw))
    import torch
    return max(1, int(torch.cuda.device_count()))


def _build_training_argv(hps: dict[str, str], checkpoint_dir: str, data_root: Path) -> list[str]:
    return [
        "training=diffuser_actor_isaac_mug",
        f"training.train_iters={hps['train_iters']}",
        f"training.batch_size={hps['batch_size']}",
        f"training.batch_size_val={hps['batch_size_val']}",
        f"training.lr={hps['lr']}",
        f"training.wd={hps['wd']}",
        f"training.val_freq={hps['val_freq']}",
        f"training.log_freq={hps['log_freq']}",
        f"training.num_workers={hps['num_workers']}",
        f"training.checkpoint_dir={checkpoint_dir}",
        f"training.wandb_project={hps['wandb_project']}",
        f"training.experiment_name={hps['run_name']}",
        f"training.dataset.train_path={data_root}/training",
        f"training.dataset.val_path={data_root}/validation",
        "hydra.run.dir=/opt/ml/output/hydra",
    ]


DEFAULT_HYPERPARAMETERS: dict[str, str] = {
    "train_iters": "100000",
    "batch_size": "8",
    "batch_size_val": "2",
    "lr": "0.0001",
    "wd": "0.005",
    "val_freq": "2000",
    "log_freq": "50",
    "num_workers": "4",
    "wandb_project": "langsteer_diffuser_actor",
    "run_name": f"sm-isaac-mug-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "nproc_per_node": "",
}


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = _repo_root()
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{root}:{os.environ.get('PYTHONPATH', '').strip(':')}"

    if "SM_CHANNEL_ISAAC_MUG" not in os.environ:
        raise RuntimeError("SM_CHANNEL_ISAAC_MUG is not set — add an 'isaac_mug' input channel.")

    hps = _merge_hyperparameters()

    channel = Path(os.environ["SM_CHANNEL_ISAAC_MUG"])
    data_root = _find_data_root(channel)
    _log_episode_counts(data_root)

    _repack_episodes(data_root, root)

    os.environ["ISAAC_MUG_DEMO_DIR"] = str(data_root)
    print(f"Dataset root: {data_root}", flush=True)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root), "--no-deps"],
        check=True,
    )
    _maybe_install_flash_attn()
    _ensure_networkx_compatible()

    nproc = _nproc(hps)
    print(f"Using {nproc} GPU(s) per node", flush=True)

    checkpoint_dir = str(
        Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model")) / "checkpoints"
    )
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_argv = _build_training_argv(hps, checkpoint_dir, data_root)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}", "--nnodes=1",
        str(root / "scripts" / "train_diffuser_actor.py"),
        *train_argv,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    saved = sorted(Path(checkpoint_dir).glob("*.pth"))
    print(f"\nCheckpoints saved ({len(saved)} files): {checkpoint_dir}", flush=True)
    for p in saved:
        print(f"  {p.name}", flush=True)


if __name__ == "__main__":
    main()
