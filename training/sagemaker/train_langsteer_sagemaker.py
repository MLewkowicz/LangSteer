#!/usr/bin/env python3
"""
SageMaker training entry for LangSteer — 3D Diffuser Actor (no-language, 600K iters).

Invoked by SageMaker after it installs training/sagemaker/requirements.txt.
Remaining setup (flash-attn, networkx fix, package registration) is done here,
then torchrun launches scripts/train_diffuser_actor.py via Hydra.

Input channel "calvin" must be mounted at SM_CHANNEL_CALVIN with the structure:
    training/D+0/*.dat
    validation/D+0/*.dat

or the packaged_ABC_D layout from the 3d_diffuser_actor pipeline:
    packaged_ABC_D/training/D+0/*.dat
    packaged_ABC_D/validation/D+0/*.dat

Checkpoints land in /opt/ml/model/checkpoints/ (SM_MODEL_DIR) and are
automatically packaged into model.tar.gz by SageMaker.

Hyperparameters are merged from:
    1. DEFAULT_HYPERPARAMETERS (this file)
    2. CLI args (--key value or --key=value, SageMaker contract)
    3. SM_HPS env var (JSON dict injected by SageMaker)

``batch_size`` is per GPU: more GPUs increase global batch, not free VRAM.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    # training/sagemaker/this_file.py -> training/sagemaker/ -> training/ -> LangSteer/
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
    candidates = [
        channel,
        channel / "packaged_ABC_D",
        channel / "data" / "calvin" / "packaged_ABC_D",
    ]
    for p in candidates:
        if (p / "training").is_dir() and (p / "validation").is_dir():
            print(f"[data] Found dataset root: {p}", flush=True)
            _log_task_directories(p)
            return p
    raise FileNotFoundError(
        f"Could not find training/ and validation/ under {channel}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _log_task_directories(data_root: Path) -> None:
    """Print which ABCD task directories are present under training/ and validation/."""
    expected_tasks = ["A+0", "B+0", "C+0", "D+0"]
    for split in ("training", "validation"):
        split_dir = data_root / split
        found, missing = [], []
        for t in expected_tasks:
            td = split_dir / t
            if td.is_dir():
                n = len(list(td.glob("*.dat"))) + len(list(td.glob("*.npy")))
                found.append(f"{t}({n} files)")
            else:
                missing.append(t)
        print(f"[data] {split}/  found: {found or 'none'}  missing: {missing or 'none'}", flush=True)
    if missing:
        print(
            "[data] WARNING: missing task directories will be silently skipped by the "
            "dataset loader — training will only see available tasks.",
            flush=True,
        )


def _maybe_install_flash_attn() -> None:
    """Install flash-attn with --no-build-isolation (may compile for several minutes)."""
    print("Installing flash-attn (no-build-isolation; may take several minutes)...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "packaging", "ninja", "flash-attn",
         "--no-build-isolation"],
        check=False,  # non-fatal: model uses MultiheadCustomAttention by default
    )
    if result.returncode != 0:
        print(
            "Warning: flash-attn install failed. "
            "Training proceeds using standard attention.",
            file=sys.stderr,
        )


def _ensure_networkx_compatible() -> None:
    """Upgrade networkx to >=2.6 (urdfpy/DGL pin to 2.2 which lacks Python 3.11 APIs)."""
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


def _build_training_argv(hps: dict[str, str], checkpoint_dir: str) -> list[str]:
    """Convert merged hyperparameters to Hydra CLI overrides."""
    return [
        "training=diffuser_actor_nolang",
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
        # Fixed absolute path — avoids CWD confusion when Hydra switches dirs across ranks
        "hydra.run.dir=/opt/ml/output/hydra",
    ]


# ---------------------------------------------------------------------------
# Defaults (all values are strings; SageMaker hyperparameter contract)
# ---------------------------------------------------------------------------

DEFAULT_HYPERPARAMETERS: dict[str, str] = {
    "train_iters": "600000",
    "batch_size": "8",       # per GPU; ml.g6e.8xlarge (4× L40S) → global 32
    "batch_size_val": "2",
    "lr": "0.0003",
    "wd": "0.005",
    "val_freq": "5000",      # 120 checkpoints over 600K steps
    "log_freq": "50",
    "num_workers": "4",
    "wandb_project": "langsteer_diffuser_actor",
    "run_name": f"sm-nolang-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "nproc_per_node": "",    # empty = auto-detect GPU count
}


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = _repo_root()
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{root}:{os.environ.get('PYTHONPATH', '').strip(':')}"

    if "SM_CHANNEL_CALVIN" not in os.environ:
        raise RuntimeError("SM_CHANNEL_CALVIN is not set — add a 'calvin' input channel.")

    hps = _merge_hyperparameters()

    # Locate dataset and expose via env var used in conf/training/diffuser_actor_nolang.yaml
    channel = Path(os.environ["SM_CHANNEL_CALVIN"])
    data_root = _find_data_root(channel)
    os.environ["CALVIN_3DA_DATASET_PATH"] = str(data_root)
    print(f"Dataset root: {data_root}", flush=True)

    # Register the langsteer package (requirements.txt already covers all deps)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root), "--no-deps"],
        check=True,
    )
    _maybe_install_flash_attn()
    _ensure_networkx_compatible()

    nproc = _nproc(hps)
    print(f"Using {nproc} GPU(s) per node", flush=True)

    # Checkpoints go directly into SM_MODEL_DIR — auto-packaged into model.tar.gz
    checkpoint_dir = str(
        Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model")) / "checkpoints"
    )
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_argv = _build_training_argv(hps, checkpoint_dir)
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
