#!/usr/bin/env python3
"""
SageMaker training entry for LangSteer — 3D Diffuser Actor (primitive-ID conditioning).

Invoked by SageMaker after it installs training/sagemaker/requirements.txt.

Input channel "primitive" must be mounted at SM_CHANNEL_PRIMITIVE with the structure:
    training/{A,B,C,D}+0/*.dat
    training/lang_annotations/primitive_lang_ann.npy
    validation/D+0/*.dat
    validation/lang_annotations/primitive_lang_ann.npy

Both CALVIN_3DA_PRIMITIVE_DATASET_PATH and CALVIN_PRIMITIVE_ANN_PATH are set to the
same local data root — the .yaml config appends the sub-paths for each.

Checkpoints land in /opt/ml/model/checkpoints/ (SM_MODEL_DIR) and are
automatically packaged into model.tar.gz by SageMaker.
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


def _find_dat_root(channel: Path) -> Path:
    """Find the directory that directly contains training/{A,B,C,D}+0/*.dat files."""
    candidates = [
        channel / "packaged_ABC_D",
        channel / "packaged_primitive",
        channel,
    ]
    for p in candidates:
        if (p / "training").is_dir() and (p / "validation").is_dir():
            print(f"[data] Found .dat root: {p}", flush=True)
            _log_task_directories(p)
            return p
    raise FileNotFoundError(
        f"Could not find training/ and validation/ under {channel}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _find_ann_root(channel: Path) -> Path:
    """Find the directory whose training/lang_annotations/primitive_lang_ann.npy exists."""
    candidates = [channel, channel / "packaged_ABC_D"]
    for p in candidates:
        ann = p / "training" / "lang_annotations" / "primitive_lang_ann.npy"
        if ann.exists():
            print(f"[data] Found annotation root: {p}", flush=True)
            return p
    raise FileNotFoundError(
        f"Could not find training/lang_annotations/primitive_lang_ann.npy under {channel}."
    )


def _log_task_directories(dat_root: Path) -> None:
    expected_tasks = ["A+0", "B+0", "C+0", "D+0"]
    missing: list[str] = []
    for split in ("training", "validation"):
        split_dir = dat_root / split
        found: list[str] = []
        split_missing: list[str] = []
        for t in expected_tasks:
            td = split_dir / t
            if td.is_dir():
                n = len(list(td.glob("*.dat"))) + len(list(td.glob("*.npy")))
                found.append(f"{t}({n} files)")
            else:
                split_missing.append(t)
        missing.extend(split_missing)
        print(
            f"[data] {split}/  found: {found or 'none'}  missing: {split_missing or 'none'}",
            flush=True,
        )
    if missing:
        print(
            "[data] WARNING: missing task directories will be silently skipped.",
            flush=True,
        )


def _maybe_install_flash_attn() -> None:
    print("Installing flash-attn (no-build-isolation; may take several minutes)...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "packaging", "ninja", "flash-attn",
         "--no-build-isolation"],
        check=False,
    )
    if result.returncode != 0:
        print(
            "Warning: flash-attn install failed. Training proceeds using standard attention.",
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


def _build_training_argv(hps: dict[str, str], checkpoint_dir: str) -> list[str]:
    return [
        "training=diffuser_actor_primitive",
        f"training.train_iters={hps['train_iters']}",
        f"training.batch_size={hps['batch_size']}",
        f"training.batch_size_val={hps['batch_size_val']}",
        f"training.lr={hps['lr']}",
        f"training.wd={hps['wd']}",
        f"training.val_freq={hps['val_freq']}",
        f"training.log_freq={hps['log_freq']}",
        f"training.num_workers={hps['num_workers']}",
        f"training.policy.num_primitives={hps['num_primitives']}",
        f"training.checkpoint_dir={checkpoint_dir}",
        f"training.wandb_project={hps['wandb_project']}",
        f"training.experiment_name={hps['run_name']}",
        "hydra.run.dir=/opt/ml/output/hydra",
    ]


DEFAULT_HYPERPARAMETERS: dict[str, str] = {
    "train_iters":    "200000",
    "batch_size":     "8",
    "batch_size_val": "2",
    "lr":             "0.0003",
    "wd":             "0.005",
    "val_freq":       "5000",
    "log_freq":       "50",
    "num_workers":    "4",
    "num_primitives": "4",
    "wandb_project":  "langsteer_diffuser_actor",
    "run_name":       f"sm-primitive-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "nproc_per_node": "",
}


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = _repo_root()
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{root}:{os.environ.get('PYTHONPATH', '').strip(':')}"

    if "SM_CHANNEL_PRIMITIVE" not in os.environ:
        raise RuntimeError(
            "SM_CHANNEL_PRIMITIVE is not set — add a 'primitive' input channel."
        )

    hps = _merge_hyperparameters()

    channel = Path(os.environ["SM_CHANNEL_PRIMITIVE"])

    # .dat files live under packaged_ABC_D/; annotation .npy files live at the
    # channel root.  The two env vars point to different local paths:
    #   CALVIN_3DA_PRIMITIVE_DATASET_PATH → packaged_ABC_D/
    #     dataset.train_path → .../training/{A,B,C,D}+0/*.dat
    #   CALVIN_PRIMITIVE_ANN_PATH → channel root
    #     primitive_ann_path_train → .../training/lang_annotations/primitive_lang_ann.npy
    dat_root = _find_dat_root(channel)
    ann_root = _find_ann_root(channel)
    os.environ["CALVIN_3DA_PRIMITIVE_DATASET_PATH"] = str(dat_root)
    os.environ["CALVIN_PRIMITIVE_ANN_PATH"] = str(ann_root)
    print(f"DAT root:  {dat_root}", flush=True)
    print(f"Ann root:  {ann_root}", flush=True)

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
