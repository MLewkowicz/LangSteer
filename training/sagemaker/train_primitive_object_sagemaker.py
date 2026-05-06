#!/usr/bin/env python3
"""
SageMaker training entry for LangSteer — 3D Diffuser Actor (primitive + object conditioning).

Invoked by SageMaker after it installs training/sagemaker/requirements.txt.

Input channel "primitive_object" must be mounted at SM_CHANNEL_PRIMITIVE_OBJECT
pointing to s3://calvin-abcd-dataset-bucket/calvin_object_action/, which has the
structure:

    packaged_ABC_D/
        training/{A,B,C,D}+0/*.dat
        training/lang_annotations/primitive_object_lang_ann.npy
        validation/{A,B,C,D}+0/*.dat
        validation/lang_annotations/primitive_object_lang_ann.npy

Both CALVIN_3DA_PRIMITIVE_OBJ_DATASET_PATH and CALVIN_PRIMITIVE_OBJ_ANN_PATH are set
to the packaged_ABC_D/ root inside the channel — the .yaml configs append sub-paths.

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


_TASK_DIRS = ["A+0", "B+0", "C+0", "D+0"]


def _find_split_root(channel: Path) -> Path:
    """Find the directory that has both training/ and validation/ subdirectories."""
    candidates = [channel / "packaged_ABC_D", channel]
    for p in candidates:
        if (p / "training").is_dir() and (p / "validation").is_dir():
            print(f"[data] Found split root: {p}", flush=True)
            return p
    raise FileNotFoundError(
        f"Could not find training/ and validation/ under {channel}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _find_episodes_root(split_dir: Path) -> Path:
    """Find the directory directly containing A+0/, B+0/, etc.

    Handles the case where the packager emitted an extra nesting level:
        split_dir/training/A+0/   (S3 layout: packaged_ABC_D/training/training/A+0/)
        split_dir/A+0/            (flat layout: packaged_ABC_D/training/A+0/)
    """
    if any((split_dir / t).is_dir() for t in _TASK_DIRS):
        return split_dir
    # One level deeper — named after the split itself or "training"/"validation"
    for sub in split_dir.iterdir():
        if sub.is_dir() and any((sub / t).is_dir() for t in _TASK_DIRS):
            print(f"[data] Extra nesting detected; episode root: {sub}", flush=True)
            return sub
    raise FileNotFoundError(
        f"Could not find task dirs {_TASK_DIRS} under {split_dir} or any subdirectory."
    )


def _find_ann_root(split_root: Path) -> Path:
    """Find the directory whose training/lang_annotations/primitive_object_lang_ann.npy exists."""
    ann = split_root / "training" / "lang_annotations" / "primitive_object_lang_ann.npy"
    if ann.exists():
        print(f"[data] Found annotation root: {split_root}", flush=True)
        return split_root
    raise FileNotFoundError(
        f"Could not find training/lang_annotations/primitive_object_lang_ann.npy "
        f"under {split_root}."
    )


def _log_task_directories(train_root: Path, val_root: Path) -> None:
    for label, root in (("training", train_root), ("validation", val_root)):
        found: list[str] = []
        missing: list[str] = []
        for t in _TASK_DIRS:
            td = root / t
            if td.is_dir():
                n = len(list(td.glob("*.dat"))) + len(list(td.glob("*.npy")))
                found.append(f"{t}({n} files)")
            else:
                missing.append(t)
        print(
            f"[data] {label}/  found: {found or 'none'}  missing: {missing or 'none'}",
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


def _build_training_argv(
    hps: dict[str, str],
    checkpoint_dir: str,
    train_path: str,
    val_path: str,
    ann_path_train: str,
    ann_path_val: str,
) -> list[str]:
    return [
        "training=diffuser_actor_primitive_object",
        f"training.dataset.train_path={train_path}",
        f"training.dataset.val_path={val_path}",
        f"training.dataset.primitive_ann_path_train={ann_path_train}",
        f"training.dataset.primitive_ann_path_val={ann_path_val}",
        f"training.train_iters={hps['train_iters']}",
        f"training.batch_size={hps['batch_size']}",
        f"training.batch_size_val={hps['batch_size_val']}",
        f"training.lr={hps['lr']}",
        f"training.wd={hps['wd']}",
        f"training.val_freq={hps['val_freq']}",
        f"training.log_freq={hps['log_freq']}",
        f"training.num_workers={hps['num_workers']}",
        f"training.policy.num_primitives={hps['num_primitives']}",
        f"training.policy.num_objects={hps['num_objects']}",
        f"training.checkpoint_dir={checkpoint_dir}",
        f"training.wandb_project={hps['wandb_project']}",
        f"training.experiment_name={hps['run_name']}",
        "hydra.run.dir=/opt/ml/output/hydra",
    ]


DEFAULT_HYPERPARAMETERS: dict[str, str] = {
    "train_iters":    "600000",
    "batch_size":     "8",
    "batch_size_val": "2",
    "lr":             "0.0003",
    "wd":             "0.005",
    "val_freq":       "5000",
    "log_freq":       "50",
    "num_workers":    "4",
    "num_primitives": "5",   # grasp=0, push=1, pull=2, place=3, rotate=4
    "num_objects":    "8",   # block=0, blue_block=1, drawer_handle=2, led_button=3,
                             # lightbulb_switch=4, pink_block=5, red_block=6, slider_handle=7
    "wandb_project":  "langsteer_diffuser_actor",
    "run_name":       f"sm-primitive-object-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "nproc_per_node": "",
}


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = _repo_root()
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{root}:{os.environ.get('PYTHONPATH', '').strip(':')}"

    if "SM_CHANNEL_PRIMITIVE_OBJECT" not in os.environ:
        raise RuntimeError(
            "SM_CHANNEL_PRIMITIVE_OBJECT is not set — add a 'primitive_object' "
            "input channel pointing to s3://calvin-abcd-dataset-bucket/calvin_object_action/"
        )

    hps = _merge_hyperparameters()

    channel = Path(os.environ["SM_CHANNEL_PRIMITIVE_OBJECT"])

    # Locate the split-level root (has training/ and validation/ subdirs),
    # then probe one level deeper for the actual episode directories.
    # Handles extra nesting: packaged_ABC_D/training/training/A+0/ vs
    #                        packaged_ABC_D/training/A+0/
    split_root  = _find_split_root(channel)
    train_root  = _find_episodes_root(split_root / "training")
    val_root    = _find_episodes_root(split_root / "validation")
    ann_root    = _find_ann_root(split_root)
    ann_path_train = str(ann_root / "training"  / "lang_annotations" / "primitive_object_lang_ann.npy")
    ann_path_val   = str(ann_root / "validation" / "lang_annotations" / "primitive_object_lang_ann.npy")
    print(f"Train episodes: {train_root}", flush=True)
    print(f"Val episodes:   {val_root}", flush=True)
    print(f"Ann (train):    {ann_path_train}", flush=True)
    print(f"Ann (val):      {ann_path_val}", flush=True)
    _log_task_directories(train_root, val_root)

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

    train_argv = _build_training_argv(
        hps, checkpoint_dir,
        train_path=str(train_root),
        val_path=str(val_root),
        ann_path_train=ann_path_train,
        ann_path_val=ann_path_val,
    )
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
