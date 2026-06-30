#!/usr/bin/env python3
"""
SageMaker training entry for pi-0.5 fine-tuning on CALVIN.

SageMaker copies the LeRobot dataset splits from S3 before this script runs:
    SM_CHANNEL_TRAIN → /opt/ml/input/data/train/   (calvin_abc_d_train contents)
    SM_CHANNEL_VAL   → /opt/ml/input/data/val/     (calvin_abc_d_val contents)

This script then:
  1. Clones OpenPi and installs it via uv
  2. Symlinks the SM channels into HF_LEROBOT_HOME so LeRobot can find them
  3. Downloads pi-0.5 base weights (from S3 if weights_s3_uri is set, else GCS)
  4. Injects the CALVIN training config into OpenPi's config registry
  5. Computes dataset normalization statistics
  6. Launches training via OpenPi's train.py
  7. Copies final checkpoints to SM_MODEL_DIR for automatic S3 packaging

Hyperparameters (all strings, merged from defaults → CLI → SM_HPS env):
    train_repo_id    Hugging Face-style repo id for training split
    val_repo_id      Hugging Face-style repo id for validation split
    exp_name         W&B / checkpoint experiment name
    num_train_steps  Number of gradient steps (default 50000)
    batch_size       Global batch size across all GPUs (default 32)
    wandb_project    W&B project name
    weights_s3_uri   S3 URI to pre-staged pi05_base weights (optional).
                     If empty, downloads from gs://openpi-assets/checkpoints/pi05_base.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


OPENPI_DIR   = Path("/opt/openpi")
WEIGHTS_DIR  = Path("/tmp/pi05_base")
LEROBOT_HOME = Path("/tmp/lerobot")
OPENPI_REPO  = "https://github.com/Physical-Intelligence/openpi.git"

DEFAULT_HYPERPARAMETERS: dict[str, str] = {
    "train_repo_id":   "aryannav/calvin_abc_d_train",
    "val_repo_id":     "aryannav/calvin_abc_d_val",
    "exp_name":        f"pi05-calvin-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "num_train_steps": "50000",
    "batch_size":      "32",
    "wandb_project":   "langsteer_pi05",
    "weights_s3_uri":  "",
}


# ---------------------------------------------------------------------------
# Hyperparameter helpers (same contract as train_langsteer_sagemaker.py)
# ---------------------------------------------------------------------------

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


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    full_env = {**os.environ, **(env or {})}
    print(f"[cmd] {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd, env=full_env)


# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------

def clone_openpi() -> None:
    print("\n=== Cloning OpenPi ===", flush=True)
    # uv is not present in the SageMaker PyTorch DLC — install it first
    _run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    if OPENPI_DIR.exists():
        shutil.rmtree(OPENPI_DIR)
    _run(["git", "clone", "--depth=1", OPENPI_REPO, str(OPENPI_DIR)])
    _run(["uv", "sync"], cwd=OPENPI_DIR)


def setup_lerobot_home(train_repo_id: str, val_repo_id: str) -> None:
    """Symlink SM data channels into HF_LEROBOT_HOME/<repo_id> directory tree."""
    print("\n=== Setting up HF_LEROBOT_HOME ===", flush=True)

    sm_train = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    sm_val   = Path(os.environ.get("SM_CHANNEL_VAL",   "/opt/ml/input/data/val"))

    for repo_id, channel in [(train_repo_id, sm_train), (val_repo_id, sm_val)]:
        dest = LEROBOT_HOME / repo_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink() if dest.is_symlink() else shutil.rmtree(dest)
        dest.symlink_to(channel)
        print(f"  {dest} → {channel}", flush=True)

    os.environ["HF_LEROBOT_HOME"] = str(LEROBOT_HOME)
    print(f"HF_LEROBOT_HOME={LEROBOT_HOME}", flush=True)


# Script run inside the openpi uv venv to create meta/tasks.jsonl.
# lerobot >=v2.1 requires tasks.jsonl for LeRobotDatasetMetadata.__init__.
#
# Our datasets were created with a lerobot version that already stored task
# indices in the data parquets (task_index: int64) and saved the string↔index
# mapping as meta/tasks.parquet (index=task_string, col=task_index).
# Newer lerobot expects the same mapping as meta/tasks.jsonl (jsonlines).
# We convert tasks.parquet → tasks.jsonl without touching data parquets.
_TASKS_JSONL_SCRIPT = """\
import json
import sys
import pandas as pd
from pathlib import Path

dataset_dir = Path(sys.argv[1])
tasks_file   = dataset_dir / "meta" / "tasks.jsonl"

if tasks_file.exists():
    print(f"  tasks.jsonl already present in {dataset_dir.name}, skipping", flush=True)
    sys.exit(0)

tasks_parquet = dataset_dir / "meta" / "tasks.parquet"
if not tasks_parquet.exists():
    print(f"  ERROR: neither tasks.jsonl nor tasks.parquet found in {dataset_dir}/meta/", flush=True)
    sys.exit(1)

# tasks.parquet has the task string as the row index and task_index as a column.
# e.g.  index="open the drawer"  task_index=3
df = pd.read_parquet(tasks_parquet)
with open(tasks_file, "w") as f:
    for task_str, row in df.iterrows():
        f.write(json.dumps({"task_index": int(row["task_index"]), "task": task_str}) + "\\n")

print(f"  Converted tasks.parquet → tasks.jsonl ({len(df)} tasks) in {dataset_dir.name}", flush=True)
"""


_EPISODES_JSONL_SCRIPT = """\
import json
import sys
import os
import glob
import pandas as pd
from pathlib import Path

dataset_dir = Path(sys.argv[1])
episodes_file = dataset_dir / "meta" / "episodes.jsonl"

if episodes_file.exists():
    print(f"  episodes.jsonl already present in {dataset_dir.name}, skipping", flush=True)
    sys.exit(0)

# Our datasets store episodes in meta/episodes/chunk-XXX/file-XXX.parquet
# (lerobot v3.0 chunked format). We flatten these into a single jsonlines file
# with the schema that lerobot v2.1's load_episodes() expects:
#   {"episode_index": int, "tasks": [str, ...], "length": int}
chunk_files = sorted(glob.glob(str(dataset_dir / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
if not chunk_files:
    print(f"  ERROR: no meta/episodes/chunk-*/file-*.parquet found in {dataset_dir}", flush=True)
    sys.exit(1)

# Process one chunk at a time to avoid GC crashes from large object arrays in pd.concat.
# Chunks are already sorted by episode_index so no global sort is needed.
# Include chunk_index and file_index so lerobot's get_data_file_path() can locate data parquets.
WANT_COLS = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
             "dataset_from_index", "dataset_to_index"]
# Discover which columns actually exist in this dataset's schema
import pyarrow.parquet as pq
schema_names = set(pq.read_schema(chunk_files[0]).names)
READ_COLS = [c for c in WANT_COLS if c in schema_names]

n_written = 0
with open(episodes_file, "w") as fh:
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file, columns=READ_COLS)
        for _, row in df.iterrows():
            tasks = list(row["tasks"]) if hasattr(row["tasks"], "__iter__") and not isinstance(row["tasks"], str) else [row["tasks"]]
            rec = {
                "episode_index": int(row["episode_index"]),
                "tasks": tasks,
                "length": int(row["length"]),
            }
            if "data/chunk_index" in row.index:
                rec["chunk_index"] = int(row["data/chunk_index"])
            if "data/file_index" in row.index:
                rec["file_index"] = int(row["data/file_index"])
            if "dataset_from_index" in row.index:
                rec["from"] = int(row["dataset_from_index"])
            if "dataset_to_index" in row.index:
                rec["to"] = int(row["dataset_to_index"])
            fh.write(json.dumps(rec) + "\\n")
            n_written += 1
        del df  # free object-dtype numpy arrays before loading next chunk

print(f"  Created episodes.jsonl ({n_written} episodes) in {dataset_dir.name}", flush=True)
os._exit(0)  # bypass Python finalizers — numpy object arrays crash on GC
"""


# Script to create meta/episodes_stats.jsonl.
# lerobot v2.1's LeRobotDatasetMetadata.load_metadata() also requires this file.
#
# Our datasets store per-episode stats inside the episode chunk parquets as
# columns named stats/<feature>/<aggregation> (e.g. stats/image/min, stats/action/mean).
# We read those columns and write them to the flat jsonlines format lerobot expects:
#   {"episode_index": N, "stats": {"image": {"min": [...], "max": [...], ...}, ...}}
_EPISODES_STATS_JSONL_SCRIPT = """\
import json
import sys
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

AGGS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]

def to_json_safe(val):
    \"\"\"Recursively convert nested numpy arrays / scalars to plain Python types.
    Object-dtype arrays need element-wise recursion; numeric arrays use .tolist().
    \"\"\"
    if isinstance(val, np.ndarray):
        if val.dtype == object:
            # object arrays may contain nested ndarrays — recurse element-wise
            return [to_json_safe(x) for x in val]
        return val.tolist()  # numeric arrays: .tolist() converts fully
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, (list, tuple)):
        return [to_json_safe(x) for x in val]
    return val

dataset_dir = Path(sys.argv[1])
out_file = dataset_dir / "meta" / "episodes_stats.jsonl"

if out_file.exists():
    print(f"  episodes_stats.jsonl already present in {dataset_dir.name}, skipping", flush=True)
    sys.exit(0)

chunk_files = sorted(glob.glob(str(dataset_dir / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
if not chunk_files:
    print(f"  ERROR: no meta/episodes/chunk-*/file-*.parquet found in {dataset_dir}", flush=True)
    sys.exit(1)

# Discover feature names from first chunk schema (no data read needed)
import pyarrow.parquet as pq
schema = pq.read_schema(chunk_files[0])
features = sorted({
    col.split("/")[1]
    for col in schema.names
    if col.startswith("stats/") and len(col.split("/")) == 3 and col.split("/")[2] in AGGS
})

n_written = 0
with open(out_file, "w") as fh:
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file)
        for _, row in df.iterrows():
            stats = {}
            for feat in features:
                feat_stats = {}
                for agg in AGGS:
                    col = f"stats/{feat}/{agg}"
                    if col in df.columns:
                        raw = row[col]
                        feat_stats[agg] = to_json_safe(np.array(raw) if not isinstance(raw, np.ndarray) else raw)
                stats[feat] = feat_stats
            fh.write(json.dumps({"episode_index": int(row["episode_index"]), "stats": stats}) + "\\n")
            n_written += 1
        del df  # free object-dtype arrays to avoid GC crash on exit

print(f"  Created episodes_stats.jsonl ({n_written} episodes) in {dataset_dir.name}", flush=True)
os._exit(0)  # bypass Python finalizers — numpy object arrays crash on GC
"""


def create_tasks_jsonl(train_repo_id: str, val_repo_id: str) -> None:
    """Create meta jsonlines files required by lerobot v2.1.

    lerobot v2.1 requires tasks.jsonl, episodes.jsonl, and episodes_stats.jsonl
    for LeRobotDatasetMetadata.__init__. Our datasets store these in v3.0 chunked
    parquet format; we flatten/convert them here.
    """
    print("\n=== Creating lerobot v2.1 meta jsonlines files ===", flush=True)

    scripts = {
        "/tmp/_create_tasks_jsonl.py": _TASKS_JSONL_SCRIPT,
        "/tmp/_create_episodes_jsonl.py": _EPISODES_JSONL_SCRIPT,
        "/tmp/_create_episodes_stats_jsonl.py": _EPISODES_STATS_JSONL_SCRIPT,
    }
    for path, content in scripts.items():
        Path(path).write_text(content)

    for repo_id in [train_repo_id, val_repo_id]:
        dataset_dir = LEROBOT_HOME / repo_id
        if not dataset_dir.exists():
            print(f"  {repo_id}: directory not found, skipping", flush=True)
            continue
        for script_path in scripts:
            _run(
                ["uv", "run", "python", script_path, str(dataset_dir)],
                cwd=OPENPI_DIR,
                env={"HF_LEROBOT_HOME": str(LEROBOT_HOME)},
            )

    for path in scripts:
        Path(path).unlink(missing_ok=True)


def download_weights(weights_s3_uri: str) -> None:
    """Download pi-0.5 base weights from S3 (preferred) or GCS."""
    print("\n=== Downloading pi-0.5 base weights ===", flush=True)

    if WEIGHTS_DIR.exists():
        print(f"  Weights already at {WEIGHTS_DIR}, skipping.", flush=True)
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if weights_s3_uri.strip():
        print(f"  Source: {weights_s3_uri} (S3)", flush=True)
        _run(["aws", "s3", "sync", weights_s3_uri.rstrip("/") + "/", str(WEIGHTS_DIR)])
    else:
        print("  Source: gs://openpi-assets/checkpoints/pi05_base (GCS)", flush=True)
        # gsutil is available on most SageMaker PyTorch DLC images; install if missing.
        try:
            subprocess.run(["gsutil", "version"], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            _run([sys.executable, "-m", "pip", "install", "-q", "gsutil"])
        _run([
            "gsutil", "-m", "cp", "-r",
            "gs://openpi-assets/checkpoints/pi05_base/.",
            str(WEIGHTS_DIR),
        ])

    print(f"  Weights ready at {WEIGHTS_DIR}", flush=True)


def inject_calvin_config() -> None:
    """Copy pi05_calvin_config.py and write wrapper scripts that avoid circular imports.

    We deliberately do NOT patch openpi's config.py — doing so causes a circular
    import because config.py would import pi05_calvin_config.py, which in turn
    imports from the still-partially-initialized config.py.

    Instead we write thin wrapper scripts (compute_norm_stats_calvin.py /
    train_calvin.py) that fully load config.py first, then register our config,
    then delegate to the original main() function.
    """
    print("\n=== Injecting CALVIN config into OpenPi ===", flush=True)

    src_config = (
        Path(__file__).resolve().parent.parent / "common" / "pi05_calvin_config.py"
    )
    dst_config = OPENPI_DIR / "src" / "openpi" / "training" / "pi05_calvin_config.py"
    shutil.copy(src_config, dst_config)
    print(f"  Copied {src_config.name} → {dst_config}", flush=True)

    # Preamble shared by both wrapper scripts:
    # 1. fully initialize openpi.training.config
    # 2. then import and register our config (no circular dep at this point)
    # 3. update BOTH _CONFIGS and _CONFIGS_DICT — get_config() uses _CONFIGS_DICT,
    #    which is built at import time and never updated automatically.
    # 4. add scripts/ dir to sys.path — scripts/ has no __init__.py so it is NOT
    #    an importable package; "from openpi.scripts.X import Y" always fails with
    #    ModuleNotFoundError. We resolve siblings by inserting the directory.
    # 5. patch LeRobotDatasetMetadata.get_data_file_path so lerobot can locate our
    #    multi-episode batch parquets. Lerobot v3.0 computes the path as:
    #      data_path.format(episode_chunk=ep//chunks_size, episode_index=ep)
    #    but our info.json uses {chunk_index}/{file_index} keys and stores multiple
    #    episodes per file. The patch reads chunk_index/file_index from episodes.jsonl
    #    and routes each episode to the correct actual parquet file.
    preamble = (
        "import sys, os, json\n"
        "from pathlib import Path\n"
        "import openpi.training.config as _cfg\n"
        "from openpi.training.pi05_calvin_config import PI05_CALVIN_CONFIGS\n"
        "_cfg._CONFIGS.extend(PI05_CALVIN_CONFIGS)\n"
        "_cfg._CONFIGS_DICT.update({c.name: c for c in PI05_CALVIN_CONFIGS})\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
        "\n"
        "def _patch_lerobot_file_path():\n"
        "    try:\n"
        "        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata\n"
        "        _ep_cache = {}\n"
        "        def _ep_map(root):\n"
        "            k = str(root)\n"
        "            if k not in _ep_cache:\n"
        "                m = {}\n"
        "                ep_f = Path(root) / 'meta' / 'episodes.jsonl'\n"
        "                if ep_f.exists():\n"
        "                    for l in ep_f.read_text().splitlines():\n"
        "                        ep = json.loads(l)\n"
        "                        m[ep['episode_index']] = (ep.get('chunk_index', 0), ep.get('file_index', 0))\n"
        "                _ep_cache[k] = m\n"
        "            return _ep_cache[k]\n"
        "        def _patched(self, ep_idx):\n"
        "            ch, fi = _ep_map(self.root).get(ep_idx, (0, 0))\n"
        "            return Path(f'data/chunk-{ch:03d}/file-{fi:03d}.parquet')\n"
        "        LeRobotDatasetMetadata.get_data_file_path = _patched\n"
        "        print('[lerobot-patch] get_data_file_path patched for batch parquets', flush=True)\n"
        "    except Exception as _e:\n"
        "        print(f'[lerobot-patch] WARNING: {_e}', flush=True)\n"
        "_patch_lerobot_file_path()\n"
        "\n"
        "def _patch_datasets_list_feature():\n"
        "    try:\n"
        "        import datasets.features.features as _ff\n"
        "        if 'List' not in _ff._FEATURE_TYPES:\n"
        "            # Parquet files created by older lerobot/datasets embed '_type': 'List'\n"
        "            # in Arrow schema metadata; newer datasets renamed this to 'Sequence'.\n"
        "            _ff._FEATURE_TYPES['List'] = _ff._FEATURE_TYPES['Sequence']\n"
        "            print('[datasets-patch] Added List->Sequence feature type alias', flush=True)\n"
        "        else:\n"
        "            print('[datasets-patch] List feature type already registered', flush=True)\n"
        "    except Exception as _e:\n"
        "        print(f'[datasets-patch] WARNING: {_e}', flush=True)\n"
        "_patch_datasets_list_feature()\n"
        "\n"
        "def _patch_lerobot_action_rename():\n"
        "    try:\n"
        "        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset\n"
        "        _orig_load_hf = LeRobotDataset.load_hf_dataset\n"
        "        def _patched_load_hf(self):\n"
        "            ds = _orig_load_hf(self)\n"
        "            # OpenPi passes delta_timestamps={'actions': [...]} (Libero convention).\n"
        "            # CALVIN parquets use 'action' (singular). Rename so _query_hf_dataset\n"
        "            # can find the column when it looks up 'actions'.\n"
        "            if 'action' in ds.column_names and 'actions' not in ds.column_names:\n"
        "                ds = ds.rename_column('action', 'actions')\n"
        "                print('[lerobot-patch] Renamed action->actions in hf_dataset', flush=True)\n"
        "            return ds\n"
        "        LeRobotDataset.load_hf_dataset = _patched_load_hf\n"
        "        print('[lerobot-patch] load_hf_dataset patched for action->actions rename', flush=True)\n"
        "    except Exception as _e:\n"
        "        print(f'[lerobot-patch] WARNING action-rename: {_e}', flush=True)\n"
        "_patch_lerobot_action_rename()\n"
    )

    # compute_norm_stats.py uses tyro.cli(main) in __main__; we must go through
    # tyro.cli so it parses --config-name from sys.argv.
    # IMPORTANT: guard with if __name__ == '__main__' — PyTorch DataLoader uses
    # the 'spawn' start method which reimports this file in each worker. Without
    # the guard, the worker re-executes tyro.cli(main) → tries to spawn more
    # workers → RuntimeError: "new process before bootstrapping phase finished".
    norm_wrapper = OPENPI_DIR / "scripts" / "compute_norm_stats_calvin.py"
    norm_wrapper.write_text(
        preamble
        + "import tyro\n"
        + "from compute_norm_stats import main\n"
        + "if __name__ == '__main__':\n"
        + "    sys.exit(tyro.cli(main) or 0)\n"
    )

    # train.py's __main__ calls main(_config.cli()); same __name__ guard needed.
    train_wrapper = OPENPI_DIR / "scripts" / "train_calvin.py"
    train_wrapper.write_text(
        preamble
        + "from train import main\n"
        + "if __name__ == '__main__':\n"
        + "    sys.exit(main(_cfg.cli()) or 0)\n"
    )

    print(f"  Wrote {norm_wrapper.name} and {train_wrapper.name}", flush=True)


def compute_norm_stats(hps: dict[str, str]) -> None:
    print("\n=== Computing normalization statistics ===", flush=True)
    # Prepend nvidia cuDNN PyPI package paths to LD_LIBRARY_PATH so JAX can
    # initialize GPU without hitting "undefined symbol: cudnnGetLibConfig".
    # See _nvidia_lib_env() for the full explanation.
    env = {"HF_LEROBOT_HOME": str(LEROBOT_HOME), **_nvidia_lib_env()}
    _run(
        ["uv", "run", "scripts/compute_norm_stats_calvin.py", "--config-name", "pi05_calvin"],
        cwd=OPENPI_DIR,
        env=env,
    )


def _nvidia_lib_env() -> dict[str, str]:
    """Fix JAX GPU initialization: cudnnGetLibConfig undefined symbol.

    Problem
    -------
    The SageMaker PyTorch DLC has a system cuDNN at /lib/x86_64-linux-gnu/.
    JAX's GPU backend finds libcudnn_graph.so.9 there via ldconfig cache
    (ldconfig bypasses LD_LIBRARY_PATH).  libcudnn_graph.so.9 needs the symbol
    cudnnGetLibConfig from libcudnn.so.9, but the system libcudnn.so.9 is
    missing that symbol (older/incomplete installation).

    Fix
    ---
    1. Find libcudnn.so.9 from the openpi uv venv (nvidia-cudnn-cu12==9.5.x
       installed by uv sync — known good, has cudnnGetLibConfig).
    2. LD_PRELOAD it so the symbol lands in the global symbol table BEFORE
       libcudnn_graph.so.9 is loaded.  The dynamic linker resolves
       cudnnGetLibConfig from the preloaded lib regardless of which
       libcudnn_graph.so.9 is picked up by ldconfig.
    3. Also prepend all nvidia PyPI lib dirs to LD_LIBRARY_PATH so the other
       cuDNN sub-libs (libcudnn_ops.so.9 etc.) resolve cleanly.
    """
    import glob

    patterns = [
        # openpi uv venv — primary source (nvidia-cudnn-cu12 installed by uv sync)
        "/opt/openpi/.venv/lib/python3.*/site-packages/nvidia/*/lib",
        "/opt/openpi/.venv/lib/python3.*/site-packages/nvidia/*/lib64",
        # SageMaker base conda env — secondary (PyTorch deps)
        "/opt/conda/lib/python3.*/site-packages/nvidia/*/lib",
        "/opt/conda/lib/python3.*/site-packages/nvidia/*/lib64",
    ]
    extra: list[str] = []
    for pat in patterns:
        extra.extend(sorted(glob.glob(pat)))
    seen: set[str] = set()
    unique = [p for p in extra if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]

    # LD_PRELOAD: force-load the good libcudnn.so.9 before any other library
    # can load the broken system version.  This puts cudnnGetLibConfig into the
    # global symbol table so libcudnn_graph.so.9 can resolve it regardless of
    # which libcudnn_graph.so.9 ldconfig/dlopen finds.
    cudnn_preload: list[str] = []
    for lib_dir in unique:
        candidate = os.path.join(lib_dir, "libcudnn.so.9")
        if os.path.exists(candidate) and candidate not in cudnn_preload:
            cudnn_preload.append(candidate)
            break  # one good libcudnn.so.9 is enough

    current_ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    new_ldpath = ":".join(unique + ([current_ldpath] if current_ldpath else []))

    env: dict[str, str] = {"LD_LIBRARY_PATH": new_ldpath}
    if cudnn_preload:
        current_preload = os.environ.get("LD_PRELOAD", "")
        new_preload = ":".join(cudnn_preload + ([current_preload] if current_preload else []))
        env["LD_PRELOAD"] = new_preload
        print(f"[cuda-fix] LD_PRELOAD={new_preload}", flush=True)
    if unique:
        print(f"[cuda-fix] LD_LIBRARY_PATH prepended: {':'.join(unique[:4])}...", flush=True)
    return env


def run_training(hps: dict[str, str]) -> None:
    print("\n=== Launching pi-0.5 training ===", flush=True)
    env = {"HF_LEROBOT_HOME": str(LEROBOT_HOME), **_nvidia_lib_env()}
    _run(
        [
            "uv", "run", "scripts/train_calvin.py", "pi05_calvin",
            f"--exp-name={hps['exp_name']}",
            f"--batch-size={hps['batch_size']}",
            f"--num-train-steps={hps['num_train_steps']}",
            "--overwrite",
        ],
        cwd=OPENPI_DIR,
        env=env,
    )


def copy_checkpoints(exp_name: str) -> None:
    print("\n=== Copying checkpoints to SM_MODEL_DIR ===", flush=True)
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    # OpenPi saves to OPENPI_DIR/checkpoints/<exp_name>/
    src = OPENPI_DIR / "checkpoints" / exp_name
    if src.exists():
        dst = model_dir / "checkpoints" / exp_name
        shutil.copytree(src, dst, dirs_exist_ok=True)
        saved = sorted(dst.rglob("*"))
        print(f"  {len(saved)} files copied to {dst}", flush=True)
    else:
        print(f"  WARNING: checkpoint dir not found at {src}", flush=True)


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if "SM_CHANNEL_TRAIN" not in os.environ:
        raise RuntimeError(
            "SM_CHANNEL_TRAIN is not set. "
            "Ensure 'train' and 'val' input channels are configured in the submit script."
        )

    hps = _merge_hyperparameters()
    print(f"Hyperparameters: {hps}", flush=True)

    clone_openpi()
    setup_lerobot_home(hps["train_repo_id"], hps["val_repo_id"])
    create_tasks_jsonl(hps["train_repo_id"], hps["val_repo_id"])
    download_weights(hps["weights_s3_uri"])
    inject_calvin_config()
    compute_norm_stats(hps)
    run_training(hps)
    copy_checkpoints(hps["exp_name"])

    print("\nTraining complete.", flush=True)


if __name__ == "__main__":
    main()
