# LangSteer — 3D Diffuser Actor SLURM Training Guide

## What This Experiment Is

Training a **3D Diffuser Actor** diffusion policy on a small subset of the CALVIN robotic manipulation benchmark. Two tasks: `open_drawer` and `move_slider_left` from CALVIN scene D. Two training modes:

| Mode | Config | Description |
|------|--------|-------------|
| **No-language** | `training=diffuser_actor_nolang` | Structurally smaller model — language attention layers surgically removed at construction time. ~3M params. Checkpoints are architecture-incompatible with language mode. |
| **Language-conditioned** | `training=diffuser_actor_calvin` | Full model with CLIP text encoder, vision-language cross-attention, trajectory-language cross-attention. ~4M params. |

The policy predicts 20-step end-effector trajectories (6-DoF pose + gripper) via DDPM (epsilon prediction, 100 diffusion steps). Input: two RGB-D views (static + wrist cameras, 200×200) → per-pixel 3D point clouds.

---

## Key Files

| Purpose | Path |
|---------|------|
| Training entry point | `scripts/train_diffuser_actor.py` |
| No-language config | `conf/training/diffuser_actor_nolang.yaml` |
| Language config | `conf/training/diffuser_actor_calvin.yaml` |
| Trainer implementation | `training/policies/diffuser_actor/trainer.py` |
| Dataset loader | `training/policies/diffuser_actor/dataset.py` |
| Model (policy components) | `policies/diffuser_actor_components/` |
| Hydra root config | `conf/config.yaml` |

---

## Dataset

**Location on local machine:** `/home/mlewkowicz/LangSteer/cache/diffuser_actor_data/`

```
cache/diffuser_actor_data/
├── training/
│   └── D+0/
│       ├── ann_0.dat       ← blosc-compressed pickle of one annotated episode
│       ├── ann_7.dat
│       └── ...             (303 files total — open_drawer + move_slider_left)
├── validation/
│   └── D+0/
│       └── ...             (62 files)
├── training.pkl            ← precomputed CLIP token embeddings, shape (N_ann, 53, 512)
└── validation.pkl          ← same for validation split
```

**Format:** Each `.dat` file is `blosc.compress(pickle.dumps(state_dict))` where `state_dict` is a 7-element list:
```
[frame_ids, rgb_pcd, action_tensors, camera_dicts, gripper_tensors, trajectories, annotation_id]
```
- `rgb_pcd`: `(T, 2, 2, 3, H, W)` float32 — (timesteps, cameras, rgb/pcd, channels, height, width)
- `trajectories`: list of `(traj_len, 8)` tensors — 3D pos + 6D rotation + gripper

**The `*.pkl` files are only needed for language-conditioned training.** The no-language model only uses `.dat` files.

**To transfer dataset to cluster:**
```bash
rsync -avz --progress cache/diffuser_actor_data/ <user>@<cluster>:<project_dir>/cache/diffuser_actor_data/
```

---

## Environment Setup

**Package manager:** `uv` (not pip/conda)

```bash
# On the cluster node
cd <project_dir>
uv sync                    # installs all deps into .venv
source .venv/bin/activate
```

**Python version:** 3.10–3.12 (strictly enforced)

**Key dependencies:** torch, diffusers, transformers, hydra-core, calvin-env (installed from GitHub), blosc, clip (OpenAI, from GitHub), wandb

---

## Training Commands

### Single GPU

```bash
# No-language model (recommended first run — simpler, fewer params)
uv run python scripts/train_diffuser_actor.py training=diffuser_actor_nolang

# Language-conditioned model
uv run python scripts/train_diffuser_actor.py training=diffuser_actor_calvin

# Smoke test (100 iters, no WandB)
uv run python scripts/train_diffuser_actor.py training=diffuser_actor_nolang \
    training.train_iters=100 training.val_freq=50 training.use_wandb=false training.batch_size=4
```

### Key CLI Overrides

```bash
training.use_wandb=false          # disable WandB (useful for debugging)
training.batch_size=8             # per-GPU batch size (default 16)
training.train_iters=20000        # total gradient steps
training.lr=3e-4                  # learning rate
training.checkpoint_dir=outputs/checkpoints/my_run
training.experiment_name=my_run_name
```

### Dataset Path Override

If the dataset is not at the default path, set via env var:

```bash
CALVIN_3DA_DATASET_PATH=/path/to/data uv run python scripts/train_diffuser_actor.py ...
# or for instructions pkl (language mode only):
CALVIN_3DA_INSTRUCTIONS_PATH=/path/to/data uv run python scripts/train_diffuser_actor.py ...
```

---

## Training Hyperparameters (No-Language Mode)

These are already set correctly in `conf/training/diffuser_actor_nolang.yaml`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `train_iters` | 20,000 | Scaled from original 600K: 900 passes × 303 eps / batch 16 ≈ 17K, rounded up |
| `batch_size` | 16 | Per GPU |
| `lr` | 3e-4 | Matches original paper (inherited config had 1e-4 by mistake) |
| `interpolation_length` | 20 | Original used 20-step trajectories (inherited config had 100) |
| `val_freq` | 500 | Validate every 500 iters |
| `use_ema` | true | EMA with power=0.75 |

**Original paper context:** 600K iters, batch 30, 6 GPUs, ~20K+ episodes (all 34 tasks × 4 CALVIN scenes). Our 303 episodes is ~5% of even the single-scene dataset.

---

## SLURM Script Template

Key points for the SLURM script:

1. The `srun` command should call `train_diffuser_actor.py`
2. Set `CALVIN_3DA_DATASET_PATH` env var pointing to dataset location on cluster
3. Adjust GPU count (single GPU is fine for 303 episodes; no DDP required but trainer supports it)

```bash
#!/bin/bash
#SBATCH --job-name=diffuser_actor_nolang
#SBATCH --output=logs/%j_da_nolang.out
#SBATCH --error=logs/%j_da_nolang.err
#SBATCH --partition=<your-partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # single GPU sufficient for 303 episodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00            # 20K iters ~2-3h on single GPU

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export CALVIN_3DA_DATASET_PATH=<path-to-data-on-cluster>
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1

mkdir -p logs

srun python scripts/train_diffuser_actor.py training=diffuser_actor_nolang \
    training.experiment_name="da_nolang_slurm_${SLURM_JOB_ID}"
```

**For multi-GPU DDP** (if desired): The trainer auto-detects `RANK`, `LOCAL_RANK`, `WORLD_SIZE` env vars set by `srun`. Increase `--ntasks-per-node` and `--gres=gpu:N` accordingly. No code changes needed — `DiffuserActorTrainingWorkspace.__init__` calls `_setup_distributed()` automatically.

---

## Checkpoints

Saved to `outputs/checkpoints/diffuser_actor_nolang/` (relative to working dir). Top-3 checkpoints by validation loss are kept. Each checkpoint contains:
- Model weights
- EMA model weights
- Optimizer state
- `arch_config: {use_instruction, lang_enhanced}` — required to reload correctly
- Iteration number and validation loss

To resume: set `training.resume=true` and `training.resume_checkpoint_path=<path>` in CLI.

---

## WandB

Project: `langsteer_diffuser_actor`
Run name: auto-generated as `diffuser_actor_nolang_YYYYMMDD_HHMMSS`

Set `WANDB_API_KEY` env var on the cluster, or use `training.use_wandb=false` to disable.
