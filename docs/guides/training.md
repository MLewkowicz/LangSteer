# Training Guide

This guide covers training 3D Diffuser Actor on the CALVIN dataset. See also [SLURM_TRAINING.md](../../SLURM_TRAINING.md) for distributed training on SLURM clusters.

## Quick Start

```bash
# Train with language conditioning
uv run python scripts/train_diffuser_actor.py training=diffuser_actor_calvin

# Train without language conditioning
uv run python scripts/train_diffuser_actor.py training=diffuser_actor_nolang
```

## Configuration

Training configs are in `conf/training/`:

- **`diffuser_actor_calvin.yaml`** — Full language-conditioned training (200K iterations, batch 16)
- **`diffuser_actor_nolang.yaml`** — Without language conditioning

### Override Parameters

```bash
uv run python scripts/train_diffuser_actor.py training=diffuser_actor_calvin \
    training.batch_size=8 \
    training.num_iterations=100000
```

## Inference with Trained Model

```bash
uv run python scripts/run_experiment.py \
    policy.ckpt_path=path/to/checkpoint.pth \
    num_episodes=10
```

## Forecaster Training

```bash
uv run python -m training.forecasters.trajectory.trainer
```

Forecaster training configs are in `conf/training/forecasters/`.
