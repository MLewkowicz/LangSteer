# DP3 Training Guide for LangSteer

This guide covers training DP3 diffusion policy on CALVIN dataset with distributed data parallel (DDP) support for SLURM clusters.

## Quick Start

### 1. Dataset Preparation

Convert CALVIN dataset to Zarr format (one-time operation):

```bash
# Set your CALVIN dataset path
export CALVIN_ROOT=/path/to/calvin_debug_dataset

# Convert to Zarr
bash scripts/convert_calvin_debug.sh

# Or manually:
python scripts/convert_calvin_to_zarr.py \
    --root_dir /path/to/calvin_dataset \
    --save_path data/calvin_training.zarr \
    --overwrite
```

### 2. Single-GPU Training (Local Development)

```bash
# Debug config (small dataset, few epochs)
export CALVIN_ZARR_PATH=data/calvin_debug.zarr
python scripts/train_dp3.py training=dp3_debug

# Full training on single GPU
export CALVIN_ZARR_PATH=data/calvin_training.zarr
python scripts/train_dp3.py training=dp3_calvin
```

### 3. Multi-GPU Training on SLURM

#### Edit SLURM script

Update `scripts/slurm_train_dp3.sh`:
- Set `CALVIN_ZARR_PATH` to your Zarr dataset path
- Adjust `#SBATCH` directives for your cluster
- Update module loads and venv path

#### Submit job

```bash
sbatch scripts/slurm_train_dp3.sh
```

#### Monitor training

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f logs/dp3_<JOB_ID>.out

# View error logs
tail -f logs/dp3_<JOB_ID>.err
```

## Configuration

### Training Configs

**`conf/training/dp3_calvin.yaml`** - Full training config
- 3000 epochs
- Batch size 128 per GPU
- EMA enabled
- WandB logging

**`conf/training/dp3_debug.yaml`** - Fast iteration
- 10 epochs
- Batch size 16
- 100 episodes max
- No WandB

### Override Parameters

```bash
# Change batch size
python scripts/train_dp3.py training=dp3_calvin training.batch_size=64

# Change number of epochs
python scripts/train_dp3.py training=dp3_calvin training.num_epochs=1000

# Disable EMA
python scripts/train_dp3.py training=dp3_calvin training.use_ema=false

# Multiple overrides
python scripts/train_dp3.py training=dp3_calvin \
    training.batch_size=32 \
    training.num_epochs=2000 \
    training.learning_rate=5e-5
```

## Distributed Training Details

### SLURM Environment

The training workspace automatically detects SLURM environment variables:
- `SLURM_PROCID` - Global rank
- `SLURM_LOCALID` - Local rank (GPU ID)
- `SLURM_NTASKS` - World size
- `SLURM_LAUNCH_NODE_IPADDR` - Master node address

### Effective Batch Size

With distributed training:
```
Effective batch size = batch_size × num_gpus × gradient_accumulation_steps
```

Example:
- 4 GPUs
- batch_size=128
- gradient_accumulation_steps=1
- **Effective batch size = 512**

### Gradient Accumulation

For larger effective batch sizes without OOM:

```bash
python scripts/train_dp3.py training=dp3_calvin \
    training.batch_size=32 \
    training.gradient_accumulation_steps=4
# Effective batch per GPU = 32 × 4 = 128
```

## Checkpointing

### Automatic Checkpointing

- Checkpoints saved every `checkpoint_every` epochs (default: 50)
- Top-K checkpoints kept based on validation loss (default: K=3)
- Latest checkpoint always saved as `latest.ckpt`

### Checkpoint Location

```
outputs/checkpoints/<experiment_name>/
├── epoch_0050.ckpt
├── epoch_0100.ckpt
├── epoch_0150.ckpt
└── latest.ckpt
```

### Resume Training

```bash
# Resume from latest checkpoint
python scripts/train_dp3.py training=dp3_calvin \
    training.resume=true

# Resume from specific checkpoint
python scripts/train_dp3.py training=dp3_calvin \
    training.resume=true \
    training.resume_checkpoint_path=outputs/checkpoints/my_run/epoch_0100.ckpt
```

## Inference with Trained Model

Use trained checkpoints with existing inference pipeline:

```bash
python scripts/run_experiment.py \
    policy.ckpt_path=outputs/checkpoints/dp3_calvin_20260213/latest.ckpt \
    num_episodes=10
```

## Monitoring

### WandB Integration

Training automatically logs to Weights & Biases:
- Training loss
- Validation loss
- Learning rate
- Epoch number

Access dashboard at: https://wandb.ai/<your_project>/langsteer_dp3

### Local Logs

```bash
# Training logs
tail -f logs/dp3_<JOB_ID>.out

# Error logs
tail -f logs/dp3_<JOB_ID>.err
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python scripts/train_dp3.py training=dp3_calvin training.batch_size=64

# Or reduce num_workers
python scripts/train_dp3.py training=dp3_calvin training.num_workers=4
```

### SLURM Job Fails to Start

Check:
1. SLURM directives match your cluster (partition, time limit, etc.)
2. Module loads are correct
3. Virtual environment path is correct
4. CALVIN_ZARR_PATH is set and accessible

### Distributed Training Hangs

Ensure all nodes can communicate:
- `MASTER_ADDR` is reachable from all nodes
- `MASTER_PORT` is not blocked by firewall
- All GPUs are available

### Import Errors

Ensure project root is in PYTHONPATH:

```bash
export PYTHONPATH=/path/to/LangSteer:$PYTHONPATH
```

## Dataset Format

### Zarr Structure

```
calvin_training.zarr/
├── data/
│   ├── img           (N, 160, 160, 3)  - RGB images
│   ├── point_cloud   (N, 2048, 6)      - XYZRGB point clouds
│   ├── depth         (N, 160, 160)     - Depth maps
│   ├── action        (N, 7)            - Relative actions
│   └── state         (N, 15)           - Robot proprioception
└── meta/
    └── episode_ends  (K,)              - Episode boundaries
```

### Data Modalities Used

Training uses:
- **Point clouds** (XYZ only, not RGB): `(T, 2048, 3)`
- **Robot state** (proprioception): `(T, 15)`
- **Actions** (relative): `(T, 7)`

Images and depth maps are stored but not used in current DP3 implementation.

## Performance Tips

### For Faster Training

1. **Increase batch size** (if GPU memory allows):
   ```bash
   training.batch_size=256
   ```

2. **Reduce validation frequency**:
   ```bash
   training.val_every=20  # Instead of 10
   ```

3. **Use gradient accumulation** for larger effective batch:
   ```bash
   training.batch_size=64 training.gradient_accumulation_steps=2
   ```

4. **Limit dataset size during development**:
   ```bash
   training.dataset.max_train_episodes=500
   ```

### For Better Results

1. **Enable EMA** (already default):
   ```bash
   training.use_ema=true
   ```

2. **Tune learning rate**:
   ```bash
   training.learning_rate=5e-5  # Try different values
   ```

3. **Increase training epochs**:
   ```bash
   training.num_epochs=5000
   ```

## Directory Structure

```
LangSteer/
├── training/              # Training infrastructure
│   ├── dp3_trainer.py    # Main training workspace
│   ├── calvin_dataset.py # Dataset loader
│   ├── replay_buffer.py  # Zarr loading
│   ├── sampler.py        # Sequence sampling
│   ├── checkpoint_util.py
│   ├── ema_model.py
│   └── pytorch_util.py
│
├── scripts/
│   ├── train_dp3.py      # Training entry point
│   ├── slurm_train_dp3.sh          # SLURM job script
│   ├── convert_calvin_to_zarr.py   # Dataset conversion
│   └── convert_calvin_debug.sh     # Conversion helper
│
├── conf/
│   └── training/
│       ├── dp3_calvin.yaml  # Full training config
│       └── dp3_debug.yaml   # Debug config
│
└── outputs/
    └── checkpoints/      # Saved checkpoints
```

## Next Steps

1. **Convert your CALVIN dataset to Zarr** (one-time)
2. **Test with debug config** to verify setup
3. **Submit SLURM job** for full training
4. **Monitor progress** via WandB or logs
5. **Evaluate trained model** using inference pipeline

## Support

For issues:
- Check logs in `logs/dp3_<JOB_ID>.{out,err}`
- Review configuration in `conf/training/`
- Ensure CALVIN dataset path is correct
- Verify SLURM environment matches your cluster

Happy training! 🚀
