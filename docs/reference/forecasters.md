# Forecasters Reference

Documentation for trajectory forecaster models in LangSteer.

## Overview

Forecasters predict clean (denoised) trajectories from noisy diffusion samples. They enable steering methods like Tweedie guidance by providing predictions of the final clean trajectory at each diffusion timestep.

## Available Forecasters

### 1. TrajectoryForecaster (Neural Network)

Trainable MLP-based forecaster that learns to predict clean trajectories.

**Architecture:**
- Input: Noisy trajectory + timestep embedding
- Output: Predicted clean trajectory
- Trained on diffusion process samples

**Location:** `forecasters/trajectory_forecaster.py`

**Configuration:**
```yaml
# conf/forecaster/base/trajectory_mlp.yaml
name: trajectory_forecaster
arch: mlp
hidden_dim: 512
num_layers: 4
dropout: 0.1
activation: relu
use_timestep_embedding: true
```

**Training:**
```bash
# See docs/guides/training.md for forecaster training
python -m training.forecasters.trajectory.trainer
```

### 2. TweedieForecaster (Analytical)

Score-based forecaster using Tweedie's formula (no training required).

**Advantages:**
- No training needed
- Theoretically grounded
- Uses DP3's trained noise prediction directly

**Limitations:**
- Assumes Gaussian noise
- May be less accurate than trained forecaster

**Location:** `forecasters/tweedie_forecaster.py`

**Usage:**
```python
from forecasters.tweedie_forecaster import TweedieForecaster

forecaster = TweedieForecaster(
    scheduler=policy.scheduler,
    prediction_type='epsilon'  # or 'sample' or 'v_prediction'
)

clean_pred = forecaster.forecast(noisy_traj, timestep)
```

## Forecaster Training

### Dataset Preparation

Forecasters are trained on the same CALVIN dataset as DP3:

```bash
# Convert CALVIN to Zarr format (if not already done)
python -m training.policies.dp3.preprocessing.convert_calvin \
    --input_dir /path/to/calvin/dataset/task_D_D/training \
    --output_dir /path/to/output.zarr
```

### Training Configuration

**Production config:** `conf/training/forecasters/trajectory/calvin.yaml`
```yaml
# Training hyperparameters
num_epochs: 500
batch_size: 256
learning_rate: 1e-4
weight_decay: 1e-5

# Dataset
zarr_path: ${oc.env:CALVIN_ZARR_PATH}
horizon: 16
obs_horizon: 2

# Model architecture (references conf/forecaster/base/)
model: trajectory_mlp

# Noise schedule (must match DP3)
num_diffusion_iters: 100
beta_schedule: squaredcos_cap_v2
```

**Debug config:** `conf/training/forecasters/trajectory/debug.yaml`
```yaml
num_epochs: 10
batch_size: 16
# ... smaller settings for quick testing
```

### Training Commands

```bash
# Full training
python -m training.forecasters.trajectory.trainer \
    training=training/forecasters/trajectory/calvin

# Debug mode
python -m training.forecasters.trajectory.trainer \
    training=training/forecasters/trajectory/debug

# Override parameters
python -m training.forecasters.trajectory.trainer \
    training=training/forecasters/trajectory/calvin \
    training.batch_size=128 \
    training.num_epochs=1000
```

### Multi-GPU Training

```bash
# DDP training (4 GPUs)
torchrun --nproc_per_node=4 -m training.forecasters.trajectory.trainer \
    training=training/forecasters/trajectory/calvin
```

## Integration with Steering

Forecasters are used by steering methods to predict trajectories:

```python
# In steering module
class TweedieSteering(BaseSteering):
    def __init__(self, config):
        self.forecaster = TweedieForecaster(...)
        self.reference_trajectory = None

    def guide(self, noisy_actions, timestep):
        # Forecast clean trajectory
        clean_pred = self.forecaster.forecast(noisy_actions, timestep)

        # Compute guidance toward reference
        guidance = self.reference_trajectory - clean_pred
        return guidance
```

**Usage:**
```bash
# Run with Tweedie steering (uses TweedieForecaster automatically)
python scripts/run_experiment.py \
    steering=tweedie \
    env.task=open_drawer
```

## Forecaster API

### Base Interface

All forecasters implement `BaseForecaster`:

```python
class BaseForecaster:
    def forecast(
        self,
        noisy_trajectory: torch.Tensor,  # (B, H, 7)
        timestep: int
    ) -> torch.Tensor:  # (B, H, 7)
        """Predict clean trajectory from noisy sample."""
        raise NotImplementedError
```

### TrajectoryForecaster Specific

```python
from forecasters.trajectory_forecaster import TrajectoryForecaster

# Initialize
forecaster = TrajectoryForecaster(
    obs_dim=18,      # Point cloud + proprio
    action_dim=7,
    hidden_dim=512,
    num_layers=4
)

# Load checkpoint
forecaster.load_checkpoint('path/to/checkpoint.pt')

# Inference
clean_trajectory = forecaster.forecast(noisy_trajectory, timestep)
```

## Checkpoints

Forecaster checkpoints are saved during training:

```
outputs/
└── forecaster_training/
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── checkpoints/
            │   ├── latest.pt
            │   ├── best.pt
            │   └── epoch_100.pt
            └── logs/
                └── metrics.csv
```

**Loading a checkpoint:**
```python
forecaster = TrajectoryForecaster(...)
forecaster.load_checkpoint('outputs/.../checkpoints/best.pt')
```

## Performance Metrics

Key metrics tracked during training:
- **Prediction MSE**: L2 distance between predicted and true clean trajectories
- **Position MSE**: MSE on position components (xyz)
- **Orientation MSE**: MSE on orientation components (rpy)
- **Validation loss**: Held-out set performance

**Typical values:**
- Position MSE: 0.001 - 0.01 (trained model)
- Total MSE: 0.01 - 0.05 (trained model)

## Troubleshooting

**Training loss not decreasing:**
- Check learning rate (try 1e-4 to 1e-3)
- Verify noise schedule matches DP3
- Ensure dataset is normalized properly

**NaN losses:**
- Reduce learning rate
- Add gradient clipping
- Check for corrupted data in dataset

**Forecaster predictions too noisy:**
- Train for more epochs
- Increase model capacity (hidden_dim, num_layers)
- Check timestep embedding is enabled

## Future Work

Potential improvements:
1. **Conditional forecasting**: Condition on observation/language
2. **Ensemble forecasters**: Average predictions from multiple models
3. **Uncertainty estimation**: Output confidence intervals
4. **Multi-step forecasting**: Predict multiple future timesteps

## See Also

- [Training Guide](../guides/training.md) - General training documentation
- [Steering Documentation](../../steering/README.md) - How forecasters are used
- [DP3 Policy](../../policies/README.md) - Base diffusion policy
