# LangSteer — AWS SageMaker Training

Trains **3D Diffuser Actor (`lang_enhanced=false`)** on CALVIN for 600K iterations using `ml.g6e.8xlarge` (4× NVIDIA L40S).

---

## Files

| File | Purpose |
|------|---------|
| `submit_langsteer_training.py` | Run locally to submit the SageMaker job |
| `train_langsteer_sagemaker.py` | Entry point executed inside the container |
| `requirements.txt` | Deps installed into the container before the entry script |
| `submit_requirements.txt` | Deps for the submit script (laptop only) |

---

## Setup

### 1. Install submitter deps (laptop only)

```bash
pip install -r training/sagemaker/submit_requirements.txt
```

### 2. Configure AWS credentials

```bash
aws configure
```

### 3. Upload the dataset to S3

Training uses packaged `.dat` files for all four CALVIN scenes (A, B, C, D).
**No `.pkl` instruction files are needed** — this is no-language training.

The S3 prefix must match `DATA_PREFIX` in `submit_langsteer_training.py` (default: `calvin`).

```bash
# Upload all four scenes
aws s3 sync cache/diffuser_actor_data/ \
    s3://calvin-abcd-dataset-bucket/calvin/
```

Expected S3 layout (verify with `aws s3 ls` before submitting):
```
calvin/
├── training/
│   ├── A+0/   *.dat  (~N files)
│   ├── B+0/   *.dat  (~N files)
│   ├── C+0/   *.dat  (~N files)
│   └── D+0/   *.dat  (303 files)
└── validation/
    ├── A+0/   *.dat  (~N files)
    ├── B+0/   *.dat  (~N files)
    ├── C+0/   *.dat  (~N files)
    └── D+0/   *.dat  (62 files)
```

**Important:** The `DATA_PREFIX` in `submit_langsteer_training.py` must match the S3 path prefix
you upload to. The default is `"calvin"`, so data goes to `s3://calvin-abcd-dataset-bucket/calvin/`.
If you used a different prefix when uploading, update `DATA_PREFIX` in the submit script to match.

Verify all four tasks are present before submitting:
```bash
for scene in A B C D; do
    echo "=== $scene ==="
    aws s3 ls s3://calvin-abcd-dataset-bucket/calvin/training/${scene}+0/ | wc -l
done
```

The training entry script will also print which task directories were found/missing at startup —
check CloudWatch logs if training loss is unexpectedly low (only D-level) or unexpectedly high
(only some scenes loaded).

### 4. Edit constants in `submit_langsteer_training.py`

```python
BUCKET_NAME    = "calvin-abcd-dataset-bucket"   # your bucket
DATA_PREFIX    = "calvin"                         # must match your S3 upload prefix
SAGEMAKER_ROLE = "arn:aws:iam::<account>:role/SageMakerExecutionRole"
```

Or override via env vars: `LANGSTEER_BUCKET`, `LANGSTEER_DATA_PREFIX`, `LANGSTEER_INSTANCE_TYPE`.

### 5. Export W&B key and submit

```bash
export WANDB_API_KEY=<your-key>
python training/sagemaker/submit_langsteer_training.py
```

The script prints the job name and a Console link, then returns immediately.

---

## Hyperparameters

Edit `HYPERPARAMETERS` in the submit script to override:

| Key | Default | Notes |
|-----|---------|-------|
| `train_iters` | `600000` | Total gradient steps |
| `batch_size` | `8` | Per GPU (×4 GPUs = global 32) |
| `lr` | `0.0003` | Peak LR; cosine schedule decays to ~3e-6 |
| `val_freq` | `5000` | Validate every N steps |
| `log_freq` | `50` | Log train loss to W&B every N steps |
| `wandb_project` | `langsteer_diffuser_actor` | W&B project |
| `run_name` | `sm-nolang-YYYYMMDDHHMMSS` | W&B run name |

Training uses a **cosine warmup schedule** (2000-step linear warmup then cosine decay to
`lr × 0.01`). Gradient norm is clipped at `1.0` each step.

---

## What Happens Inside the Container

1. SageMaker installs `requirements.txt` automatically.
2. Entry script runs:
   - `pip install -e . --no-deps` — registers the `langsteer` package.
   - Installs `flash-attn` with `--no-build-isolation` (non-fatal if it fails).
   - Upgrades `networkx>=2.6` (DGL/urdfpy compat).
   - Sets `CALVIN_3DA_DATASET_PATH` to the mounted S3 channel path.
   - **Prints which A/B/C/D task directories were found** — check logs if behavior is wrong.
3. `torchrun --nproc_per_node=<N_GPUS> scripts/train_diffuser_actor.py` is launched with Hydra overrides:
   - `training=diffuser_actor_nolang` (`lang_enhanced=false`, `use_instruction=false`, all 4 tasks)
   - `train_iters=600000`, plus all other hyperparameters from step 2
   - `training.checkpoint_dir=/opt/ml/model/checkpoints` (absolute — avoids Hydra CWD issues)
4. Checkpoints land in `/opt/ml/model/checkpoints/` and are auto-packaged into `model.tar.gz`.

---

## Retrieving Checkpoints

```bash
JOB=langsteer-nolang-<timestamp>
aws s3 cp \
    s3://calvin-abcd-dataset-bucket/langsteer/models/diffuser_actor_nolang/$JOB/output/model.tar.gz \
    ./model.tar.gz
tar -xzf model.tar.gz
ls checkpoints/   # last.pth  best.pth  NNNNNNN.pth
```

---

## Env Var Reference

| Var | Default | Override |
|-----|---------|---------|
| `LANGSTEER_BUCKET` | `calvin-abcd-dataset-bucket` | S3 bucket |
| `LANGSTEER_DATA_PREFIX` | `calvin` | S3 prefix — **must match upload path** |
| `LANGSTEER_SAGEMAKER_ROLE` | see script | IAM execution role ARN |
| `LANGSTEER_INSTANCE_TYPE` | `ml.g6e.8xlarge` | Instance type |
| `LANGSTEER_TRAINING_IMAGE` | _(DLC auto-resolve)_ | Full ECR URI |
| `LANGSTEER_MAX_RUNTIME` | `86400` | Wall-clock cap (seconds) |
| `WANDB_API_KEY` | — | Required for W&B logging |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ResourceLimitExceeded` | AWS Console → Service Quotas → SageMaker → request `ml.g6e.8xlarge` quota |
| `FileNotFoundError: training/ and validation/` | Verify S3 layout; check that `DATA_PREFIX` in submit script matches the S3 upload path |
| Loss curve stable ~4-5 then sudden jump to ~15 | Only some task scenes loaded — check CloudWatch logs for `[data]` lines showing which tasks were found; verify all 4 scenes are on S3 under the correct prefix |
| DLC image not found for `2.4.0`/`py311` | Set `LANGSTEER_TRAINING_IMAGE` to a known ECR URI |
| OOM | Reduce `batch_size` from 8 to 4 in `HYPERPARAMETERS` |
