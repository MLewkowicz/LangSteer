# LangSteer — AWS SageMaker Training

Three training configurations are available. Jump to the relevant section:

- [No-language (baseline)](#no-language-baseline) — `diffuser_actor_nolang`, 600K iters
- [Primitive-only conditioning](#primitive-only-conditioning) — `diffuser_actor_primitive`, 200K iters
- [Primitive + Object conditioning](#primitive--object-conditioning) — `diffuser_actor_primitive_object`, 200K iters ← **new**

---

## Files

| File | Purpose |
|------|---------|
| `submit_langsteer_training.py` | Submit no-language job |
| `submit_primitive_training.py` | Submit primitive-only job |
| `submit_primitive_object_training.py` | Submit primitive+object job |
| `train_langsteer_sagemaker.py` | Container entry point (no-language) |
| `train_primitive_sagemaker.py` | Container entry point (primitive-only) |
| `train_primitive_object_sagemaker.py` | Container entry point (primitive+object) |
| `requirements.txt` | Deps installed into the container |
| `submit_requirements.txt` | Deps for the submit scripts (laptop only) |

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
| `FileNotFoundError: training/ and validation/` | Verify S3 layout; check that the data prefix in the submit script matches the S3 upload path |
| Loss curve stable ~4-5 then sudden jump to ~15 | Only some task scenes loaded — check CloudWatch logs for `[data]` lines showing which tasks were found; verify all 4 scenes are on S3 under the correct prefix |
| DLC image not found for `2.4.0`/`py311` | Set `LANGSTEER_TRAINING_IMAGE` to a known ECR URI |
| OOM | Reduce `batch_size` from 8 to 4 in `HYPERPARAMETERS` |

---

## No-language (baseline)

Trains **3D Diffuser Actor (`lang_enhanced=false`)** on CALVIN for 600K iterations using `ml.g6e.8xlarge` (4× NVIDIA L40S).

### Setup

#### 1. Install submitter deps (laptop only)

```bash
pip install -r training/sagemaker/submit_requirements.txt
```

#### 2. Configure AWS credentials

```bash
aws configure
```

#### 3. Upload dataset to S3

```bash
aws s3 sync cache/diffuser_actor_data/ s3://calvin-abcd-dataset-bucket/calvin/
```

Expected layout:
```
calvin/
├── training/  {A,B,C,D}+0/*.dat
└── validation/ {A,B,C,D}+0/*.dat
```

#### 4. Edit constants and submit

```python
# submit_langsteer_training.py
BUCKET_NAME    = "calvin-abcd-dataset-bucket"
DATA_PREFIX    = "calvin"
SAGEMAKER_ROLE = "arn:aws:iam::<account>:role/SageMakerExecutionRole"
```

```bash
export WANDB_API_KEY=<your-key>
python training/sagemaker/submit_langsteer_training.py
```

---

## Primitive-only conditioning

Trains with a learned embedding for each action primitive (`grasp`, `push`, `pull`, `place`, `rotate`).

### Setup

Same AWS credentials and submitter deps as above.

Data must already be packaged with `primitive_lang_ann.npy` annotations.

```bash
export WANDB_API_KEY=<your-key>
python training/sagemaker/submit_primitive_training.py
```

### Hyperparameters

| Key | Default | Notes |
|-----|---------|-------|
| `train_iters` | `200000` | |
| `batch_size` | `8` | Per GPU |
| `num_primitives` | `5` | grasp=0 push=1 pull=2 place=3 rotate=4 |

---

## Primitive + Object conditioning

Trains with two learned embeddings: one for the action primitive (5 classes) and one for the object being manipulated (8 classes). Both are looked up and concatenated to a `(B, 2, D)` conditioning sequence fed into the cross-attention stack.

**The data is already on S3** at `s3://calvin-abcd-dataset-bucket/calvin_object_action/` — no upload needed.

### Setup

#### 1. Install submitter deps (laptop only)

```bash
pip install -r training/sagemaker/submit_requirements.txt
```

#### 2. Configure AWS credentials

```bash
aws configure
```

#### 3. Verify data is present

```bash
aws s3 ls s3://calvin-abcd-dataset-bucket/calvin_object_action/packaged_ABC_D/training/lang_annotations/
# Should show: primitive_object_lang_ann.npy

for scene in A B C D; do
    echo "=== training/$scene+0 ==="
    aws s3 ls s3://calvin-abcd-dataset-bucket/calvin_object_action/packaged_ABC_D/training/${scene}+0/ | wc -l
done
```

Expected S3 layout:
```
calvin_object_action/
└── packaged_ABC_D/
    ├── training/
    │   ├── A+0/  *.dat
    │   ├── B+0/  *.dat
    │   ├── C+0/  *.dat
    │   ├── D+0/  *.dat
    │   └── lang_annotations/
    │       └── primitive_object_lang_ann.npy
    └── validation/
        ├── A+0/  *.dat  (etc.)
        └── lang_annotations/
            └── primitive_object_lang_ann.npy
```

#### 4. Set your SageMaker role

Edit `SAGEMAKER_ROLE` in `submit_primitive_object_training.py`, or:

```bash
export LANGSTEER_SAGEMAKER_ROLE=arn:aws:iam::<account>:role/SageMakerExecutionRole
```

#### 5. Submit

```bash
export WANDB_API_KEY=<your-key>
python training/sagemaker/submit_primitive_object_training.py
```

The script prints the job name and a direct Console link, then returns immediately.

### Hyperparameters

Edit `HYPERPARAMETERS` in `submit_primitive_object_training.py` to override:

| Key | Default | Notes |
|-----|---------|-------|
| `train_iters` | `200000` | Total gradient steps |
| `batch_size` | `8` | Per GPU (×4 on ml.g6e.8xlarge = global 32) |
| `batch_size_val` | `2` | |
| `lr` | `0.0003` | Peak LR; cosine schedule decays to ~3e-6 |
| `wd` | `0.005` | AdamW weight decay |
| `val_freq` | `5000` | Validate every N steps |
| `log_freq` | `50` | Log train loss to W&B every N steps |
| `num_primitives` | `5` | grasp=0 push=1 pull=2 place=3 rotate=4 |
| `num_objects` | `8` | block=0 blue_block=1 drawer_handle=2 led_button=3 lightbulb_switch=4 pink_block=5 red_block=6 slider_handle=7 |
| `wandb_project` | `langsteer_diffuser_actor` | W&B project |
| `run_name` | `sm-primitive-object-TIMESTAMP` | W&B run name |

### What happens inside the container

1. SageMaker mounts the S3 channel at `SM_CHANNEL_PRIMITIVE_OBJECT`.
2. `train_primitive_object_sagemaker.py` runs:
   - Installs the `langsteer` package (`pip install -e . --no-deps`).
   - Installs `flash-attn` (non-fatal if it fails).
   - Upgrades `networkx>=2.6` (DGL/urdfpy compat).
   - Locates `packaged_ABC_D/` inside the channel for `.dat` files and for `primitive_object_lang_ann.npy`.
   - Sets `CALVIN_3DA_PRIMITIVE_OBJ_DATASET_PATH` and `CALVIN_PRIMITIVE_OBJ_ANN_PATH`.
   - Prints which A/B/C/D task directories were found — check CloudWatch if loss looks wrong.
3. `torchrun` launches `scripts/train_diffuser_actor.py` with:
   - `training=diffuser_actor_primitive_object`
   - `training.policy.use_primitive_id=true`, `use_object_id=true`
   - All hyperparameters from step 2.
   - `training.checkpoint_dir=/opt/ml/model/checkpoints` (absolute path).
4. Checkpoints land in `/opt/ml/model/checkpoints/` and are auto-packaged into `model.tar.gz`.

### Retrieving checkpoints

```bash
JOB=langsteer-prim-obj-<timestamp>
aws s3 cp \
    s3://calvin-abcd-dataset-bucket/langsteer/models/diffuser_actor_primitive_object/$JOB/output/model.tar.gz \
    ./model.tar.gz
tar -xzf model.tar.gz
ls checkpoints/   # last.pth  best.pth  NNNNNNN.pth
```

### Env var reference

| Var | Default | Override |
|-----|---------|---------|
| `LANGSTEER_BUCKET` | `calvin-abcd-dataset-bucket` | S3 bucket |
| `LANGSTEER_PRIMITIVE_OBJ_PREFIX` | `calvin_object_action` | S3 data prefix |
| `LANGSTEER_SAGEMAKER_ROLE` | see script | IAM execution role ARN |
| `LANGSTEER_INSTANCE_TYPE` | `ml.g6e.8xlarge` | Instance type |
| `LANGSTEER_TRAINING_IMAGE` | _(DLC auto-resolve)_ | Full ECR URI override |
| `LANGSTEER_MAX_RUNTIME` | `86400` | Wall-clock cap (seconds) |
| `WANDB_API_KEY` | — | Required for W&B logging |
