#!/usr/bin/env python3
"""
Submit pi-0.5 fine-tuning on CALVIN to Amazon SageMaker (SDK v3 ModelTrainer).

Run from your laptop with AWS credentials configured:
    python training/sagemaker/submit_pi05_training.py

BEFORE RUNNING
--------------
1. pip install -r training/sagemaker/submit_requirements.txt
2. aws configure  (or use instance/Studio role)
3. Ensure the CALVIN LeRobot dataset is in S3:
       s3://{BUCKET_NAME}/{TRAIN_DATA_PREFIX}/   (aryannav/calvin_abc_d_train contents)
       s3://{BUCKET_NAME}/{VAL_DATA_PREFIX}/     (aryannav/calvin_abc_d_val contents)
4. (One-time) Download pi-0.5 base weights to S3:
       gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base - | \
           aws s3 cp - s3://{BUCKET_NAME}/{WEIGHTS_PREFIX}/ --recursive
   Or set WEIGHTS_S3_URI in hyperparameters to an existing S3 path.
5. Update the constants below (or override via env vars).
6. export WANDB_API_KEY=<your-key>
7. python training/sagemaker/submit_pi05_training.py

ENV OVERRIDES
-------------
PI05_BUCKET              S3 bucket name
PI05_INSTANCE_TYPE       SageMaker instance (default ml.p4d.24xlarge = 8×A100 40GB)
PI05_MAX_RUNTIME         Wall-clock cap in seconds (default 86400 = 24h)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
import importlib.util

_repo_root = Path(__file__).resolve().parent.parent.parent
for _rm in (str(_repo_root), str(_repo_root / "training"), ".", ""):
    while _rm in sys.path:
        sys.path.remove(_rm)

_compat = _repo_root / "utils" / "collections_compat.py"
_spec = importlib.util.spec_from_file_location("_collections_compat", _compat)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Missing collections compat shim: {_compat}")
_compat_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compat_mod)

try:
    from sagemaker.train.model_trainer import ModelTrainer
    from sagemaker.train.configs import (
        Compute,
        InputData,
        OutputDataConfig,
        SourceCode,
        StoppingCondition,
    )
    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import Session
except ImportError as exc:
    raise ImportError(
        "Install SageMaker SDK: pip install -r training/sagemaker/submit_requirements.txt"
    ) from exc


# =============================================================================
# CONFIGURATION — edit these or override via env vars listed above
# =============================================================================

BUCKET_NAME     = os.environ.get("PI05_BUCKET", "calvin-abcd-dataset-bucket")
# S3 prefixes for the two LeRobot dataset splits (contents of each directory)
TRAIN_DATA_PREFIX   = "lerobot/calvin/aryannav/calvin_abc_d_train"
VAL_DATA_PREFIX     = "lerobot/calvin/aryannav/calvin_abc_d_val"
# S3 prefix for pre-staged pi-0.5 base weights (optional — see BEFORE RUNNING above)
# Leave empty to download from GCS gs://openpi-assets/checkpoints/pi05_base at runtime.
WEIGHTS_S3_PREFIX   = "pi05_base"

SAGEMAKER_ROLE  = "arn:aws:iam::317694661330:role/SageMakerExecutionRole"
MODEL_PREFIX    = "langsteer/models/pi05_calvin"
INSTANCE_TYPE   = os.environ.get("PI05_INSTANCE_TYPE", "ml.p4d.24xlarge")  # 8×A100 40GB
MAX_RUNTIME     = int(os.environ.get("PI05_MAX_RUNTIME", 86400))

JOB_ENV_KEYS = (
    "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY",
    "WANDB_RUN_NAME", "WANDB_TAGS", "WANDB_MODE", "WANDB_DISABLED",
)

HYPERPARAMETERS = {
    "train_repo_id":    "aryannav/calvin_abc_d_train",
    "val_repo_id":      "aryannav/calvin_abc_d_val",
    "exp_name":         f"pi05-calvin-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "num_train_steps":  "50000",
    "batch_size":       "32",
    "wandb_project":    "langsteer_pi05",
    # Set to non-empty S3 URI to load weights from S3 instead of GCS.
    # e.g. "s3://calvin-abcd-dataset-bucket/pi05_base"
    "weights_s3_uri":   f"s3://{BUCKET_NAME}/{WEIGHTS_S3_PREFIX}" if WEIGHTS_S3_PREFIX else "",
}

IGNORE_PATTERNS = [
    ".git", "__pycache__", "*.pyc",
    ".env", ".venv", "venv",
    ".DS_Store", "data", "cache", "outputs",
    "*.ipynb", ".ipynb_checkpoints", "uv.lock",
]

REPO_ROOT = _repo_root


def main() -> None:
    sess = Session()
    region = sess.boto_region_name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    train_uri   = f"s3://{BUCKET_NAME}/{TRAIN_DATA_PREFIX.strip('/')}/"
    val_uri     = f"s3://{BUCKET_NAME}/{VAL_DATA_PREFIX.strip('/')}/"
    s3_output   = f"s3://{BUCKET_NAME}/{MODEL_PREFIX.strip('/')}"

    training_image = image_uris.retrieve(
        framework="pytorch", region=region,
        version="2.4.0", py_version="py311",
        instance_type=INSTANCE_TYPE, image_scope="training",
    )

    print(f"Role:            {SAGEMAKER_ROLE}")
    print(f"Region:          {region}")
    print(f"Image:           {training_image}")
    print(f"Train data (S3): {train_uri}")
    print(f"Val data (S3):   {val_uri}")
    print(f"Output (S3):     {s3_output}/")
    print(f"Instance:        {INSTANCE_TYPE}")
    print(f"Train steps:     {HYPERPARAMETERS['num_train_steps']}")

    job_env = {k: os.environ[k] for k in JOB_ENV_KEYS if os.environ.get(k, "").strip()}
    if job_env:
        print("Forwarding env:", ", ".join(sorted(job_env.keys())))
    else:
        print("WARNING: WANDB_API_KEY not set — W&B logging will be disabled.")

    source_code = SourceCode(
        source_dir=str(REPO_ROOT),
        entry_script="training/sagemaker/train_pi05_sagemaker.py",
        requirements="training/sagemaker/requirements.txt",
        ignore_patterns=IGNORE_PATTERNS,
    )

    trainer_kw: dict = {
        "training_image":     training_image,
        "source_code":        source_code,
        "compute":            Compute(instance_type=INSTANCE_TYPE, instance_count=1),
        "role":               SAGEMAKER_ROLE,
        "sagemaker_session":  sess,
        "hyperparameters":    HYPERPARAMETERS,
        "base_job_name":      f"pi05-calvin-{timestamp}",
        "output_data_config": OutputDataConfig(s3_output_path=s3_output),
        "stopping_condition": StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME),
    }
    if job_env:
        trainer_kw["environment"] = job_env

    trainer = ModelTrainer(**trainer_kw)
    print("\nSubmitting training job...")
    trainer.train(input_data_config=[
        InputData(channel_name="train", data_source=train_uri),
        InputData(channel_name="val",   data_source=val_uri),
    ])

    tj   = trainer._latest_training_job
    name = getattr(tj, "training_job_name", None) or str(tj)
    print("\n" + "=" * 72)
    print(f"Job submitted:   {name}")
    print(f"Artifacts:       {s3_output}/")
    print(f"Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/training-jobs/{name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
