#!/usr/bin/env python3
"""
Submit LangSteer 3D Diffuser Actor (primitive-ID conditioning) training to SageMaker.

Run from your laptop with AWS credentials configured:
    python training/sagemaker/submit_primitive_training.py

BEFORE RUNNING
--------------
1. pip install -r training/sagemaker/submit_requirements.txt
2. aws configure  (or use instance/Studio role)
3. Upload the primitive dataset to S3. The bucket prefix must contain:
       training/{A,B,C,D}+0/*.dat
       training/lang_annotations/primitive_lang_ann.npy
       validation/D+0/*.dat
       validation/lang_annotations/primitive_lang_ann.npy

   Example sync (run after preprocessing + packaging):
       aws s3 sync /path/to/packaged_primitive/ s3://{BUCKET_NAME}/{PRIMITIVE_DATA_PREFIX}/
       aws s3 cp /path/to/training/lang_annotations/primitive_lang_ann.npy \
           s3://{BUCKET_NAME}/{PRIMITIVE_DATA_PREFIX}/training/lang_annotations/primitive_lang_ann.npy
       aws s3 cp /path/to/validation/lang_annotations/primitive_lang_ann.npy \
           s3://{BUCKET_NAME}/{PRIMITIVE_DATA_PREFIX}/validation/lang_annotations/primitive_lang_ann.npy

4. Update BUCKET_NAME, PRIMITIVE_DATA_PREFIX, SAGEMAKER_ROLE below (or set env vars).
5. export WANDB_API_KEY=<your-key>
6. python training/sagemaker/submit_primitive_training.py

ENV OVERRIDES
-------------
LANGSTEER_BUCKET               S3 bucket name
LANGSTEER_PRIMITIVE_PREFIX     S3 prefix for the primitive dataset
LANGSTEER_INSTANCE_TYPE        SageMaker instance (default ml.g6e.8xlarge = 4×L40S)
LANGSTEER_TRAINING_IMAGE       Full ECR URI override (skip DLC auto-resolve)
LANGSTEER_MAX_RUNTIME          Wall-clock cap in seconds (default 86400 = 24h)
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

BUCKET_NAME          = os.environ.get("LANGSTEER_BUCKET", "calvin-abcd-dataset-bucket")
PRIMITIVE_DATA_PREFIX = os.environ.get("LANGSTEER_PRIMITIVE_PREFIX", "calvin_")
SAGEMAKER_ROLE       = "arn:aws:iam::317694661330:role/SageMakerExecutionRole"
MODEL_PREFIX         = "langsteer/models/diffuser_actor_primitive"
INSTANCE_TYPE        = os.environ.get("LANGSTEER_INSTANCE_TYPE", "ml.g6e.8xlarge")
TRAINING_IMAGE       = os.environ.get("LANGSTEER_TRAINING_IMAGE", "")
MAX_RUNTIME          = int(os.environ.get("LANGSTEER_MAX_RUNTIME", 86400))

JOB_ENV_KEYS = (
    "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY",
    "WANDB_RUN_NAME", "WANDB_TAGS", "WANDB_MODE", "WANDB_DISABLED",
)

# =============================================================================
# Hyperparameters — forwarded to train_primitive_sagemaker.py as strings
# =============================================================================

HYPERPARAMETERS = {
    "train_iters":    "600000",
    "batch_size":     "8",       # per GPU
    "batch_size_val": "2",
    "lr":             "0.0003",
    "wd":             "0.005",
    "val_freq":       "5000",
    "log_freq":       "50",
    "num_workers":    "4",
    "num_primitives": "4",
    "wandb_project":  "langsteer_diffuser_actor",
    "run_name":       f"sm-primitive-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
}

REPO_ROOT = _repo_root

IGNORE_PATTERNS = [
    ".git", "__pycache__", "*.pyc",
    ".env", ".venv", "venv",
    ".DS_Store", "data", "cache", "outputs",
    "*.ipynb", ".ipynb_checkpoints", "uv.lock",
]


def main() -> None:
    sess = Session()
    region = sess.boto_region_name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    primitive_uri = f"s3://{BUCKET_NAME}/{PRIMITIVE_DATA_PREFIX}".rstrip("/") + "/"
    s3_output     = f"s3://{BUCKET_NAME}/{MODEL_PREFIX.strip('/')}"

    if TRAINING_IMAGE.strip():
        training_image = TRAINING_IMAGE.strip()
    else:
        training_image = image_uris.retrieve(
            framework="pytorch", region=region,
            version="2.4.0", py_version="py311",
            instance_type=INSTANCE_TYPE, image_scope="training",
        )

    print(f"Role:            {SAGEMAKER_ROLE}")
    print(f"Region:          {region}")
    print(f"Image:           {training_image}")
    print(f"Primitive data:  {primitive_uri}")
    print(f"Output (S3):     {s3_output}/")
    print(f"Instance:        {INSTANCE_TYPE}")
    print(f"Train iters:     {HYPERPARAMETERS['train_iters']}")

    job_env = {k: os.environ[k] for k in JOB_ENV_KEYS if os.environ.get(k, "").strip()}
    if job_env:
        print("Forwarding env:", ", ".join(sorted(job_env.keys())))
    else:
        print("WARNING: WANDB_API_KEY not set — W&B logging will be disabled.")

    source_code = SourceCode(
        source_dir=str(REPO_ROOT),
        entry_script="training/sagemaker/train_primitive_sagemaker.py",
        requirements="training/sagemaker/requirements.txt",
        ignore_patterns=IGNORE_PATTERNS,
    )

    trainer_kw: dict = {
        "training_image":    training_image,
        "source_code":       source_code,
        "compute":           Compute(instance_type=INSTANCE_TYPE, instance_count=1),
        "role":              SAGEMAKER_ROLE,
        "sagemaker_session": sess,
        "hyperparameters":   HYPERPARAMETERS,
        "base_job_name":     f"langsteer-primitive-{timestamp}",
        "output_data_config": OutputDataConfig(s3_output_path=s3_output),
        "stopping_condition": StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME),
    }
    if job_env:
        trainer_kw["environment"] = job_env

    trainer = ModelTrainer(**trainer_kw)
    print("\nSubmitting training job...")
    trainer.train(input_data_config=[
        InputData(channel_name="primitive", data_source=primitive_uri),
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
