"""
OpenPi TrainConfig for CALVIN dataset fine-tuning.

This file is copied into the OpenPi source tree at runtime by train_pi05_sagemaker.py
and then imported + registered into OpenPi's config registry via the wrapper scripts.

The base pi-0.5 weights are expected at /tmp/pi05_base/ (downloaded by the entry script).
HF_LEROBOT_HOME should point to a directory containing the LeRobot-format CALVIN splits.

Our CALVIN LeRobot dataset uses:
  - "image"       — static camera (200×200 RGB)
  - "wrist_image" — gripper camera (84×84 RGB)
  - "state"       — robot_obs (15-dim float32)
  - "action"      — rel_actions (7-dim float32, singular — note: Libero uses "actions")
  - "task"        — language instruction string (note: Libero uses "prompt")
"""

import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.libero_policy as libero_policy
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class CALVINDataConfig(_config.DataConfigFactory):
    """
    Data config for CALVIN LeRobot dataset.

    Differences from LeRobotLiberoDataConfig:
      - Dataset action key is "action" (singular), not "actions"
      - Dataset instruction key is "task", not "prompt"

    Reuses LiberoInputs/LiberoOutputs since CALVIN and Libero share the same
    observation structure (static + wrist cameras, robot state) and 7-DOF
    relative end-effector action format.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        # Data parquets have task_index (int64); the data_loader applies
        # PromptFromLeRobotTask which converts task_index → "prompt" (string)
        # using tasks.jsonl BEFORE the repack transform runs.  So by the time
        # repack sees the frame, the key is "prompt" (not "task").
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",  # renamed action→actions by lerobot patch
                        "prompt": "prompt",    # added by PromptFromLeRobotTask
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        model_transforms = _config.ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


PI05_CALVIN_CONFIGS: list[_config.TrainConfig] = [
    _config.TrainConfig(
        name="pi05_calvin",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=CALVINDataConfig(
            repo_id="aryannav/calvin_abc_d_train",
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/tmp/pi05_base/params",
        ),
        batch_size=32,
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=50_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        log_interval=100,
        save_interval=1_000,
        fsdp_devices=8,   # 8× A100 on ml.p4d.24xlarge; set to 1 for single-GPU runs
        seed=42,
        wandb_enabled=True,
    ),
]
