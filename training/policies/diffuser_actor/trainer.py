"""DiffuserActor Training Workspace with SLURM/DDP Support.

Iteration-based training for 3D Diffuser Actor on CALVIN.
Adapted from 3d_diffuser_actor/engine.py + main_trajectory_calvin.py.
"""

import io
import math
import os
import logging
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
from typing import Optional

from policies.diffuser_actor_components import DiffuserActor
from training.policies.diffuser_actor.dataset import CalvinDataset, traj_collate_fn
from training.common.checkpoint_util import TopKCheckpointManager
from training.common.ema_model import EMAModel

logger = logging.getLogger(__name__)


class DiffuserActorTrainingWorkspace:
    """
    Iteration-based training workspace for DiffuserActor on CALVIN.

    Key features:
    - Iteration-based (200K steps)
    - Loss computed inside model.forward() (returns scalar)
    - Separate weight decay groups (no decay for bias/LayerNorm)
    - DDP with find_unused_parameters=True (frozen backbone)
    - Validation runs inference and computes trajectory metrics
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Setup distributed training
        self._setup_distributed()
        self._set_seed(cfg.seed)

        # Device
        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(cfg.device)

        # Build model
        self.model = self._build_model(cfg)
        self.model.to(self.device)

        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,  # frozen backbone
            )

        # EMA model
        self.ema_model = None
        if cfg.get("use_ema", False):
            self.ema_model = EMAModel(
                model=self.model.module if self.is_distributed else self.model,
                power=cfg.get("ema_power", 0.75),
                update_after_step=cfg.get("ema_update_after_step", 0),
                inv_gamma=cfg.get("ema_inv_gamma", 1.0),
                max_value=cfg.get("ema_max_value", 0.9999),
            )

        # Optimizer with separate weight decay groups
        self.optimizer = self._build_optimizer(
            self.model.module if self.is_distributed else self.model, cfg
        )

        # Cosine LR schedule with linear warmup
        self.scheduler = self._build_scheduler(self.optimizer, cfg)

        # Training state
        self.global_step = 0
        self.best_loss = None
        self.use_instruction = cfg.policy.get("use_instruction", True)
        self.use_primitive_id = cfg.policy.get("use_primitive_id", False)
        if self.use_primitive_id and not self.use_instruction:
            raise ValueError(
                "use_primitive_id=True requires use_instruction=True "
                "(primitive mode reuses the instruction cross-attention pipeline)."
            )

        # Logging
        self.use_wandb = cfg.get("use_wandb", True) and self.is_main_process
        if self.use_wandb:
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "langsteer_diffuser_actor"),
                name=cfg.get("experiment_name", "diffuser_actor_calvin"),
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
            )

        if self.is_main_process:
            logger.info("DiffuserActorTrainingWorkspace initialized")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_distributed(self):
        """Setup distributed training (SLURM-aware)."""
        if "SLURM_PROCID" in os.environ:
            self.rank = int(os.environ["SLURM_PROCID"])
            self.local_rank = int(os.environ["SLURM_LOCALID"])
            self.world_size = int(os.environ["SLURM_NTASKS"])
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = os.environ.get(
                    "SLURM_LAUNCH_NODE_IPADDR", "localhost"
                )
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
            self.is_distributed = True
            dist.init_process_group(
                backend="nccl", init_method="env://",
                world_size=self.world_size, rank=self.rank,
            )
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.is_distributed = True
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.is_distributed = False

        self.is_main_process = self.rank == 0

        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    def _set_seed(self, seed: int):
        seed = seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _build_model(self, cfg: DictConfig) -> DiffuserActor:
        """Build DiffuserActor model from config."""
        policy_cfg = cfg.policy
        gripper_loc_bounds = policy_cfg.get(
            "gripper_loc_bounds", [[-2, -2, -2], [2, 2, 2]]
        )
        if isinstance(gripper_loc_bounds, str):
            from training.policies.diffuser_actor.preprocessing.calvin_utils import (
                get_gripper_loc_bounds,
            )
            gripper_loc_bounds = get_gripper_loc_bounds(gripper_loc_bounds)

        model = DiffuserActor(
            backbone=policy_cfg.get("backbone", "clip"),
            image_size=tuple(policy_cfg.get("image_size", [256, 256])),
            embedding_dim=policy_cfg.get("embedding_dim", 120),
            num_vis_ins_attn_layers=policy_cfg.get("num_vis_ins_attn_layers", 2),
            use_instruction=policy_cfg.get("use_instruction", True),
            fps_subsampling_factor=policy_cfg.get("fps_subsampling_factor", 5),
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=policy_cfg.get("rotation_parametrization", "6D"),
            quaternion_format=policy_cfg.get("quaternion_format", "xyzw"),
            diffusion_timesteps=policy_cfg.get("diffusion_timesteps", 100),
            nhist=policy_cfg.get("nhist", 3),
            relative=policy_cfg.get("relative", True),
            lang_enhanced=policy_cfg.get("lang_enhanced", True),
            use_primitive_id=policy_cfg.get("use_primitive_id", False),
            num_primitives=policy_cfg.get("num_primitives", 4),
        )

        if self.is_main_process:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {n_params:,}")

        return model

    def _build_optimizer(self, model: nn.Module, cfg: DictConfig):
        """Build AdamW optimizer with separate weight decay groups."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": cfg.lr},
            {"params": [], "weight_decay": cfg.get("wd", 5e-3), "lr": cfg.lr},
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        return optim.AdamW(optimizer_grouped_parameters)

    def _build_scheduler(self, optimizer: optim.Optimizer, cfg: DictConfig):
        """Cosine LR schedule with linear warmup.

        Warms up linearly from 0 → lr over lr_warmup_steps steps, then decays
        via cosine to lr * 0.01 by the end of training.
        """
        warmup_steps = cfg.get("lr_warmup_steps", 2000)
        total_steps = cfg.train_iters
        min_lr_ratio = 0.01  # final lr = lr * min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine_val * (1.0 - min_lr_ratio) + min_lr_ratio

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get_datasets(self, cfg: DictConfig):
        """Initialize train and validation datasets."""
        dataset_cfg = cfg.dataset

        # CLIP-mode instruction embeddings (.pkl with precomputed features).
        # Skipped entirely in primitive-id mode (model doesn't need them) and
        # in no-language mode.
        if self.use_instruction and not self.use_primitive_id:
            train_instructions = self._load_instructions(
                dataset_cfg.instructions_path, "training"
            )
            val_instructions = self._load_instructions(
                dataset_cfg.instructions_path, "validation"
            )
        else:
            train_instructions = None
            val_instructions = None

        # Primitive-ID mode: build a flat {annotation_id -> int} array from the
        # primitive_lang_ann.npy annotations' `task` field (which holds primitive
        # names like "grasp"/"push"/"pull"/"place" after preprocessing).
        train_primitive_ids = None
        val_primitive_ids = None
        if self.use_primitive_id:
            train_primitive_ids = self._load_primitive_ids(
                dataset_cfg.get("primitive_ann_path_train"), self.cfg.policy
            )
            val_primitive_ids = self._load_primitive_ids(
                dataset_cfg.get("primitive_ann_path_val"), self.cfg.policy
            )

        # Load from config if specified, otherwise default to all 4 CALVIN scenes
        taskvar_cfg = dataset_cfg.get("taskvar", None)
        if taskvar_cfg is not None:
            taskvar = [tuple(tv) for tv in taskvar_cfg]
        else:
            taskvar = [("A", 0), ("B", 0), ("C", 0), ("D", 0)]

        image_rescale = tuple(
            float(x) for x in str(dataset_cfg.get("image_rescale", "0.75,1.25")).split(",")
        )

        train_dataset = CalvinDataset(
            root=dataset_cfg.train_path,
            instructions=train_instructions,
            primitive_ids=train_primitive_ids,
            taskvar=taskvar,
            max_episode_length=dataset_cfg.get("max_episode_length", 5),
            cache_size=dataset_cfg.get("cache_size", 100),
            max_episodes_per_task=dataset_cfg.get("max_episodes_per_task", -1),
            num_iters=cfg.train_iters,
            cameras=("front", "wrist"),
            training=True,
            image_rescale=image_rescale,
            return_low_lvl_trajectory=True,
            dense_interpolation=dataset_cfg.get("dense_interpolation", True),
            interpolation_length=dataset_cfg.get("interpolation_length", 100),
            relative_action=dataset_cfg.get("relative_action", True),
        )
        val_dataset = CalvinDataset(
            root=dataset_cfg.val_path,
            instructions=val_instructions,
            primitive_ids=val_primitive_ids,
            taskvar=taskvar,
            max_episode_length=dataset_cfg.get("max_episode_length", 5),
            cache_size=dataset_cfg.get("cache_size_val", 100),
            max_episodes_per_task=dataset_cfg.get("max_episodes_per_task", -1),
            cameras=("front", "wrist"),
            training=False,
            image_rescale=image_rescale,
            return_low_lvl_trajectory=True,
            dense_interpolation=dataset_cfg.get("dense_interpolation", True),
            interpolation_length=dataset_cfg.get("interpolation_length", 100),
            relative_action=dataset_cfg.get("relative_action", True),
        )
        return train_dataset, val_dataset

    def _build_instruction(self, sample):
        """Return the conditioning tensor matching the active mode.

        - primitive-id mode: long tensor of shape (B, 1) with ids in [0, K)
        - CLIP mode:         float tensor of shape (B, 53, 512) CLIP features
        - no-language mode:  None
        """
        if self.use_primitive_id:
            return sample["primitive_id"].to(self.device).long()
        if self.use_instruction:
            return sample["instr"].to(self.device)
        return None

    @staticmethod
    def _load_primitive_ids(ann_path, policy_cfg):
        """Load a primitive_lang_ann.npy and map its `task` primitives -> int ids.

        Returns a 1D numpy int array of length N (number of annotations),
        indexed by `annotation_id` (what `episode[6][0]` carries).
        """
        if ann_path is None:
            raise ValueError(
                "use_primitive_id=True but dataset.primitive_ann_path_{train,val} "
                "is not set. Point to the primitive_lang_ann.npy produced by "
                "scripts/preprocess_primitive_annotations.py."
            )
        ann = np.load(ann_path, allow_pickle=True).item()
        tasks = list(ann["language"]["task"])
        # Fixed vocabulary for LangSteer's 4-primitive scheme.
        vocab = {"grasp": 0, "push": 1, "pull": 2, "place": 3}
        num_primitives = policy_cfg.get("num_primitives", 4)
        if len(vocab) != num_primitives:
            raise ValueError(
                f"num_primitives={num_primitives} but vocabulary has {len(vocab)}. "
                "Adjust both if extending the primitive set."
            )
        ids = np.array([vocab[str(t)] for t in tasks], dtype=np.int64)
        logger.info(f"Loaded {len(ids)} primitive ids from {ann_path}: "
                    f"{dict((k, int((ids == v).sum())) for k, v in vocab.items())}")
        return ids

    @staticmethod
    def _load_instructions(instructions_path, split):
        """Load precomputed CLIP instruction embeddings."""
        path = f"{instructions_path}/{split}.pkl"
        with open(path, "rb") as f:
            instructions = pickle.load(f)["embeddings"]
        return instructions

    def _get_loaders(self, train_dataset, val_dataset, cfg):
        """Build data loaders with distributed samplers.

        Returns (train_loader, val_loader, train_sampler).  The sampler is
        returned so the training loop can call set_epoch() on each restart,
        which re-shuffles the data ordering and avoids the sampler being
        frozen at epoch 0 for the entire 600K-step run.
        """
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        train_sampler = DistributedSampler(train_dataset) if self.is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=True) if self.is_distributed else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=(train_sampler is None),
            num_workers=cfg.get("num_workers", 1),
            worker_init_fn=seed_worker,
            collate_fn=traj_collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get("batch_size_val", 4),
            shuffle=(val_sampler is None),
            num_workers=0,
            worker_init_fn=seed_worker,
            collate_fn=traj_collate_fn,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False,
            generator=g,
        )
        return train_loader, val_loader, train_sampler

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Main iteration-based training loop."""
        cfg = self.cfg

        # Setup datasets and loaders
        train_dataset, val_dataset = self._get_datasets(cfg)
        train_loader, val_loader, train_sampler = self._get_loaders(train_dataset, val_dataset, cfg)

        # Checkpoint manager
        topk_manager = None
        if self.is_main_process:
            checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints/diffuser_actor"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            topk_manager = TopKCheckpointManager(
                save_dir=checkpoint_dir,
                monitor_key="val_loss",
                mode="min",
                k=cfg.get("save_top_k", 3),
            )

        # Resume from checkpoint — uses resume_checkpoint_path from config.
        # Starting from scratch unless explicitly set; no stale checkpoint is
        # ever loaded automatically.
        resume_path = cfg.get("resume_checkpoint_path", None)
        if cfg.get("resume", False) and resume_path:
            self._load_checkpoint(resume_path)
            if self.is_main_process:
                logger.info(f"Resuming from checkpoint: {resume_path} (step {self.global_step})")
        else:
            if self.is_main_process:
                logger.info("Starting training from scratch (no checkpoint loaded)")

        # Fast-forward the LR scheduler to match global_step after a resume
        if self.global_step > 0:
            for _ in range(self.global_step):
                self.scheduler.step()

        # Training loop (iteration-based)
        iter_loader = iter(train_loader)
        loader_epoch = 0  # tracks DataLoader restarts for set_epoch shuffling
        self.model.train()
        nhist = cfg.policy.get("nhist", 3)
        accumulate_grad = cfg.get("accumulate_grad_batches", 1)
        grad_clip = cfg.get("grad_clip", 1.0)

        if self.is_main_process:
            logger.info(
                f"Starting training for {cfg.train_iters} iterations "
                f"(lr={cfg.lr}, warmup={cfg.get('lr_warmup_steps', 2000)}, "
                f"grad_clip={grad_clip}, wd={cfg.get('wd', 5e-3)})"
            )

        for step_id in trange(self.global_step, cfg.train_iters, disable=not self.is_main_process):
            # Get next batch; call set_epoch on each DataLoader restart so the
            # DistributedSampler re-shuffles with a new seed each pass.
            try:
                sample = next(iter_loader)
            except StopIteration:
                loader_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(loader_epoch)
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            # Zero gradients
            if step_id % accumulate_grad == 0:
                self.optimizer.zero_grad()

            # Prepare trajectory (skip first frame — it's the current gripper pose)
            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

            # Select gripper history
            curr_gripper = sample["curr_gripper_history"][:, -nhist:]

            # Forward pass (loss computed inside model).
            # `instruction` is polymorphic: long tensor of primitive ids in
            # primitive mode, float CLIP features in CLIP mode, None otherwise.
            instr = self._build_instruction(sample)
            loss = self.model(
                sample["trajectory"].to(self.device),
                sample["trajectory_mask"].to(self.device),
                sample["rgbs"].to(self.device),
                sample["pcds"].to(self.device),
                curr_gripper.to(self.device),
                instruction=instr,
            )

            # Backward pass
            loss.backward()

            if step_id % accumulate_grad == accumulate_grad - 1:
                # Gradient clipping before the optimizer step
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                self.optimizer.step()
                self.scheduler.step()

                # EMA update
                if self.ema_model is not None:
                    model_ref = self.model.module if self.is_distributed else self.model
                    self.ema_model.step(model_ref)

            self.global_step = step_id + 1

            # Per-step loss logging (frequent)
            if self.is_main_process and self.use_wandb and (step_id + 1) % cfg.get("log_freq", 10) == 0:
                import wandb
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                    },
                    step=step_id,
                )

            # Val checkpoint (infrequent)
            if self.is_main_process and (step_id + 1) % cfg.get("val_freq", 500) == 0:
                # Validation
                self.model.eval()
                val_metrics = self._evaluate(val_loader, step_id)
                self.model.train()

                # Save checkpoint
                val_loss = val_metrics.get("val/traj_pos_acc_001", None)
                self._save_checkpoint(topk_manager, step_id, val_loss)

                if self.use_wandb:
                    import wandb
                    wandb.log(val_metrics, step=step_id)

                logger.info(
                    f"Step {step_id} | train_loss={loss.item():.4f} | lr={self.scheduler.get_last_lr()[0]:.2e} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )

        # Cleanup
        if self.is_distributed:
            dist.destroy_process_group()
        if self.is_main_process and self.use_wandb:
            import wandb
            wandb.finish()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, loader, step_id, max_iters=20):
        """Run evaluation and compute trajectory metrics."""
        values = {}
        device = self.device
        nhist = self.cfg.policy.get("nhist", 3)

        for i, sample in enumerate(loader):
            if i >= max_iters:
                break

            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]
            curr_gripper = sample["curr_gripper_history"][:, -nhist:]

            # Run inference
            instr = self._build_instruction(sample)
            pred = self.model(
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                curr_gripper.to(device),
                instruction=instr,
                run_inference=True,
            )

            gt = sample["trajectory"].to(device)
            metrics = self._compute_metrics(pred, gt)
            for k, v in metrics.items():
                key = f"val/{k}"
                if key not in values:
                    values[key] = []
                values[key].append(v.item())

        # Average metrics
        values = {k: np.mean(v) for k, v in values.items()}

        if self.use_wandb:
            import wandb
            wandb.log(values, step=step_id)

        return values

    @staticmethod
    def _compute_metrics(pred, gt):
        """Compute trajectory prediction metrics."""
        # Position L2
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()

        # Symmetric quaternion L1
        quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
        quat_l1_neg = (pred[..., 3:7] + gt[..., 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_neg).float()
        quat_l1 = select_mask * quat_l1 + (1 - select_mask) * quat_l1_neg

        # Gripper accuracy
        if pred.shape[-1] > 7 and gt.shape[-1] > 7:
            openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).float()
            gripper_acc = openess.mean()
        else:
            gripper_acc = torch.tensor(0.0)

        return {
            "traj_pos_l2": pos_l2.mean(),
            "traj_pos_acc_001": (pos_l2 < 0.01).float().mean(),
            "traj_rot_l1": quat_l1.mean(),
            "traj_rot_acc_0025": (quat_l1 < 0.025).float().mean(),
            "traj_gripper_acc": gripper_acc,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, topk_manager, step_id, metric):
        """Save checkpoint (rank 0 only)."""
        if not self.is_main_process:
            return

        model_ref = self.model.module if self.is_distributed else self.model
        checkpoint = {
            "weight": model_ref.state_dict() if not self.is_distributed
                      else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "iter": step_id + 1,
            "best_loss": self.best_loss,
            "arch_config": {
                "use_instruction": self.cfg.policy.get("use_instruction", True),
                "lang_enhanced": self.cfg.policy.get("lang_enhanced", True),
                "use_primitive_id": self.cfg.policy.get("use_primitive_id", False),
                "num_primitives": self.cfg.policy.get("num_primitives", 4),
            },
        }
        if self.ema_model is not None:
            checkpoint["ema_weight"] = self.ema_model.averaged_model.state_dict()

        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints/diffuser_actor"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save latest
        torch.save(checkpoint, checkpoint_dir / "last.pth")

        # Save best
        if metric is not None and (self.best_loss is None or metric <= self.best_loss):
            self.best_loss = metric
            checkpoint["best_loss"] = self.best_loss
            torch.save(checkpoint, checkpoint_dir / "best.pth")
            logger.info(f"New best checkpoint at step {step_id} (metric={metric:.4f})")

        # Save periodic
        torch.save(checkpoint, checkpoint_dir / f"{step_id:07d}.pth")

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training."""
        if not os.path.isfile(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        self.model.load_state_dict(ckpt["weight"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # Keep the resumed LR as-is (scheduler will restore the correct
            # value when fast-forwarded in train()); override only to guard
            # against stale checkpoints from a different lr config.
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg.lr
        self.global_step = ckpt.get("iter", 0)
        self.best_loss = ckpt.get("best_loss", None)

        logger.info(f"Resumed from step {self.global_step}")
