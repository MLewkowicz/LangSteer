"""DP3 Training Workspace with SLURM/DDP Support.

Supports both single-GPU and multi-GPU distributed training on SLURM clusters.
Adapted from 3D-Diffusion-Policy to follow LangSteer conventions.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import random
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Optional

from policies.dp3_components.dp3_policy import DP3
from policies.dp3_components.normalizer import LinearNormalizer
from diffusers import DDIMScheduler
from training.calvin_dataset import CalvinDataset
from training.checkpoint_util import TopKCheckpointManager
from training.ema_model import EMAModel

logger = logging.getLogger(__name__)


class DP3TrainingWorkspace:
    """
    Training workspace for DP3 policy on CALVIN dataset.

    Features:
    - Single-GPU and multi-GPU DDP training
    - SLURM cluster support
    - Gradient accumulation
    - EMA model updates
    - TopK checkpoint management
    - WandB logging
    - Resume from checkpoint

    Args:
        cfg: Training configuration (OmegaConf DictConfig)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Setup distributed training
        self._setup_distributed()

        # Set random seeds (must be after distributed setup)
        self._set_seed(cfg.seed)

        # Device setup
        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(cfg.device)

        # Build model
        self.model = self._build_model(cfg)
        self.model.to(self.device)

        # Wrap model in DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

        # EMA model
        self.ema_model = None
        if cfg.use_ema:
            self.ema_model = EMAModel(
                model=self.model.module if self.is_distributed else self.model,
                power=cfg.ema_power,
                update_after_step=cfg.get('ema_update_after_step', 0),
                inv_gamma=cfg.get('ema_inv_gamma', 1.0),
                max_value=cfg.get('ema_max_value', 0.9999)
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Logging (only on rank 0)
        self.use_wandb = cfg.get('use_wandb', True) and self.is_main_process
        if self.use_wandb:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.experiment_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume='allow'
            )

        logger.info(f"DP3TrainingWorkspace initialized on device: {self.device}")
        if self.is_distributed:
            logger.info(f"Distributed training: rank={self.rank}/{self.world_size}, local_rank={self.local_rank}")

    def _setup_distributed(self):
        """Setup distributed training environment (SLURM-aware)."""
        # Check if we're in a SLURM environment
        if 'SLURM_PROCID' in os.environ:
            # SLURM environment
            self.rank = int(os.environ['SLURM_PROCID'])
            self.local_rank = int(os.environ['SLURM_LOCALID'])
            self.world_size = int(os.environ['SLURM_NTASKS'])

            # Get master address and port
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '29500'

            self.is_distributed = True

            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )

            logger.info(f"SLURM distributed training initialized: rank={self.rank}, world_size={self.world_size}")

        elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # Standard distributed environment (torchrun, etc.)
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.is_distributed = True

            dist.init_process_group(
                backend='nccl',
                init_method='env://'
            )

            logger.info(f"Distributed training initialized: rank={self.rank}, world_size={self.world_size}")

        else:
            # Single-GPU training
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.is_distributed = False
            logger.info("Single-GPU training")

        self.is_main_process = (self.rank == 0)

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        # Adjust seed for each rank to ensure different data shuffling
        seed = seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _build_model(self, cfg: DictConfig) -> DP3:
        """Build DP3 model from configuration."""
        # Build shape_meta
        shape_meta = cfg.policy.get("shape_meta", {
            'obs': {
                'point_cloud': {'shape': [cfg.policy.num_points, 3], 'type': 'point_cloud'},
                'agent_pos': {'shape': [15], 'type': 'low_dim'}
            },
            'action': {'shape': [7]}
        })

        # Initialize noise scheduler
        scheduler = DDIMScheduler(
            num_train_timesteps=cfg.policy.scheduler.num_train_timesteps,
            beta_start=cfg.policy.scheduler.beta_start,
            beta_end=cfg.policy.scheduler.beta_end,
            beta_schedule=cfg.policy.scheduler.beta_schedule,
            clip_sample=cfg.policy.scheduler.clip_sample,
            prediction_type=cfg.policy.scheduler.prediction_type,
        )

        # Encoder config
        encoder_cfg = cfg.policy.encoder
        pointcloud_encoder_cfg = {
            'out_channels': encoder_cfg.output_dim,
            'use_layernorm': True,
            'final_norm': 'layernorm',
        }

        # Build DP3 model
        model = DP3(
            shape_meta=shape_meta,
            noise_scheduler=scheduler,
            horizon=cfg.policy.pred_horizon,
            n_action_steps=cfg.policy.action_horizon,
            n_obs_steps=cfg.policy.obs_horizon,
            num_inference_steps=cfg.policy.scheduler.num_inference_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=cfg.policy.diffusion.diffusion_step_embed_dim,
            down_dims=cfg.policy.diffusion.down_dims,
            kernel_size=cfg.policy.diffusion.kernel_size,
            n_groups=cfg.policy.diffusion.n_groups,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=encoder_cfg.output_dim,
            use_pc_color=encoder_cfg.use_pc_color,
            pointnet_type=encoder_cfg.pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        )

        return model

    def _build_dataloader(self, dataset: CalvinDataset, is_train: bool = True) -> DataLoader:
        """Build dataloader with optional distributed sampler."""
        sampler = None
        shuffle = is_train

        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=is_train,
                seed=self.cfg.seed
            )
            shuffle = False  # DistributedSampler handles shuffling

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=is_train  # Drop last incomplete batch for training
        )

        return dataloader

    def train(self):
        """Main training loop."""
        cfg = self.cfg

        # Load dataset
        if self.is_main_process:
            logger.info(f"Loading dataset from: {cfg.dataset.zarr_path}")

        dataset = CalvinDataset(
            zarr_path=cfg.dataset.zarr_path,
            horizon=cfg.horizon,
            pad_before=cfg.policy.obs_horizon - 1,
            pad_after=cfg.policy.action_horizon - 1,
            val_ratio=cfg.dataset.val_ratio,
            max_train_episodes=cfg.dataset.get('max_train_episodes', None),
            seed=cfg.seed
        )

        val_dataset = dataset.get_validation_dataset()

        # Build dataloaders
        train_loader = self._build_dataloader(dataset, is_train=True)
        val_loader = self._build_dataloader(val_dataset, is_train=False)

        # Fit normalizer (only on rank 0, then broadcast)
        if self.is_main_process:
            logger.info("Fitting normalizer...")
            normalizer = dataset.get_normalizer(mode='limits')
        else:
            normalizer = LinearNormalizer()

        # Broadcast normalizer to all ranks
        if self.is_distributed:
            # Convert normalizer to state dict for broadcasting
            if self.is_main_process:
                normalizer_state = normalizer.state_dict()
            else:
                normalizer_state = None

            # Broadcast using object list
            normalizer_state = [normalizer_state]
            dist.broadcast_object_list(normalizer_state, src=0)

            if not self.is_main_process:
                normalizer.load_state_dict(normalizer_state[0])

        # Set normalizer in model
        model_to_set = self.model.module if self.is_distributed else self.model
        model_to_set.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.averaged_model.set_normalizer(normalizer)

        # Setup checkpoint manager (only on rank 0)
        topk_manager = None
        if self.is_main_process:
            checkpoint_dir = Path(cfg.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            topk_manager = TopKCheckpointManager(
                save_dir=checkpoint_dir,
                monitor_key='val_loss',
                mode='min',
                k=cfg.get('save_top_k', 3)
            )

        # Learning rate scheduler
        if cfg.get('use_lr_scheduler', True):
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.num_epochs,
                eta_min=cfg.get('lr_min', 1e-6)
            )
        else:
            scheduler = None

        # Resume from checkpoint if specified
        if cfg.get('resume', False):
            self._load_checkpoint(cfg.get('resume_checkpoint_path', None))

        # Training loop
        if self.is_main_process:
            logger.info(f"Starting training for {cfg.num_epochs} epochs")

        for epoch in range(self.epoch, cfg.num_epochs):
            self.epoch = epoch

            # Set epoch for distributed sampler
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)

            # Train for one epoch
            train_loss = self._train_epoch(train_loader)

            # Validation
            val_loss = None
            if epoch % cfg.val_every == 0:
                val_loss = self._validate(val_loader)

            # Learning rate scheduler step
            if scheduler is not None:
                scheduler.step()

            # Checkpoint saving (only on rank 0)
            if self.is_main_process and epoch % cfg.checkpoint_every == 0:
                self._save_checkpoint(
                    topk_manager,
                    val_loss if val_loss is not None else train_loss,
                    epoch
                )

            # Logging (only on rank 0)
            if self.is_main_process:
                log_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                if val_loss is not None:
                    log_dict['val_loss'] = val_loss

                if self.use_wandb:
                    import wandb
                    wandb.log(log_dict, step=self.global_step)

                logger.info(
                    f"Epoch {epoch}/{cfg.num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f if val_loss is not None else 'N/A'} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

        # Cleanup
        if self.is_distributed:
            dist.destroy_process_group()

        if self.is_main_process:
            logger.info("Training complete!")
            if self.use_wandb:
                import wandb
                wandb.finish()

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        losses = []

        # Progress bar (only on rank 0)
        if self.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        else:
            pbar = dataloader

        gradient_accumulation_steps = self.cfg.get('gradient_accumulation_steps', 1)

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {
                'obs': {
                    'point_cloud': batch['obs']['point_cloud'].to(self.device, non_blocking=True),
                    'agent_pos': batch['obs']['agent_pos'].to(self.device, non_blocking=True)
                },
                'action': batch['action'].to(self.device, non_blocking=True)
            }

            # Forward pass
            model_to_use = self.model.module if self.is_distributed else self.model
            loss, loss_dict = model_to_use.compute_loss(batch)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.cfg.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.gradient_clip
                    )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update EMA
                if self.ema_model is not None:
                    self.ema_model.step(model_to_use)

                self.global_step += 1

            # Logging
            loss_value = loss.item() * gradient_accumulation_steps
            losses.append(loss_value)

            if self.is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f"{loss_value:.4f}"})

        # Compute mean loss across all processes
        mean_loss = np.mean(losses)
        if self.is_distributed:
            mean_loss_tensor = torch.tensor(mean_loss, device=self.device)
            dist.all_reduce(mean_loss_tensor, op=dist.ReduceOp.AVG)
            mean_loss = mean_loss_tensor.item()

        return mean_loss

    def _validate(self, dataloader: DataLoader) -> float:
        """Validation loop."""
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {
                    'obs': {
                        'point_cloud': batch['obs']['point_cloud'].to(self.device, non_blocking=True),
                        'agent_pos': batch['obs']['agent_pos'].to(self.device, non_blocking=True)
                    },
                    'action': batch['action'].to(self.device, non_blocking=True)
                }

                # Forward pass
                model_to_use = self.model.module if self.is_distributed else self.model
                loss, _ = model_to_use.compute_loss(batch)
                losses.append(loss.item())

        # Compute mean loss across all processes
        mean_loss = np.mean(losses)
        if self.is_distributed:
            mean_loss_tensor = torch.tensor(mean_loss, device=self.device)
            dist.all_reduce(mean_loss_tensor, op=dist.ReduceOp.AVG)
            mean_loss = mean_loss_tensor.item()

        return mean_loss

    def _save_checkpoint(self, topk_manager: TopKCheckpointManager, metric: float, epoch: int):
        """Save checkpoint (only called on rank 0)."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': (self.model.module if self.is_distributed else self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': OmegaConf.to_container(self.cfg, resolve=True),
        }

        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.averaged_model.state_dict()

        # Save checkpoint via TopK manager
        save_path = Path(self.cfg.checkpoint_dir) / f"epoch_{epoch:04d}.ckpt"
        torch.save(checkpoint, save_path)

        # Update TopK
        topk_manager.save_topk(save_path, metric, epoch)

        # Also save as latest
        latest_path = Path(self.cfg.checkpoint_dir) / "latest.ckpt"
        torch.save(checkpoint, latest_path)

        logger.info(f"Checkpoint saved: {save_path}")

    def _load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load checkpoint for resuming training."""
        if checkpoint_path is None:
            checkpoint_path = Path(self.cfg.checkpoint_dir) / "latest.ckpt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        model_to_load = self.model.module if self.is_distributed else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load EMA model
        if self.ema_model is not None and 'ema_model_state_dict' in checkpoint:
            self.ema_model.averaged_model.load_state_dict(checkpoint['ema_model_state_dict'])

        # Load training state
        self.epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.global_step = checkpoint['global_step']

        logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")
