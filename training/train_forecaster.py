"""Training script for trajectory forecaster.

Trains a neural network to predict clean trajectories from noisy ones, using the
same infrastructure as DP3 training (DDP, normalizer, checkpointing, etc.).
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import random
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
from typing import Optional

from forecasters.trajectory_forecaster import TrajectoryForecaster
from policies.dp3_components.encoder import DP3Encoder
from policies.dp3_components.normalizer import LinearNormalizer
from diffusers import DDIMScheduler
from training.calvin_dataset import CalvinDataset
from training.checkpoint_util import TopKCheckpointManager
from training.ema_model import EMAModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecasterTrainingWorkspace:
    """
    Training workspace for trajectory forecaster on CALVIN dataset.

    Features:
    - Single-GPU and multi-GPU DDP training
    - SLURM cluster support
    - Gradient accumulation
    - EMA model updates
    - TopK checkpoint management
    - WandB logging (optional)
    - Resume from checkpoint

    Args:
        cfg: Training configuration (OmegaConf DictConfig)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Setup distributed training
        self._setup_distributed()

        # Set random seeds (must be after distributed setup)
        self._set_seed(cfg.training.seed)

        # Device setup
        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(cfg.device)

        # Build model
        self.forecaster = self._build_forecaster(cfg)
        self.forecaster.to(self.device)

        # Wrap model in DDP if distributed
        if self.is_distributed:
            self.forecaster = DDP(
                self.forecaster,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

        # EMA model
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = EMAModel(
                model=self.forecaster.module if self.is_distributed else self.forecaster,
                power=cfg.training.ema_power,
                update_after_step=0,
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.forecaster.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Logging (only on rank 0)
        self.use_wandb = cfg.training.use_wandb and self.is_main_process
        if self.use_wandb:
            import wandb
            wandb.init(
                project=cfg.training.wandb_project,
                name=cfg.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume='allow'
            )

        # Output directory
        self.output_dir = Path(cfg.training.output_dir)
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Save config
            with open(self.output_dir / "config.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

        # Checkpoint manager
        if self.is_main_process:
            self.checkpoint_manager = TopKCheckpointManager(
                save_dir=self.output_dir / "checkpoints",
                k=cfg.training.keep_topk,
                metric_mode='min'  # Lower loss is better
            )

        logger.info(f"ForecasterTrainingWorkspace initialized on device: {self.device}")
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

    def _build_forecaster(self, cfg: DictConfig) -> TrajectoryForecaster:
        """Build trajectory forecaster from configuration."""
        # Build encoder if requested
        encoder = None
        if cfg.encoder.use_encoder:
            if cfg.encoder.encoder_type == "dp3":
                # Build DP3Encoder
                observation_space = {
                    'point_cloud': [2048, 3],  # XYZ only (no color)
                    'agent_pos': [15]
                }
                pointcloud_encoder_cfg = {
                    'out_channels': cfg.encoder.encoder_output_dim,
                    'use_layernorm': True,
                    'final_norm': 'layernorm',
                }
                encoder = DP3Encoder(
                    observation_space=observation_space,
                    out_channel=cfg.encoder.encoder_output_dim,
                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                    use_pc_color=cfg.encoder.use_pc_color,
                    pointnet_type=cfg.encoder.pointnet_type,
                )

                # Optionally load pretrained DP3 encoder
                if cfg.encoder.dp3_checkpoint is not None:
                    logger.info(f"Loading DP3 encoder from {cfg.encoder.dp3_checkpoint}")
                    checkpoint = torch.load(cfg.encoder.dp3_checkpoint, map_location='cpu')
                    # Extract encoder state_dict from DP3 checkpoint
                    encoder_state_dict = {k.replace('obs_encoder.', ''): v
                                         for k, v in checkpoint['model_state_dict'].items()
                                         if k.startswith('obs_encoder.')}
                    encoder.load_state_dict(encoder_state_dict)

                # Optionally freeze encoder
                if cfg.encoder.freeze:
                    logger.info("Freezing encoder weights")
                    for param in encoder.parameters():
                        param.requires_grad = False
            else:
                raise ValueError(f"Unknown encoder_type: {cfg.encoder.encoder_type}")

        # Build forecaster
        forecaster = TrajectoryForecaster(
            obs_encoding_dim=cfg.model.obs_encoding_dim,
            trajectory_dim=cfg.model.trajectory_dim,
            horizon=cfg.model.horizon,
            time_embed_dim=cfg.model.time_embed_dim,
            traj_encoder_type=cfg.model.traj_encoder_type,
            hidden_dims=cfg.model.hidden_dims,
            encoder=encoder,
            use_layernorm=cfg.model.use_layernorm,
        )

        logger.info(f"Forecaster initialized with {sum(p.numel() for p in forecaster.parameters())} parameters")
        if encoder is not None:
            trainable = sum(p.numel() for p in forecaster.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters: {trainable}")

        return forecaster

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
                seed=self.cfg.training.seed
            )
            shuffle = False  # DistributedSampler handles shuffling

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=is_train
        )

        return dataloader

    def train(self):
        """Main training loop."""
        cfg = self.cfg

        # Load dataset
        logger.info(f"Loading dataset from {cfg.dataset.zarr_path}")
        dataset = CalvinDataset(
            zarr_path=cfg.dataset.zarr_path,
            horizon=cfg.dataset.horizon,
            pad_before=cfg.dataset.pad_before,
            pad_after=cfg.dataset.pad_after,
            max_train_episodes=cfg.dataset.max_train_episodes,
        )

        # Get validation dataset
        val_dataset = dataset.get_validation_dataset(val_ratio=cfg.dataset.val_ratio)

        # Fit normalizer on training data (only on rank 0, then broadcast)
        if self.is_main_process:
            logger.info("Fitting normalizer on training data...")
            self.normalizer = dataset.get_normalizer(mode='limits')
        else:
            self.normalizer = LinearNormalizer()

        # Broadcast normalizer to all ranks
        if self.is_distributed:
            # Simple broadcast: save/load normalizer stats
            if self.is_main_process:
                norm_stats = self.normalizer.get_all_stats()
                torch.save(norm_stats, '/tmp/normalizer_stats.pt')
            dist.barrier()
            if not self.is_main_process:
                norm_stats = torch.load('/tmp/normalizer_stats.pt')
                self.normalizer = LinearNormalizer.create_from_stats(norm_stats)
            dist.barrier()

        # Build dataloaders
        train_loader = self._build_dataloader(dataset, is_train=True)
        val_loader = self._build_dataloader(val_dataset, is_train=False)

        logger.info(f"Training dataset: {len(dataset)} episodes, {len(train_loader)} batches")
        logger.info(f"Validation dataset: {len(val_dataset)} episodes, {len(val_loader)} batches")

        # Initialize noise scheduler (for adding noise to trajectories)
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.noise_scheduler.num_train_timesteps,
            beta_start=cfg.noise_scheduler.beta_start,
            beta_end=cfg.noise_scheduler.beta_end,
            beta_schedule=cfg.noise_scheduler.beta_schedule,
            clip_sample=cfg.noise_scheduler.clip_sample,
        )

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.epoch, cfg.training.num_epochs):
            self.epoch = epoch

            # Set epoch for distributed sampler
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)

            # Train epoch
            train_loss = self._train_epoch(train_loader)

            # Validation
            if epoch % cfg.training.val_every == 0:
                val_loss = self._validate(val_loader)

                # Logging
                if self.is_main_process:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'global_step': self.global_step
                        })

                    # Save checkpoint
                    if epoch % cfg.training.checkpoint_every == 0 or val_loss < best_val_loss:
                        self._save_checkpoint(epoch, val_loss)

                    best_val_loss = min(best_val_loss, val_loss)
            else:
                if self.is_main_process:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'global_step': self.global_step
                        })

        logger.info("Training complete!")

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.forecaster.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {self.epoch}",
            disable=not self.is_main_process
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            obs_dict = {k: v.to(self.device) for k, v in batch['obs'].items()}
            clean_trajectory = batch['action'].to(self.device)  # (B, H, D)

            # Normalize
            clean_trajectory = self.normalizer['action'].normalize(clean_trajectory)
            obs_dict = {k: self.normalizer[f'obs.{k}'].normalize(v)
                       for k, v in obs_dict.items()
                       if f'obs.{k}' in self.normalizer}

            # Sample random timestep (geometric distribution)
            B = clean_trajectory.shape[0]
            if self.cfg.noise_sampling.strategy == "geometric":
                m = torch.distributions.geometric.Geometric(
                    self.cfg.noise_sampling.geometric_p * torch.ones(B)
                )
                timesteps = torch.clip(m.sample(), 0, 99).long().to(self.device)
            elif self.cfg.noise_sampling.strategy == "uniform":
                timesteps = torch.randint(
                    0, self.cfg.noise_scheduler.num_train_timesteps,
                    (B,), device=self.device
                )
            else:
                raise ValueError(f"Unknown noise sampling strategy: {self.cfg.noise_sampling.strategy}")

            # Add noise to trajectory (forward diffusion)
            noise = torch.randn_like(clean_trajectory)
            noisy_trajectory = self.noise_scheduler.add_noise(clean_trajectory, noise, timesteps)

            # Forward pass
            pred_clean = self.forecaster(noisy_trajectory, timesteps, obs_dict=obs_dict)

            # MSE loss
            loss = F.mse_loss(pred_clean, clean_trajectory)

            # Backward
            loss = loss / self.cfg.training.gradient_accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.cfg.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.cfg.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.forecaster.parameters(),
                        self.cfg.training.gradient_clip
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update EMA
                if self.ema_model is not None:
                    self.ema_model.step(
                        self.forecaster.module if self.is_distributed else self.forecaster
                    )

                self.global_step += 1

            total_loss += loss.item() * self.cfg.training.gradient_accumulation_steps
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item() * self.cfg.training.gradient_accumulation_steps})

        avg_loss = total_loss / num_batches

        # Average across all processes (DDP)
        if self.is_distributed:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> float:
        """Validate on validation set."""
        self.forecaster.eval()
        total_loss = 0.0
        num_batches = 0

        num_val_steps = min(self.cfg.training.num_val_steps, len(dataloader))

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_val_steps:
                break

            # Move to device
            obs_dict = {k: v.to(self.device) for k, v in batch['obs'].items()}
            clean_trajectory = batch['action'].to(self.device)

            # Normalize
            clean_trajectory = self.normalizer['action'].normalize(clean_trajectory)
            obs_dict = {k: self.normalizer[f'obs.{k}'].normalize(v)
                       for k, v in obs_dict.items()
                       if f'obs.{k}' in self.normalizer}

            # Sample random timestep
            B = clean_trajectory.shape[0]
            if self.cfg.noise_sampling.strategy == "geometric":
                m = torch.distributions.geometric.Geometric(
                    self.cfg.noise_sampling.geometric_p * torch.ones(B)
                )
                timesteps = torch.clip(m.sample(), 0, 99).long().to(self.device)
            elif self.cfg.noise_sampling.strategy == "uniform":
                timesteps = torch.randint(
                    0, self.cfg.noise_scheduler.num_train_timesteps,
                    (B,), device=self.device
                )
            else:
                raise ValueError(f"Unknown noise sampling strategy")

            # Add noise
            noise = torch.randn_like(clean_trajectory)
            noisy_trajectory = self.noise_scheduler.add_noise(clean_trajectory, noise, timesteps)

            # Forward pass
            pred_clean = self.forecaster(noisy_trajectory, timesteps, obs_dict=obs_dict)

            # MSE loss
            loss = F.mse_loss(pred_clean, clean_trajectory)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Average across all processes (DDP)
        if self.is_distributed:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save checkpoint."""
        if not self.is_main_process:
            return

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': (self.forecaster.module if self.is_distributed
                                else self.forecaster).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalizer': self.normalizer.get_all_stats(),
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'val_loss': val_loss,
        }

        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.averaged_model.state_dict()

        # Save via checkpoint manager
        self.checkpoint_manager.save_checkpoint(
            state_dict=checkpoint,
            metric_value=val_loss,
            epoch=epoch
        )

        # Always save latest
        latest_path = self.output_dir / "checkpoints" / "latest.ckpt"
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")


@hydra.main(version_base=None, config_path="../conf/forecaster", config_name="trajectory_forecaster_debug")
def main(cfg: DictConfig):
    """Main entry point."""
    workspace = ForecasterTrainingWorkspace(cfg)
    workspace.train()


if __name__ == "__main__":
    main()
