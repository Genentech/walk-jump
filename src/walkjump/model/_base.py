from typing import Optional

import hydra
import torch
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LinearLR

from walkjump._optim import AdamW
from walkjump.data import AbBatch

_DEFAULT_TRAINING_PARAMETERS = {
    "sigma": 1.0,
    "lr": 1e-4,
    "lr_start_factor": 0.0001,
    "weight_decay": 0.01,
    "warmup_batches": 0.01,
}


class TrainableScoreModel(LightningModule):
    def __init__(self, arch: DictConfig, training: Optional[DictConfig] = None):
        super().__init__()
        self.arch_cfg = arch
        self.training_cfg = training
        self.sigma = (self.training_cfg or DictConfig(_DEFAULT_TRAINING_PARAMETERS)).get("sigma")

        self.model = hydra.utils.instantiate(self.arch_cfg)
        self.save_hyperparameters(logger=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.score(y)

    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(), lr=self.training_cfg.lr, weight_decay=self.training_cfg.weight_decay
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": LinearLR(
                    opt,
                    start_factor=self.training_cfg.lr_start_factor,
                    end_factor=1.0,
                    total_iters=self.training_cfg.warmup_batches,
                ),
                "frequency": 1,
                "interval": "step",
            },
        }

    def training_step(self, batch: AbBatch, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        batch_size = batch.batch_tensor.size(0)
        self.log("train_loss", loss, sync_dist=True, batch_size=batch_size, rank_zero_only=True)
        return loss

    def validation_step(self, batch: AbBatch, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        batch_size = batch.batch_tensor.size(0)
        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size, rank_zero_only=True)
        return loss

    def sample_noise(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.randn_like(x.float())

    def xhat(self, y: torch.Tensor) -> torch.Tensor:
        return y + self.score(y).mul(pow(self.sigma, 2))

    def score(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(self, batch: AbBatch) -> torch.Tensor:
        raise NotImplementedError
