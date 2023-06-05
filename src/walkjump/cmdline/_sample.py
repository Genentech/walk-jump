import hydra
import torch
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from walkjump.cmdline.utils import instantiate_redesign_mask, instantiate_seeds


@hydra.main(version_base=None, config_path="../hydra_config", config_name="sample")
def main(cfg: DictConfig) -> None:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)
    model = hydra.utils.instantiate(cfg.model)

    if cfg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print(f"using device {cfg.device}")
        device = torch.device(cfg.device)

    model.to(device)

    instantiate_redesign_mask(cfg.designs.redesign_regions or [])
    instantiate_seeds(cfg.designs)
