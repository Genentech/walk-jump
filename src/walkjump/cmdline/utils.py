import hydra
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

from walkjump.model import DenoiseModel, NoiseEnergyModel

model_typer = {
    "denoise": DenoiseModel,
    "noise_ebm": NoiseEnergyModel,
}


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config."""
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        print("[instantiate_callbacks] No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("[instantiate_callbacks] Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(f"[instantiate_callbacks] Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
