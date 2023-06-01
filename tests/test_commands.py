from typing import Callable

import hydra
import pytest
from omegaconf import DictConfig, OmegaConf

from walkjump.cmdline import train

CONFIG_PATH = "../src/walkjump/hydra_config"
OVERRIDES = {"train": ["++dryrun=true"]}


@pytest.mark.parametrize("cmd_name,cmd", [("train", train)])
def test_cmdline(cmd_name: str, cmd: Callable[[DictConfig], bool]):
    """TODO: train score and ebm models for one step"""
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = hydra.compose(config_name=cmd_name, overrides=OVERRIDES.get(cmd_name))
        print(OmegaConf.to_yaml(cfg))
        assert cmd(cfg)
