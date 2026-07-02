import os

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# in_shape must be large enough for the realistic 35-tap temporal / 21-tap spatial core kernels.
IN_SHAPE = (2, 50, 32, 32)  # C_in, T, H, W


def _instantiate(config_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_dir = os.path.join(repo_root, "configs", "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    OmegaConf.set_struct(cfg, False)
    # No _convert_ override: keep nested core/readout as DictConfig so TokenizedCoreReadout
    # can set core.channels / readout.in_shape (the config's _recursive_: false prevents pre-instantiation).
    return hydra.utils.instantiate(cfg, in_shape=IN_SHAPE, n_neurons_dict={"sessionA": 6})


def test_perneuron_config_forward():
    model = _instantiate("tokenized_gaussian_readout_perneuron")
    model.eval()
    cond = model(torch.randn(2, *IN_SHAPE), data_key="sessionA")
    assert cond.ndim == 4 and cond.shape[2] == 6


def test_embedding_config_forward():
    model = _instantiate("tokenized_gaussian_readout_embedding")
    model.eval()
    cond = model(torch.randn(2, *IN_SHAPE), data_key="sessionA")
    assert cond.ndim == 4 and cond.shape[2] == 6
