import functools

import pytest
import torch
from omegaconf import OmegaConf

from openretina.data_io.base_dataloader import DataPoint
from openretina.models.tokenized_core_readout import TokenizedCoreReadout
from openretina.modules.readout.tokenized.channel_maps import PerNeuronLinear
from openretina.modules.readout.tokenized.heads import ClassifierTokenHead
from openretina.modules.readout.tokenized.multi_readout import MultiTokenizedGaussianReadoutWrapper
from openretina.modules.readout.tokenized.temporal import StridedTemporalConvAggregator

IN_SHAPE = (2, 40, 16, 16)  # C_in, T, H, W
N_NEURONS = {"sessionA": 6}
TOKEN_DIM = 5
CODEBOOK = 16


def _core_cfg():
    return OmegaConf.create(
        {
            "_target_": "openretina.modules.core.base_core.SimpleCoreWrapper",
            "_convert_": "object",
            "temporal_kernel_sizes": [11],
            "spatial_kernel_sizes": [5],
            "gamma_input": 0.0,
            "gamma_in_sparse": 0.0,
            "gamma_hidden": 0.0,
            "gamma_temporal": 0.0,
            "input_padding": False,
            "hidden_padding": False,
            "convolution_type": "separable",
            "dropout_rate": 0.0,
            "cut_first_n_frames": 0,
            "channels": "???",
        }
    )


def _readout_cfg():
    return OmegaConf.create(
        {
            "_target_": "openretina.modules.readout.tokenized.multi_readout.MultiTokenizedGaussianReadoutWrapper",
            "_convert_": "object",
            "token_dim": TOKEN_DIM,
            "channel_map": {
                "_target_": "openretina.modules.readout.tokenized.channel_maps.PerNeuronLinear",
                "_partial_": True,
            },
            "temporal_aggregator": {
                "_target_": "openretina.modules.readout.tokenized.temporal.StridedTemporalConvAggregator",
                "in_dim": TOKEN_DIM,
                "kernel_size": 3,
                "stride": 3,
            },
            "head": {
                "_target_": "openretina.modules.readout.tokenized.heads.ClassifierTokenHead",
                "cond_dim": TOKEN_DIM,
                "codebook_size": CODEBOOK,
            },
        }
    )


def _make_model(**kwargs):
    return TokenizedCoreReadout(
        in_shape=IN_SHAPE,
        hidden_channels=[8],
        n_neurons_dict=N_NEURONS,
        core=_core_cfg(),
        readout=_readout_cfg(),
        learning_rate=1e-3,
        **kwargs,
    )


def test_forward_returns_conditioning():
    model = _make_model()
    model.eval()
    x = torch.randn(2, *IN_SHAPE)
    cond = model(x, data_key="sessionA")
    assert cond.ndim == 4  # [B, T_tok, N, d]
    assert cond.shape[0] == 2 and cond.shape[2] == 6 and cond.shape[3] == TOKEN_DIM


def test_training_step_returns_finite_loss():
    model = _make_model()
    x = torch.randn(2, *IN_SHAPE)
    t_tok = model(x, data_key="sessionA").size(1)
    targets = torch.randint(0, CODEBOOK, (2, t_tok, 6))
    loss = model.training_step(("sessionA", DataPoint(x, targets)), 0)
    assert torch.isfinite(loss)


def test_alignment_mismatch_raises():
    model = _make_model()
    x = torch.randn(2, *IN_SHAPE)
    bad_targets = torch.randint(0, CODEBOOK, (2, 999, 6))
    with pytest.raises(ValueError, match="mismatch"):
        model.training_step(("sessionA", DataPoint(x, bad_targets)), 0)


def test_configure_optimizers_default_preserves_token_accuracy_monitor():
    # With no optimizer/scheduler config, the tokenized default must be kept intact:
    # AdamW + ReduceLROnPlateau(mode="max") watching val_token_accuracy (NOT val_correlation,
    # which the generic openretina default would use but which token models never log).
    model = _make_model()
    cfg = model.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.AdamW)
    sched = cfg["lr_scheduler"]
    assert isinstance(sched["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert sched["scheduler"].mode == "max"
    assert sched["monitor"] == "val_token_accuracy"


def test_configure_optimizers_uses_configured_optimizer():
    opt_cfg = OmegaConf.create({"_target_": "torch.optim.SGD", "lr": 5e-3, "momentum": 0.9})
    model = _make_model(optimizer=opt_cfg)
    cfg = model.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.SGD)
    assert cfg["optimizer"].param_groups[0]["lr"] == 5e-3


def test_configure_optimizers_uses_configured_scheduler():
    sched_cfg = OmegaConf.create(
        {
            "_target_": "torch.optim.lr_scheduler.StepLR",
            "step_size": 5,
            "gamma": 0.5,
            "monitor": None,
            "interval": "epoch",
            "frequency": 1,
        }
    )
    model = _make_model(lr_scheduler=sched_cfg)
    cfg = model.configure_optimizers()
    assert isinstance(cfg["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.StepLR)
    assert cfg["lr_scheduler"]["monitor"] is None


def test_wrapper_can_overfit_tiny_tokens():
    # Direct module-level overfit (no Lightning trainer) to confirm the signal flows and learns.
    torch.manual_seed(0)
    agg = StridedTemporalConvAggregator(in_dim=TOKEN_DIM, kernel_size=3, stride=3)
    head = ClassifierTokenHead(cond_dim=TOKEN_DIM, codebook_size=8)
    wr = MultiTokenizedGaussianReadoutWrapper(
        in_shape=(4, 1, 8, 8),
        n_neurons_dict={"a": 3},
        token_dim=TOKEN_DIM,
        channel_map=functools.partial(PerNeuronLinear),
        temporal_aggregator=agg,
        head=head,
    )
    core_out = torch.randn(2, 4, 9, 8, 8)
    targets = torch.randint(0, 8, (2, 3, 3))  # T_tok = output_length(9) = 3
    opt = torch.optim.Adam(wr.parameters(), lr=0.05)
    first = head.compute_loss(wr(core_out, "a"), targets).item()
    for _ in range(300):
        opt.zero_grad()
        loss = head.compute_loss(wr(core_out, "a"), targets)
        loss.backward()
        opt.step()
    assert loss.item() < first
