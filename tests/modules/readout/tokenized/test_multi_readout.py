import functools

import pytest
import torch

from openretina.modules.readout.tokenized.channel_maps import PerNeuronLinear
from openretina.modules.readout.tokenized.heads import ClassifierTokenHead
from openretina.modules.readout.tokenized.multi_readout import MultiTokenizedGaussianReadoutWrapper
from openretina.modules.readout.tokenized.temporal import StridedTemporalConvAggregator


def _make_wrapper(n_neurons_dict, token_dim=5):
    agg = StridedTemporalConvAggregator(in_dim=token_dim, kernel_size=3, stride=3)
    head = ClassifierTokenHead(cond_dim=token_dim, codebook_size=16)
    return MultiTokenizedGaussianReadoutWrapper(
        in_shape=(8, 1, 12, 10),
        n_neurons_dict=n_neurons_dict,
        token_dim=token_dim,
        channel_map=functools.partial(PerNeuronLinear),
        temporal_aggregator=agg,
        head=head,
    )


def test_forward_single_session_shape():
    wr = _make_wrapper({"sessionA": 6})
    wr.eval()
    core_out = torch.randn(2, 8, 9, 12, 10)  # [B, C', T', H', W']
    cond = wr(core_out, data_key="sessionA")
    assert cond.shape == (2, 3, 6, 5)  # [B, T_tok=output_length(9), N, d]


def test_forward_all_sessions_concatenates_neurons():
    wr = _make_wrapper({"a": 6, "b": 4})
    wr.eval()
    core_out = torch.randn(2, 8, 9, 12, 10)
    cond = wr(core_out, data_key=None)
    assert cond.shape == (2, 3, 10, 5)  # neurons concatenated: 6 + 4


def test_head_and_aggregator_not_in_session_keys():
    wr = _make_wrapper({"a": 6, "b": 4})
    assert sorted(wr.readout_keys()) == ["a", "b"]


def test_regularizer_is_scalar():
    wr = _make_wrapper({"a": 6})
    assert wr.regularizer("a").ndim == 0


def test_dimension_mismatch_raises():
    agg = StridedTemporalConvAggregator(in_dim=99, kernel_size=3, stride=3)  # != token_dim
    head = ClassifierTokenHead(cond_dim=5, codebook_size=16)
    with pytest.raises(ValueError, match="token_dim"):
        MultiTokenizedGaussianReadoutWrapper(
            in_shape=(8, 1, 12, 10),
            n_neurons_dict={"a": 6},
            token_dim=5,
            channel_map=functools.partial(PerNeuronLinear),
            temporal_aggregator=agg,
            head=head,
        )
