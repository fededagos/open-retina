import pytest
import torch

from openretina.modules.readout.tokenized.channel_maps import EmbeddingConditioned, PerNeuronLinear


def test_perneuron_linear_shape():
    cm = PerNeuronLinear(in_channels=8, out_dim=5, n_neurons=7)
    feats = torch.randn(4, 8, 7)  # [Nb, C, outdims]
    z = cm(feats)
    assert z.shape == (4, 7, 5)  # [Nb, outdims, d]
    assert cm.out_dim == 5


def test_perneuron_linear_d1_equals_channel_dot_product():
    torch.manual_seed(0)
    in_channels, n_neurons = 8, 7
    cm = PerNeuronLinear(in_channels=in_channels, out_dim=1, n_neurons=n_neurons, bias=False)
    # arbitrary per-neuron channel weights, shaped like FullGaussian2d.features [1, C, outdims]
    feat = torch.randn(1, in_channels, n_neurons)
    with torch.no_grad():
        # weight is [n_neurons, in_channels, out_dim]; set to match feat
        cm.weight.copy_(feat[0].transpose(0, 1).unsqueeze(-1))  # [n_neurons, in_channels, 1]

    feats = torch.randn(4, in_channels, n_neurons)
    z = cm(feats).squeeze(-1)  # [4, n_neurons]
    reference = (feats * feat).sum(1)  # the scalar collapse from FullGaussian2d
    assert torch.allclose(z, reference, atol=1e-5)


def test_regularizer_is_scalar():
    cm = PerNeuronLinear(in_channels=8, out_dim=5, n_neurons=7)
    r = cm.regularizer()
    assert r.ndim == 0


@pytest.mark.parametrize("mode", ["film", "concat"])
def test_embedding_conditioned_shape(mode):
    cm = EmbeddingConditioned(in_channels=8, out_dim=5, n_neurons=7, embed_dim=4, mode=mode)
    feats = torch.randn(4, 8, 7)
    z = cm(feats)
    assert z.shape == (4, 7, 5)
    assert cm.out_dim == 5


def test_embedding_conditioned_distinguishes_neurons():
    torch.manual_seed(0)
    cm = EmbeddingConditioned(in_channels=8, out_dim=5, n_neurons=7, embed_dim=4, mode="film")
    # identical channel features across all neurons -> outputs must still differ (per-neuron embedding)
    single = torch.randn(4, 8, 1)
    feats = single.expand(4, 8, 7).contiguous()
    z = cm(feats)
    assert not torch.allclose(z[:, 0], z[:, 1], atol=1e-4)


def test_embedding_conditioned_rejects_bad_mode():
    with pytest.raises(ValueError):
        EmbeddingConditioned(in_channels=8, out_dim=5, n_neurons=7, mode="nope")
