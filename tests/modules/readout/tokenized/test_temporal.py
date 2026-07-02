import torch

from openretina.modules.readout.tokenized.temporal import StridedTemporalConvAggregator


def test_output_length_matches_forward():
    agg = StridedTemporalConvAggregator(in_dim=6, kernel_size=3, stride=3)
    for t_in in (9, 12, 30, 31):
        x = torch.randn(2, t_in, 4, 6)  # [B, T, N, d]
        out = agg(x)
        assert out.shape == (2, agg.output_length(t_in), 4, 6)


def test_out_dim_can_differ():
    agg = StridedTemporalConvAggregator(in_dim=6, out_dim=10, kernel_size=3, stride=3)
    x = torch.randn(2, 9, 4, 6)
    out = agg(x)
    assert out.shape == (2, 3, 4, 10)


def test_temporal_filter_shared_across_neurons():
    torch.manual_seed(0)
    agg = StridedTemporalConvAggregator(in_dim=6, kernel_size=3, stride=3)
    # identical input across the N axis -> identical output across N
    single = torch.randn(2, 9, 1, 6)
    x = single.expand(2, 9, 4, 6).contiguous()
    out = agg(x)
    assert torch.allclose(out[:, :, 0], out[:, :, 3], atol=1e-6)
