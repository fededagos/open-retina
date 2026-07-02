import torch
import torch.nn as nn
from jaxtyping import Float


class TemporalAggregator(nn.Module):
    """Aggregates a per-frame conditioning sequence into per-token windows.

    Input:  [B, T, N, d]  (conditioning per frame per neuron)
    Output: [B, T_tok, N, d_out]  (one token-conditioning vector per ~window)
    """

    def forward(self, x: Float[torch.Tensor, "B T N d"]) -> Float[torch.Tensor, "B T_tok N d_out"]:
        raise NotImplementedError

    def output_length(self, t_in: int) -> int:
        raise NotImplementedError


class StridedTemporalConvAggregator(TemporalAggregator):
    """Learned windowed aggregation via a strided 1D conv shared across neurons.

    kernel_size / stride encode frames-per-token (the ~100 ms window).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        *,
        kernel_size: int,
        stride: int,
        padding: int = 0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv1d(in_dim, self.out_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: Float[torch.Tensor, "B T N d"]) -> Float[torch.Tensor, "B T_tok N d_out"]:
        b, t, n, d = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, d, t)  # [B*N, d, T]
        x = self.conv(x)  # [B*N, out_dim, T_tok]
        t_tok = x.shape[-1]
        x = x.reshape(b, n, self.out_dim, t_tok).permute(0, 3, 1, 2)  # [B, T_tok, N, out_dim]
        return x

    def output_length(self, t_in: int) -> int:
        return (t_in + 2 * self.padding - self.kernel_size) // self.stride + 1
