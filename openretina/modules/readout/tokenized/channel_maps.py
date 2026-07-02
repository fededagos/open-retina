import torch
import torch.nn as nn
from jaxtyping import Float


class ChannelToToken(nn.Module):
    """Maps per-neuron sampled channel vectors to d-dim token conditioning.

    Input:  [Nb, C, outdims]  (channels sampled at each neuron's RF)
    Output: [Nb, outdims, d]  (token-conditioning vector per neuron)
    """

    out_dim: int

    def forward(self, feats: Float[torch.Tensor, "Nb C outdims"]) -> Float[torch.Tensor, "Nb outdims d"]:
        raise NotImplementedError

    def regularizer(self) -> torch.Tensor:
        return torch.zeros(())


class PerNeuronLinear(ChannelToToken):
    """Option A: an independent linear map C'->d per neuron.

    With out_dim=1 this reproduces FullGaussian2d's per-neuron channel dot product exactly.
    Parameter cost scales as in_channels * out_dim * n_neurons.
    """

    def __init__(self, in_channels: int, out_dim: int, n_neurons: int, bias: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(n_neurons, in_channels, out_dim))
        nn.init.normal_(self.weight, std=1.0 / in_channels)
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(n_neurons, out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, feats: Float[torch.Tensor, "Nb C outdims"]) -> Float[torch.Tensor, "Nb outdims d"]:
        z = torch.einsum("bcn,ncd->bnd", feats, self.weight)
        if self.bias is not None:
            z = z + self.bias
        return z

    def regularizer(self) -> torch.Tensor:
        return self.weight.abs().sum()
