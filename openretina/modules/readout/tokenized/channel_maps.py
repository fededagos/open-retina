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


class EmbeddingConditioned(ChannelToToken):
    """Option C: a shared C'->d projection modulated by a per-neuron embedding.

    Per-neuron parameter cost is just embed_dim (vs in_channels*out_dim in PerNeuronLinear),
    so this scales to many neurons and transfers to new neurons by adding embedding rows.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        n_neurons: int,
        embed_dim: int = 16,
        mode: str = "film",
        hidden: int | None = None,
    ):
        super().__init__()
        if mode not in ("film", "concat"):
            raise ValueError(f"mode must be 'film' or 'concat', got {mode!r}")
        self.out_dim = out_dim
        self.mode = mode
        self.embedding = nn.Embedding(n_neurons, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.proj = nn.Linear(in_channels, out_dim)
        if mode == "film":
            self.film = nn.Linear(embed_dim, 2 * out_dim)
        else:
            hidden_dim = hidden if hidden is not None else out_dim
            self.mlp = nn.Sequential(
                nn.Linear(out_dim + embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
            )

    def forward(self, feats: Float[torch.Tensor, "Nb C outdims"]) -> Float[torch.Tensor, "Nb outdims d"]:
        nb, _, n = feats.shape
        wf = self.proj(feats.permute(0, 2, 1))  # [Nb, N, d]
        e = self.embedding.weight  # [N, embed_dim]
        if self.mode == "film":
            gamma, beta = self.film(e).chunk(2, dim=-1)  # each [N, d]
            return gamma.unsqueeze(0) * wf + beta.unsqueeze(0)
        e_exp = e.unsqueeze(0).expand(nb, n, -1)  # [Nb, N, embed_dim]
        return self.mlp(torch.cat([wf, e_exp], dim=-1))

    def regularizer(self) -> torch.Tensor:
        return torch.zeros((), device=self.embedding.weight.device)
