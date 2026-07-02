import torch
from jaxtyping import Float

from openretina.modules.readout.gaussian import FullGaussian2d
from openretina.modules.readout.tokenized.channel_maps import ChannelToToken


class TokenizedFullGaussian2d(FullGaussian2d):
    """Sampled Gaussian readout that emits d-dim token conditioning per neuron.

    Reuses the parent's Gaussian RF sampling (sample_feature_vectors) and replaces the
    scalar channel collapse with a swappable ChannelToToken map.
    """

    def __init__(self, in_shape, outdims, channel_to_token: ChannelToToken, *, bias: bool = False, **gaussian_kwargs):
        super().__init__(in_shape=in_shape, outdims=outdims, bias=bias, **gaussian_kwargs)
        self.channel_to_token = channel_to_token

    def forward(self, x, sample=None, shift=None, out_idx=None, **kwargs) -> Float[torch.Tensor, "n_batch outdims d"]:
        feats = self.sample_feature_vectors(x, sample=sample, shift=shift, out_idx=out_idx)  # [Nb, C, outdims]
        return self.channel_to_token(feats)  # [Nb, outdims, d]

    # Signature mirrors FullGaussian2d.regularizer(reduction="sum", average=None) so this is a
    # Liskov-compatible override; `average` is accepted and ignored.
    def regularizer(self, reduction="sum", average=None) -> torch.Tensor:
        return self.channel_to_token.regularizer()
