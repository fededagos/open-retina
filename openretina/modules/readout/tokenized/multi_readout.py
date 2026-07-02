from typing import Callable

import torch
import torch.nn as nn
from jaxtyping import Float

from openretina.modules.readout.tokenized.channel_maps import ChannelToToken
from openretina.modules.readout.tokenized.heads import TokenHead
from openretina.modules.readout.tokenized.readout import TokenizedFullGaussian2d
from openretina.modules.readout.tokenized.temporal import TemporalAggregator


class MultiTokenizedGaussianReadoutWrapper(nn.Module):
    """Multi-session tokenized Gaussian readout.

    Per-session TokenizedFullGaussian2d modules (Gaussian RF + channel->token map) live in an
    inner ModuleDict; the temporal aggregator and token head are SHARED across all sessions.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        token_dim: int,
        channel_map: Callable[..., ChannelToToken],
        temporal_aggregator: TemporalAggregator,
        head: TokenHead,
        *,
        bias: bool = False,
        init_mu_range: float = 0.1,
        init_sigma: float = 0.15,
        batch_sample: bool = True,
        align_corners: bool = True,
        gauss_type: str = "full",
        grid_mean_predictor=None,
        shared_features=None,
        shared_grid=None,
        init_grid=None,
        mean_activity=None,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.in_channels = in_shape[0]
        self.gamma = gamma
        self.channel_map = channel_map
        self.gaussian_kwargs = dict(
            in_shape=in_shape,
            init_mu_range=init_mu_range,
            init_sigma=init_sigma,
            batch_sample=batch_sample,
            align_corners=align_corners,
            gauss_type=gauss_type,
            grid_mean_predictor=grid_mean_predictor,
            shared_features=shared_features,
            shared_grid=shared_grid,
            init_grid=init_grid,
            mean_activity=mean_activity,
            bias=bias,
        )
        self.readouts = nn.ModuleDict()
        self.add_sessions(n_neurons_dict)

        self.temporal_aggregator = temporal_aggregator
        self.head = head

        # Fail fast on a misconfigured conditioning-dim seam instead of an opaque
        # Conv1d/Linear shape error deep in the first forward pass.
        agg_in = getattr(self.temporal_aggregator, "in_dim", token_dim)
        if agg_in != token_dim:
            raise ValueError(
                f"token_dim ({token_dim}) must equal temporal_aggregator.in_dim ({agg_in}): "
                "the channel map emits token_dim-wide conditioning that the aggregator consumes."
            )
        agg_out = getattr(self.temporal_aggregator, "out_dim", None)
        head_cond = getattr(self.head, "cond_dim", None)
        if agg_out is not None and head_cond is not None and agg_out != head_cond:
            raise ValueError(
                f"temporal_aggregator.out_dim ({agg_out}) must equal head.cond_dim ({head_cond}): "
                "the head consumes the aggregator's output conditioning."
            )

    def add_sessions(self, n_neurons_dict: dict[str, int]) -> None:
        duplicates = set(self.readouts.keys()).intersection(n_neurons_dict.keys())
        if duplicates:
            raise ValueError(f"Found duplicate sessions: {duplicates}. Use unique session names.")
        for key, n_neurons in n_neurons_dict.items():
            channel_to_token = self.channel_map(
                in_channels=self.in_channels, out_dim=self.token_dim, n_neurons=n_neurons
            )
            self.readouts[key] = TokenizedFullGaussian2d(
                outdims=n_neurons, channel_to_token=channel_to_token, **self.gaussian_kwargs
            )

    def readout_keys(self) -> list[str]:
        return sorted(self.readouts.keys())

    def _run_session(self, folded: torch.Tensor, key: str, b: int, t: int) -> torch.Tensor:
        z = self.readouts[key](folded)  # [B*T, n, d]
        return z.reshape(b, t, z.size(1), z.size(2))  # [B, T, n, d]

    def forward(
        self, core_out: Float[torch.Tensor, "B C T H W"], data_key: str | None = None
    ) -> Float[torch.Tensor, "B T_tok N d"]:
        b, c, t, h, w = core_out.shape
        folded = core_out.transpose(1, 2).reshape(b * t, c, h, w)  # [B*T, C, H, W]

        if data_key is None:
            per_frame = torch.cat([self._run_session(folded, key, b, t) for key in self.readout_keys()], dim=2)
        else:
            per_frame = self._run_session(folded, data_key, b, t)  # [B, T, N, d]

        return self.temporal_aggregator(per_frame)  # [B, T_tok, N, d]

    def regularizer(self, data_key: str) -> torch.Tensor:
        return self.readouts[data_key].regularizer() * self.gamma
