from torch import nn
from torchvision.ops import stochastic_depth
import torch
from typing import Tuple
from einops import einsum
from typing import Literal
import math
from einops import rearrange

class DropPath(nn.Module):
    """Stochastic depth for regularization https://arxiv.org/abs/1603.09382"""

    def __init__(self, p: float = 0.0, mode: str = "row"):
        super(DropPath, self).__init__()
        assert 0 <= p <= 1
        assert mode in ("batch", "row")
        self.p, self.mode = p, mode

    def forward(self, inputs: torch.Tensor):
        return stochastic_depth(
            inputs, p=self.p, mode=self.mode, training=self.training
        )
    
class RotaryPosEmb(nn.Module):
    """
    Rotary position embedding (RoPE)
    Reference
    - Su et al. 2021 https://arxiv.org/abs/2104.09864
    - Sun et al. 2022 https://arxiv.org/abs/2212.10554
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        reg_tokens: int,
        scale_base: int = 512,
        use_xpos: bool = True,
    ):
        super(RotaryPosEmb, self).__init__()
        self.num_tokens = num_tokens
        self.reg_tokens = reg_tokens

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        )
        self.create_embedding(n=num_tokens)

    def create_embedding(self, n: int):
        device = self.scale.device
        t = torch.arange(n, dtype=self.inv_freq.dtype, device=device)
        freq = torch.einsum("i , j -> i j", t, self.inv_freq)
        freq = torch.cat((freq, freq), dim=-1)
        if self.use_xpos:
            power = (t - (n // 2)) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)
        else:
            scale = torch.ones(1, device=device)
        self.register_buffer("emb_sin", torch.sin(freq), persistent=False)
        self.register_buffer("emb_cos", torch.cos(freq), persistent=False)
        self.register_buffer("emb_scale", scale, persistent=False)

    def get_embedding(self, n: int):
        if self.emb_sin is None or self.emb_sin.shape[-2] < n:
            self.create_embedding(n)
        return self.emb_sin[:n], self.emb_cos[:n], self.emb_scale[:n]

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def rotate(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            (q * cos * scale) + (cls.rotate_half(q) * sin * scale),
            (k * cos * scale) + (cls.rotate_half(k) * sin * scale),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        n, device = q.size(2), q.device.type
        q_reg, k_reg = None, None
        if self.reg_tokens:
            q_reg = q[:, :, -self.reg_tokens :, :]
            k_reg = k[:, :, -self.reg_tokens :, :]
            n -= self.reg_tokens
            q = q[:, :, : -self.reg_tokens, :]
            k = k[:, :, : -self.reg_tokens, :]
        sin, cos, scale = self.get_embedding(n)
        q, k = self.rotate(q, k, sin, cos, scale)
        if q_reg is not None and k_reg is not None:
            q = torch.cat((q, q_reg), dim=2)
            k = torch.cat((k, k_reg), dim=2)
        return q, k


class SinCosPosEmb(nn.Module):
    def __init__(self, emb_dim: int, input_shape: Tuple[int, int]):
        super(SinCosPosEmb, self).__init__()
        assert emb_dim % 2 == 0, f"emb_dim must be divisible by 2, got {emb_dim}."
        self.emb_dim = emb_dim
        
        h, w = input_shape
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, h, w])
        emb_h = self._1d_sin_cos_pos_emb(self.emb_dim // 2, pos=grid[0])
        emb_w = self._1d_sin_cos_pos_emb(self.emb_dim // 2, pos=grid[1])
        pos_emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)

        self.register_buffer("pos_emb", pos_emb, persistent=False)

    @staticmethod
    def _1d_sin_cos_pos_emb(emb_dim: int, pos: torch.Tensor):
        omega = torch.arange(emb_dim // 2, dtype=torch.float32)
        omega /= emb_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = torch.flatten(pos)
        out = einsum(pos, omega, "m, d -> m d")
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        return emb

    def forward(self, inputs: torch.Tensor):
        b, t, p, d = inputs.shape
        return inputs + self.pos_emb[None, None, :p]


class SinusoidalPosEmb(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_length: int,
        dimension: Literal["spatial", "temporal"],
        dropout: float = 0.0,
    ):
        super(SinusoidalPosEmb, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dimension = dimension
        
        # Use a larger max_length to accommodate test set
        # Adjust this value if needed (150 for your test set)
        actual_max_length = max(max_length, 750)
        
        position = torch.arange(actual_max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        match self.dimension:
            case "temporal":
                pos_encoding = torch.zeros(1, actual_max_length, 1, d_model)
                pos_encoding[0, :, 0, 0::2] = torch.sin(position * div_term)
                pos_encoding[0, :, 0, 1::2] = torch.cos(position * div_term)
            case "spatial":
                pos_encoding = torch.zeros(1, 1, actual_max_length, d_model)
                pos_encoding[0, 0, :, 0::2] = torch.sin(position * div_term)
                pos_encoding[0, 0, :, 1::2] = torch.cos(position * div_term)
            case _:
                raise NotImplementedError(
                    f"invalid dimension {self.dimension} in "
                    f"SinusoidalPositionalEncoding"
                )

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        match self.dimension:
            case "temporal":
                outputs += self.pos_encoding[:, : inputs.size(1), :, :]
            case "spatial":
                outputs += self.pos_encoding[:, :, : inputs.size(2), :]
        return self.dropout(outputs)
