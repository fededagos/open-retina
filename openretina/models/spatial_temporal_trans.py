import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple
import torch.nn.functional as F
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Any


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        ff_activation: str,
        mha_dropout: float,
        ff_dropout: float,
        is_causal: bool,
        norm: str,
        normalize_qk: bool,
    ):
        super(TransformerBlock, self).__init__()
        self.input_shape = input_shape

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_dim = ff_dim
        self.ff_activation = ff_activation
        self.mha_dropout = mha_dropout
        self.ff_dropout = ff_dropout

        self.is_causal = is_causal
        self.normalize_qk = normalize_qk
 
        self.Demb = input_shape[-1]
        self.emb_dim = input_shape[-1]  # Added: alias for Demb
        self.inner_dim = head_dim * num_heads

        # Check if flash attention is available
        self.flash_attention = hasattr(F, 'scaled_dot_product_attention')

        self.register_buffer("scale", torch.tensor(head_dim**-0.5))

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        if self.flash_attention:
            # Adding dropout to Flash attention layer significantly increase memory usage
            outputs = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        else:
            l, s = q.size(-2), k.size(-2)
            attn_bias = torch.zeros(l, s, dtype=q.dtype, device=q.device)
            if self.is_causal:
                mask = torch.ones(l, s, dtype=torch.bool, device=q.device).tril(0)
                attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
                attn_bias.to(q.dtype)
            attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
            attn_weights += attn_bias
            attn = torch.softmax(attn_weights, dim=-1)
            outputs = torch.matmul(attn, v)
        outputs = F.dropout(outputs, p=self.mha_dropout, training=self.training)
        return outputs


class ParallelAttentionBlock(TransformerBlock):
    """
    Standard transformer attention block with multi-head self-attention,
    feedforward network, and residual connections.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        ff_activation: str,
        mha_dropout: float,
        ff_dropout: float,
        is_causal: bool,
        norm: str,
        normalize_qk: bool,
    ):
        super(ParallelAttentionBlock, self).__init__(
            input_shape=input_shape,
            num_heads=num_heads,
            head_dim=head_dim,
            ff_dim=ff_dim,
            ff_activation=ff_activation,
            mha_dropout=mha_dropout,
            ff_dropout=ff_dropout,
            is_causal=is_causal,
            norm=norm,
            normalize_qk=normalize_qk,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.Demb)
        
        # Get activation function from string
        if ff_activation.lower() == 'gelu':
            ff_activation_fn = nn.GELU
        elif ff_activation.lower() == 'relu':
            ff_activation_fn = nn.ReLU
        elif ff_activation.lower() == 'elu':
            ff_activation_fn = nn.ELU
        else:
            ff_activation_fn = nn.GELU  # default
        
        # Fused linear layer for Q, K, V, and FF
        ff_out = ff_dim
        self.fused_dims = (self.inner_dim, self.inner_dim, self.inner_dim, ff_out)
        self.fused_linear = nn.Linear(
            in_features=self.emb_dim, 
            out_features=sum(self.fused_dims), 
            bias=False
        )
        
        # Attention output projection
        self.attn_out = nn.Linear(
            in_features=self.inner_dim, 
            out_features=self.emb_dim, 
            bias=False
        )
        
        # Feedforward output
        self.ff_out = nn.Sequential(
            ff_activation_fn(),
            nn.Dropout(p=ff_dropout),
            nn.Linear(in_features=ff_dim, out_features=self.emb_dim, bias=False),
        )

        # Q, K normalization (optional)
        self.normalize_qk = normalize_qk
        if self.normalize_qk:
            self.norm_q = nn.LayerNorm(self.inner_dim)
            self.norm_k = nn.LayerNorm(self.inner_dim)

        # Stochastic depth / drop path (optional - set to identity if not needed)
        self.drop_path1 = nn.Identity()  # Can be replaced with DropPath if needed
        self.drop_path2 = nn.Identity()
        
        # RoPE (optional - disabled by default)
        self.use_rope = False
        
        # Initialize weights
        self.apply(self.init_weight)

    def parallel_attention(self, inputs: torch.Tensor):
        # Normalize input
        outputs = self.norm(inputs)
        
        # Fused linear projection for Q, K, V, and FF
        q, k, v, ff = self.fused_linear(outputs).split(self.fused_dims, dim=-1)
        
        # Optionally normalize Q and K
        if self.normalize_qk:
            q, k = self.norm_q(q), self.norm_k(k)
        
        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rotary_position_embedding(q=q, k=k)

        # Scaled dot-product attention
        attn = self.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "b h n d -> b n (h d)")
        
        # Parallel residual connections for attention and feedforward
        outputs = (
            inputs
            + self.drop_path1(self.attn_out(attn))
            + self.drop_path2(self.ff_out(ff))
        )
        return outputs

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        outputs = self.parallel_attention(outputs)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        depth: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        ff_activation: str,
        mha_dropout: float,
        ff_dropout: float,
        is_causal: bool,
        norm: str,
        normalize_qk: bool,
    ):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList(
            [
                ParallelAttentionBlock(
                    input_shape=input_shape,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    ff_activation=ff_activation,
                    mha_dropout=mha_dropout,
                    ff_dropout=ff_dropout,
                    is_causal=is_causal,
                    norm=norm,
                    normalize_qk=normalize_qk,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs)
        return outputs


class ViViT(nn.Module):
    def __init__(self, args: Any, input_shape: Tuple[int, ...], verbose: int = 0):
        super(ViViT, self).__init__()
        emb_dim, num_heads = args.Demb, args.num_heads

        # Set defaults
        if not hasattr(args, 'core_parallel_attention'):
            args.parallel_attention = True
        if args.parallel_attention and verbose:
            print(f"Use parallel attention and MLP in ViViT.")

        if not hasattr(args, "core_use_causal_attention"):
            args.use_causal_attention = False
        if args.use_causal_attention and verbose:
            print(f"Enable causal attention in temporal Transformer.")

        normalize_qk = hasattr(args, "core_norm_qk") and args.norm_qk

        # Spatial transformer processes each time step independently
        # Input: (num_spatial_patches, emb_dim)
        self.spatial_transformer = Transformer(
            input_shape=(input_shape[1], emb_dim),  # (num_patches, emb_dim)
            depth=args.spatial_depth,
            num_heads=num_heads,
            head_dim=args.head_dim,
            ff_dim=args.ff_dim,
            ff_activation=args.ff_activation,
            mha_dropout=args.mha_dropout,
            ff_dropout=args.ff_dropout,
            is_causal=False,
            norm=args.norm,
            normalize_qk=normalize_qk,
        )
        
        # Temporal transformer processes each spatial location across time
        # Input: (num_temporal_patches, emb_dim)
        self.temporal_transformer = Transformer(
            input_shape=(input_shape[0], emb_dim),  # (num_time_patches, emb_dim)
            depth=args.temporal_depth,
            num_heads=num_heads,
            head_dim=args.head_dim,
            ff_dim=args.ff_dim,
            ff_activation=args.ff_activation,
            mha_dropout=args.mha_dropout,
            ff_dropout=args.ff_dropout,
            is_causal=args.use_causal_attention,
            norm=args.norm,
            normalize_qk=normalize_qk,
        )

        self.output_shape = input_shape
        
        # Optional regularization
        self.reg_scale = 0.0  # Set to non-zero if you want L1 regularization

    def compile(self):
        """Compile spatial and temporal transformer modules"""
        print(f"torch.compile spatial and temporal transformers in ViViT")
        self.spatial_transformer = torch.compile(
            self.spatial_transformer,
            fullgraph=True,
        )
        self.temporal_transformer = torch.compile(
            self.temporal_transformer,
            fullgraph=True,
        )

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(self, inputs: torch.Tensor):
        """
        Input: (B, T, P, C) where T=temporal patches, P=spatial patches, C=emb_dim
        Output: (B, T, P, C)
        """
        outputs = inputs
        b, t, p, _ = outputs.shape

        # Spatial attention: process each time step independently
        # Reshape to process all time steps in batch
        outputs = rearrange(outputs, "b t p c -> (b t) p c")
        outputs = self.spatial_transformer(outputs)
        outputs = rearrange(outputs, "(b t) p c -> b t p c", b=b)

        # Temporal attention: process each spatial location across time
        # Reshape to process all spatial locations in batch
        outputs = rearrange(outputs, "b t p c -> (b p) t c")
        outputs = self.temporal_transformer(outputs)
        outputs = rearrange(outputs, "(b p) t c -> b t p c", b=b)

        return outputs