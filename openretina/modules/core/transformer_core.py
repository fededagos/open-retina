import torch
import torch.nn as nn
from typing import Tuple, Any
import warnings

#sono cambiati!!
from openretina.models.new_tokenizer import Tokenizer
from openretina.models.spatial_temporal_trans import ViViT

from einops.layers.torch import Rearrange
from openretina.modules.core.base_core import Core
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from typing import Any


class ViViTCoreWrapper(Core):
 
    def __init__(
            self,
            in_shape: tuple[int, int, int, int, int],  # (B, C, T, H, W)
            patch_size: int,
            temporal_patch_size: int,
            spatial_stride: int,
            temporal_stride: int,
            Demb: int,
            ptoken: float,
            pad_frame: bool,
            norm: str,
            patch_mode: bool,
            pos_encoding: int, 
            num_heads: int,
            reg_tokens: int,
            num_spatial_blocks: int,
            num_temporal_blocks: int,
            dropout: float,
            mlp_ratio: float,
            channels: int,
            spatial_depth:int,
            temporal_depth: int,
            head_dim: int,
            ff_dim: int,
            ff_activation: str,
            mha_dropout: float,
            drop_path: float,
            use_rope: bool,
            ff_dropout: float,
            use_causal_attention: bool,
            **kwargs  # Catches _target_, _convert_, and any other Hydra internals
        ):
        super(ViViTCoreWrapper, self).__init__()
        
        self.input_shape = in_shape
        
        print("1. Creating Tokenizer...")
        
        self.tokenizer = Tokenizer(
            input_shape=in_shape,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            Demb=Demb,
            ptoken=ptoken,
            pad_frame=bool(pad_frame),
            norm=norm,
            patch_mode=patch_mode,
            pos_encoding=pos_encoding
        )

        print(f"2. Tokenizer created. Output shape: {self.tokenizer.output_shape}")
        
# Create args object for ViViT
        from types import SimpleNamespace
        args = SimpleNamespace(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            Demb=Demb,
            ptoken=ptoken,
            pad_frame=pad_frame,
            norm=norm,
            patch_mode=patch_mode,
            pos_encoding = pos_encoding,
            num_heads=num_heads,
            reg_tokens = reg_tokens,
            num_spatial_blocks=num_spatial_blocks,
            num_temporal_blocks=num_temporal_blocks,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            channels=channels,
            spatial_depth = spatial_depth,
            temporal_depth=temporal_depth,
            head_dim=head_dim,
            ff_dim=ff_dim,
            ff_activation=ff_activation,
            drop_path = drop_path,
            use_rope = use_rope,
            mha_dropout=mha_dropout,
            ff_dropout=ff_dropout,
            use_causal_attention=use_causal_attention,
        )
        
        # Use the vivit_input_shape which is (T, num_patches, Demb)
        # This matches what ViViT expects
        self.vivit = ViViT(
            args,
            input_shape=self.tokenizer.vivit_input_shape,  # Use vivit_input_shape
        )
        
        print(f"3. ViViT created with input shape: {self.tokenizer.vivit_input_shape}")

        # Get the spatial dimensions for rearranging
        new_h, new_w = self.tokenizer.new_shape
        self.new_h = new_h
        self.new_w = new_w
        
        print(f"4. Spatial shape after patching: h={new_h}, w={new_w}")
        
        # Rearrange from (b, t, p, c) back to (b, c, t, h, w) format
        self.rearrange = Rearrange("b t (h w) c -> b c t h w", h=new_h, w=new_w)
        self.activation = nn.ELU()  # Note: should be nn.ELU() not nn.ELU

        # Output shape for readout: (Demb, T, H, W)
        self.output_shape = (
            Demb,
            self.tokenizer.output_shape[0],  # T
            new_h,
            new_w,
        )
        
        print(f"5. Core output shape: {self.output_shape}")

    def forward(
        self,
        inputs: torch.Tensor
    ):
        """
        Input: (B, C, T, H, W)
        Output: (B, Demb, T, H, W) where T, H, W are reduced based on patching
        """
        outputs = inputs
        
        # Tokenize: (B, C, T, H, W) -> (B, nt, nh*nw, Demb)
        outputs = self.tokenizer(outputs)
        
        # ViViT expects (B, T, P, Demb) which is what tokenizer outputs
        # ViViT processes and returns same shape
        outputs = self.vivit(outputs)
        
        # Rearrange back to spatial format: (B, T, H*W, Demb) -> (B, Demb, T, H, W)
        outputs = self.rearrange(outputs)
        
        # Apply activation
        outputs = self.activation(outputs)
        
        return outputs