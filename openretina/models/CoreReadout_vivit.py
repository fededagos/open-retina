import logging
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
from jaxtyping import Int

from openretina.modules.core.base_core import Core
from openretina.modules.readout.multi_readout import MultiGaussianReadoutWrapper
from openretina.models.core_readout import BaseCoreReadout
from openretina.models.transformer_core import ViViTCore

LOGGER = logging.getLogger(__name__)


class ViViTCoreReadout(BaseCoreReadout):
    """
    ViViT-based model for neural prediction.
    Integrates ViViT core with the existing readout architecture.
    
    This model can be used as a drop-in replacement for CoreReadout:
        model = ViViTCoreReadout(
            in_shape=(2, 150, 72, 64),
            n_neurons_dict=data_info['n_neurons_dict'],
            ...
        )
    """
    
    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        n_neurons_dict: dict[str, int],
        # ViViT Core parameters
        patch_size: tuple[int, int] = (8, 8),
        temporal_patch_size: int = 6,
        Demb: int = 128,
        num_spatial_blocks: int = 4,
        num_temporal_blocks: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        core_dropout: float = 0.1,
        ptoken: float = 0.1,
        cut_first_n_frames_in_core: int = 0,
        core_output_type: str = 'flatten',
        # Readout parameters
        readout_scale: bool = True,
        readout_bias: bool = True,
        readout_gaussian_masks: bool = True,
        readout_gaussian_mean_scale: float = 6.0,
        readout_gaussian_var_scale: float = 4.0,
        readout_positive: bool = True,
        readout_gamma: float = 0.4,
        readout_gamma_masks: float = 0.0,
        readout_reg_avg: bool = False,
        # Training parameters
        learning_rate: float = 0.001,
        data_info: dict[str, Any] | None = None,
    ):
        """
        Args:
            in_shape: Input shape (channels, time, height, width)
            n_neurons_dict: Dictionary mapping session keys to number of neurons
            patch_size: Spatial patch size (height, width)
            temporal_patch_size: Number of frames per temporal patch
            Demb: Embedding dimension for transformer
            num_spatial_blocks: Number of attention blocks in spatial transformer
            num_temporal_blocks: Number of attention blocks in temporal transformer
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension in transformers
            core_dropout: Dropout rate in transformer
            ptoken: Patch dropout rate during training
            cut_first_n_frames_in_core: Number of frames to cut from core output
            core_output_type: How to format core output ('flatten', 'spatiotemporal', etc.)
            readout_*: Standard readout parameters
            learning_rate: Learning rate for optimizer
            data_info: Additional data information
        """
        # Create ViViT core
        core = ViViTCore(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            Demb=Demb,
            num_spatial_blocks=num_spatial_blocks,
            num_temporal_blocks=num_temporal_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=core_dropout,
            ptoken=ptoken,
            cut_first_n_frames=cut_first_n_frames_in_core,
            output_type=core_output_type,
            n_neurons_dict=n_neurons_dict,
        )
        
        # Determine readout input shape by running one forward pass
        in_shape_readout = self.compute_readout_input_shape(in_shape, core)
        LOGGER.info(f"Core output shape: {in_shape_readout}")
        
        # Create readout
        readout = MultiGaussianReadoutWrapper(
            in_shape_readout,
            n_neurons_dict,
            readout_scale,
            readout_bias,
            readout_gaussian_masks,
            readout_gaussian_mean_scale,
            readout_gaussian_var_scale,
            readout_positive,
            readout_gamma,
            readout_gamma_masks,
            readout_reg_avg,
        )
        
        # Initialize parent class
        super().__init__(
            core=core,
            readout=readout,
            learning_rate=learning_rate,
            data_info=data_info
        )
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()


class SimplifiedViViTCoreReadout(BaseCoreReadout):
    """
    Simplified ViViT model with fewer configurable parameters.
    Good starting point for experiments.
    """
    
    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        n_neurons_dict: dict[str, int],
        # Simplified core parameters
        Demb: int = 128,
        num_blocks: int = 4,  # Used for both spatial and temporal
        # Standard parameters
        learning_rate: float = 0.001,
        readout_gamma: float = 0.4,
        cut_first_n_frames_in_core: int = 0,
        data_info: dict[str, Any] | None = None,
    ):
        """
        Simplified initialization with fewer parameters.
        
        Args:
            in_shape: Input shape (channels, time, height, width)
            n_neurons_dict: Dictionary mapping session keys to number of neurons
            Demb: Embedding dimension
            num_blocks: Number of transformer blocks (used for both spatial and temporal)
            learning_rate: Learning rate
            readout_gamma: Readout regularization weight
            cut_first_n_frames_in_core: Number of frames to cut from core output
            data_info: Additional data information
        """
        core = ViViTCore(
            patch_size=(8, 8),
            temporal_patch_size=6,
            Demb=Demb,
            num_spatial_blocks=num_blocks,
            num_temporal_blocks=num_blocks,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1,
            ptoken=0.1,
            #cut_first_n_frames=cut_first_n_frames_in_core,
            output_type='flatten',
            n_neurons_dict=n_neurons_dict,
        )
        
        in_shape_readout = self.compute_readout_input_shape(in_shape, core)
        LOGGER.info(f"Core output shape: {in_shape_readout}")
        
        readout = MultiGaussianReadoutWrapper(
            in_shape_readout,
            n_neurons_dict,
            scale=True,
            bias=True,
            gaussian_masks=True,
            gaussian_mean_scale=6.0,
            gaussian_var_scale=4.0,
            positive_weights=True,
            gamma=readout_gamma,
            gamma_masks=0.0,
            reg_avg=False,
        )
        
        super().__init__(
            core=core,
            readout=readout,
            learning_rate=learning_rate,
            data_info=data_info
        )
        self.save_hyperparameters()

