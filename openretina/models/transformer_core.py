import warnings
import torch
import torch.nn as nn
from lightning import LightningModule
from einops import rearrange

from openretina.models.spatial_temporal_trans import SpatialTransformer, TemporalTransformer


class ViViTCore(LightningModule):
    """
    ViViT (Video Vision Transformer) Core module that integrates with the existing
    framework architecture. Compatible with the Core base class interface.
    
    This core processes videos through:
    1. Tokenization (patching + embedding)
    2. Spatial Transformer (attention over spatial patches per frame)
    3. Temporal Transformer (causal attention over temporal patches)
    """
    
    def __init__(
        self,
        patch_size: tuple[int, int] = (8, 8),
        temporal_patch_size: int = 6,
        Demb: int = 128,
        num_spatial_blocks: int = 4,
        num_temporal_blocks: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        ptoken: float = 0.1,
        cut_first_n_frames: int = 0,
        output_type: str = 'spatiotemporal',  # 'spatiotemporal', 'spatial_avg', 'temporal_avg', 'flatten'
        n_neurons_dict: dict[str, int] | None = None,  # for compatibility
        **kwargs  # to absorb any extra arguments for compatibility
    ):
        """
        Args:
            patch_size: Spatial patch size (height, width)
            temporal_patch_size: Number of frames per temporal patch
            Demb: Embedding dimension
            num_spatial_blocks: Number of attention blocks in spatial transformer
            num_temporal_blocks: Number of attention blocks in temporal transformer
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            dropout: Dropout rate
            ptoken: Patch dropout rate during training
            cut_first_n_frames: Number of frames to cut from output (for compatibility)
            output_type: How to format the output:
                - 'spatiotemporal': Keep 4D shape (B, TP, SP, Demb)
                - 'spatial_avg': Average over spatial dimension (B, TP, Demb)
                - 'temporal_avg': Average over temporal dimension (B, SP, Demb)
                - 'flatten': Flatten to 3D (B, TP*SP, Demb)
            n_neurons_dict: Dictionary of session to neuron count (for compatibility)
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.Demb = Demb
        self.ptoken = ptoken
        self._cut_first_n_frames = cut_first_n_frames
        self.output_type = output_type
        
        # These will be set when we see the first input
        self.input_channels = None
        self.img_size = None
        self.SP = None  # Number of spatial patches
        
        # Tokenization layers (initialized lazily on first forward pass)
        self.flatten_dim = None
        self.ln = None
        self.fc = None
        
        # Spatial and Temporal Transformers
        self.spatial_transformer = SpatialTransformer(
            Demb=Demb,
            num_spatial_blocks=num_spatial_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.temporal_transformer = TemporalTransformer(
            Demb=Demb,
            num_temporal_blocks=num_temporal_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self._initialized = False
    
    def _lazy_init(self, input_tensor: torch.Tensor) -> None:
        """Initialize tokenization layers based on first input."""
        if self._initialized:
            return
        
        B, C, T, H, W = input_tensor.shape
        self.input_channels = C
        self.img_size = (H, W)
        
        # Compute number of spatial patches
        self.num_patches_h = H // self.patch_size[0]
        self.num_patches_w = W // self.patch_size[1]
        self.SP = self.num_patches_h * self.num_patches_w
        
        # Initialize tokenization layers
        self.flatten_dim = C * self.temporal_patch_size * self.patch_size[0] * self.patch_size[1]
        self.ln = nn.LayerNorm(self.flatten_dim).to(input_tensor.device)
        self.fc = nn.Linear(self.flatten_dim, self.Demb).to(input_tensor.device)
        
        self._initialized = True
    
    def pad_video(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Zero-pad video along time and spatial dimensions."""
        B, C, T, H, W = x.shape
        
        ph, pw = self.patch_size
        tps = self.temporal_patch_size
        
        pad_t = (tps - T % tps) % tps
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        
        x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        return x_padded, T + pad_t
    
    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 3D spatio-temporal patches."""
        B, C, T, H, W = x.shape
        ph, pw = self.patch_size
        tps = self.temporal_patch_size
        
        patches = rearrange(
            x,
            'b c (t tp) (h ph) (w pw) -> b (t h w) c tp ph pw',
            tp=tps, ph=ph, pw=pw
        )
        return patches
    
    def embed_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Flatten and embed patches.
        patches: (B, num_tokens, C, tps, ph, pw)
        Returns: (B, num_tokens, Demb)
        """
        B, N, C, tps, ph, pw = patches.shape
        x = patches.reshape(B, N, -1)
        x = self.ln(x)
        x = self.fc(x)
        return x
    
    def patch_dropout(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Randomly zero some token embeddings during training."""
        if not self.training or self.ptoken == 0:
            return embeddings
        
        B, N, D = embeddings.shape
        mask = (torch.rand(B, N, device=embeddings.device) > self.ptoken).unsqueeze(-1)
        return embeddings * mask
    
    def forward(self, input_: torch.Tensor, data_key: str | None = None, **kwargs) -> torch.Tensor:
        """
        Forward pass through the core.
        
        Args:
            input_: Video input of shape (B, C, T, H, W)
            data_key: Session key (for compatibility, not used in core)
            **kwargs: Additional arguments (for compatibility)
        
        Returns:
            output: Transformed features. Shape depends on output_type:
                - 'spatiotemporal': (B, Demb, TP, SP, 1) - reshaped for readout compatibility
                - 'spatial_avg': (B, Demb, TP, 1, 1)
                - 'temporal_avg': (B, Demb, 1, SP, 1)
                - 'flatten': (B, Demb, TP*SP, 1, 1)
        """
        # Lazy initialization on first forward pass
        self._lazy_init(input_)
        
        # Pad video to match patch sizes
        x_padded, padded_T = self.pad_video(input_)
        
        # Extract patches
        patches = self.extract_patches(x_padded)  # (B, TP*SP, C, tps, ph, pw)
        
        # Embed patches
        embeddings = self.embed_patches(patches)  # (B, TP*SP, Demb)
        
        # Apply patch dropout
        embeddings = self.patch_dropout(embeddings)
        
        # Calculate TP
        TP = padded_T // self.temporal_patch_size
        
        # Reshape for transformers
        x = rearrange(embeddings, 'b (t s) d -> b t s d', t=TP, s=self.SP)
        
        # Apply spatial transformer
        x = self.spatial_transformer(x)  # (B, TP, SP, Demb)
        
        # Apply temporal transformer with causal masking
        x = self.temporal_transformer(x)  # (B, TP, SP, Demb)
        
        # Cut first n frames if specified
        if self._cut_first_n_frames > 0:
            frames_per_patch = self.temporal_patch_size
            patches_to_cut = (self._cut_first_n_frames + frames_per_patch - 1) // frames_per_patch
            x = x[:, patches_to_cut:, :, :]
        
        # Format output based on output_type
        # Standard readout expects (B, C, H, W, D) format
        if self.output_type == 'spatiotemporal':
            # Keep full spatiotemporal structure
            # Reshape to (B, Demb, TP, SP, 1) for compatibility
            output = rearrange(x, 'b t s d -> b d t s')
            output = output.unsqueeze(-1)  # Add dummy dimension
        elif self.output_type == 'spatial_avg':
            # Average over spatial dimension
            x = x.mean(dim=2)  # (B, TP, Demb)
            output = rearrange(x, 'b t d -> b d t')
            output = output.unsqueeze(-1).unsqueeze(-1)
        elif self.output_type == 'temporal_avg':
            # Average over temporal dimension
            x = x.mean(dim=1)  # (B, SP, Demb)
            output = rearrange(x, 'b s d -> b d s')
            output = output.unsqueeze(-2).unsqueeze(-1)
        elif self.output_type == 'flatten':
            # Flatten spatial and temporal
            x = rearrange(x, 'b t s d -> b (t s) d')
            output = rearrange(x, 'b n d -> b d n')
            output = output.unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        
        return output
    
    def initialize(self) -> None:
        """Initialize the core (called by model initialization)."""
        # Transformer weights are already initialized by PyTorch
        # Additional initialization can be added here if needed
        pass
    
    def regularizer(self) -> torch.Tensor | float:
        """
        Compute regularization loss.
        For transformers, we can add L2 regularization on attention weights if needed.
        """
        # Currently no regularization, but can be added
        return 0.0
    
    def __repr__(self) -> str:
        s = f"{super().__repr__()} [{self.__class__.__name__}]\n"
        s += f"  Patch size: {self.patch_size}\n"
        s += f"  Temporal patch size: {self.temporal_patch_size}\n"
        s += f"  Embedding dim: {self.Demb}\n"
        s += f"  Spatial blocks: {self.spatial_transformer.num_spatial_blocks}\n"
        s += f"  Temporal blocks: {self.temporal_transformer.num_temporal_blocks}\n"
        s += f"  Output type: {self.output_type}\n"
        return s

