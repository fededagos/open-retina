import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VideoTokenizer(nn.Module):
    """
    Tokenizes video clips into 3D spatio-temporal patches, embeds them,
    and optionally applies patch dropout.
    """
    def __init__(
        self,
        img_size=(72, 64),
        patch_size=(8, 8),
        temporal_patch_size=6,
        in_channels=2,
        Demb=128,
        ptoken=0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.Demb = Demb
        self.ptoken = ptoken

        # Compute number of spatial patches
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_spatial_patches = self.num_patches_h * self.num_patches_w

        # LayerNorm + Linear for embedding
        self.flatten_dim = in_channels * temporal_patch_size * patch_size[0] * patch_size[1]
        self.ln = nn.LayerNorm(self.flatten_dim)
        self.fc = nn.Linear(self.flatten_dim, Demb)

    def forward(self, x):
        """
        Process a single batch of video data into patch embeddings.
        
        Input: x of shape (B, C, T, H, W)
            B: batch size
            C: number of channels
            T: number of frames
            H, W: height and width
            
        Output: embeddings of shape (B, num_temporal_patches, num_spatial_patches, Demb)
        """
        B, C, T, H, W = x.shape
        
        # Check that dimensions are compatible
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input spatial dimensions {(H, W)} don't match expected {self.img_size}"
        assert C == self.in_channels, \
            f"Input channels {C} don't match expected {self.in_channels}"
        assert T % self.temporal_patch_size == 0, \
            f"Number of frames {T} must be divisible by temporal_patch_size {self.temporal_patch_size}"
        
        # Calculate number of temporal patches
        num_temporal_patches = T // self.temporal_patch_size
        
        # Reshape into patches
        # (B, C, T, H, W) -> (B, C, num_temporal_patches, temporal_patch_size, num_patches_h, patch_size[0], num_patches_w, patch_size[1])
        x = x.view(
            B, C, 
            num_temporal_patches, self.temporal_patch_size,
            self.num_patches_h, self.patch_size[0],
            self.num_patches_w, self.patch_size[1]
        )
        
        # Rearrange to group patch dimensions together
        # (B, num_temporal_patches, num_patches_h, num_patches_w, C, temporal_patch_size, patch_size[0], patch_size[1])
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        
        # Flatten each patch
        # (B, num_temporal_patches, num_patches_h, num_patches_w, flatten_dim)
        x = x.reshape(B, num_temporal_patches, self.num_patches_h, self.num_patches_w, self.flatten_dim)
        
        # Combine spatial patches
        # (B, num_temporal_patches, num_spatial_patches, flatten_dim)
        x = x.reshape(B, num_temporal_patches, self.num_spatial_patches, self.flatten_dim)
        
        # Apply LayerNorm and Linear projection
        x = self.ln(x)
        x = self.fc(x)  # (B, num_temporal_patches, num_spatial_patches, Demb)
        
        # Optional: Apply patch dropout during training
        if self.training and self.ptoken > 0:
            x = self.apply_patch_dropout(x)
        
        x = rearrange(x, 'b t s d -> b (t s) d')
        
        return x, num_temporal_patches, self.num_spatial_patches
    
    def apply_patch_dropout(self, x):
        """
        Randomly drop patches during training.
        
        Input: x of shape (B, num_temporal_patches, num_spatial_patches, Demb)
        Output: x with some patches zeroed out
        """
        B, T, S, D = x.shape
        # Create dropout mask
        mask = torch.rand(B, T, S, 1, device=x.device) > self.ptoken
        return x * mask