import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#problems here, what kind of input does the function recieve?? ideally it should be the output of natmov_dataloaders_v2, from it I have to extract movies ()
#1 session: input: torch.Size([128, 2, 50, 72, 64]), target: torch.Size([128, 50, 80])

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
        frames_per_video=150,
        batch_size=16
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.Demb = Demb
        self.ptoken = ptoken
        self.frames_per_video = frames_per_video
        self.batch_size = batch_size

        # Compute number of spatial patches
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_spatial_patches = self.num_patches_h * self.num_patches_w

        # LayerNorm + Linear for embedding
        self.flatten_dim = in_channels * temporal_patch_size * patch_size[0] * patch_size[1]
        self.ln = nn.LayerNorm(self.flatten_dim)
        self.fc = nn.Linear(self.flatten_dim, Demb)

    def split_into_videos(self, x):
        """
        Split concatenated frames into individual videos.
        Input: x of shape (C, T_total, H, W) where T_total = num_videos * frames_per_video
        Output: (num_videos, C, frames_per_video, H, W)

        to update adapted to batches, 
        """
        C, T_total, H, W = x.shape
        num_videos = T_total // self.frames_per_video
        
        # Reshape to separate videos
        x = x.view(C, num_videos, self.frames_per_video, H, W)
        # Rearrange to (num_videos, C, frames_per_video, H, W)
        x = x.permute(1, 0, 2, 3, 4)
        
        return x

    def pad_video(self, x):
        """Zero-pad video along time and spatial dimensions."""
        B, C, T, H, W = x.shape
        
        ph, pw = self.patch_size
        tps = self.temporal_patch_size

        pad_t = (tps - T % tps) % tps
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw

        x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        return x_padded

    def extract_patches(self, x):
        """Extract 3D spatio-temporal patches."""
        squeeze_output = False

        B, C, T, H, W = x.shape
        ph, pw = self.patch_size
        tps = self.temporal_patch_size

        patches = rearrange(
            x,
            'b c (t tp) (h ph) (w pw) -> b (t h w) c tp ph pw',
            tp=tps, ph=ph, pw=pw
        )

        return patches

    def embed_patches(self, patches):
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

    def patch_dropout(self, embeddings):
        """Randomly zero some token embeddings."""
        B, N, D = embeddings.shape
        mask = (torch.rand(B, N, device=embeddings.device) > self.ptoken).unsqueeze(-1)
        return embeddings * mask

    def forward(self, x, return_batches=True, return_patches=False):
        """
        Input: x of shape (C, T_total, H, W) where T_total = num_videos * frames_per_video
               OR (B, C, T, H, W) for already batched data
        Output: 
            If return_patches=True: patches without embedding
                - return_batches=True: list of patches, each (batch_size, TP*SP, C, tps, ph, pw)
                - return_batches=False: all patches (num_videos, TP*SP, C, tps, ph, pw)
            If return_patches=False: token embeddings
                - return_batches=True: list of embeddings, each (batch_size, TP*SP, Demb)
                - return_batches=False: all embeddings (num_videos, TP*SP, Demb)
        """
        # Check if input is already batched
        videos = x
        num_videos = videos.shape[0]
        
        # Process all videos
        all_results = []
        
        for i in range(0, num_videos, self.batch_size):
            batch_end = min(i + self.batch_size, num_videos)
            batch_videos = videos[i:batch_end]  # (batch_size, C, T, H, W)
            
            # Pad and extract patches
            x_padded = self.pad_video(batch_videos)
            patches = self.extract_patches(x_padded)
            
            # Ensure batch dim
            if patches.ndim == 5:
                patches = patches.unsqueeze(0)
            
            if return_patches:
                # Return raw patches without embedding
                all_results.append(patches)
            else:
                # Embed patches
                embeddings = self.embed_patches(patches)
                embeddings = self.patch_dropout(embeddings)
                all_results.append(embeddings)
        
        if return_batches:
            # Return list of batches
            return all_results
        else:
            # Concatenate all batches into single tensor
            return torch.cat(all_results, dim=0)

    def get_num_temporal_patches(self):
        """Calculate number of temporal patches given frames_per_video and temporal_patch_size."""
        padded_frames = self.frames_per_video
        remainder = padded_frames % self.temporal_patch_size
        if remainder != 0:
            padded_frames += (self.temporal_patch_size - remainder)
        return padded_frames // self.temporal_patch_size

    def get_num_spatial_patches(self):
        """Return number of spatial patches."""
        return self.num_spatial_patches


