import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl


class AttentionBlock(nn.Module):
    """
    Standard transformer attention block with multi-head self-attention,
    feedforward network, and residual connections.
    """
    def __init__(self, Demb, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(Demb)
        self.attn = nn.MultiheadAttention(
            embed_dim=Demb,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(Demb)
        
        mlp_hidden = int(Demb * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(Demb, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, Demb),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, attn_mask=None):
        """
        x: (B, N, Demb)
        attn_mask: optional attention mask
        """
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x


class SpatialTransformer(nn.Module):
    """
    Spatial transformer that operates over spatial patches within each frame.
    Processes embeddings of shape (Nbatch, TP, SP, Demb) by combining batch
    and temporal dimensions to apply attention over spatial dimension.
    """
    def __init__(
        self,
        Demb=128,
        num_spatial_blocks=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        chunk_size=None  # Process in chunks to save memory
    ):
        super().__init__()
        self.Demb = Demb
        self.num_spatial_blocks = num_spatial_blocks
        self.chunk_size = chunk_size
        
        # Stack of attention blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(Demb, num_heads, mlp_ratio, dropout)
            for _ in range(num_spatial_blocks)
        ])
        
        self.ln_final = nn.LayerNorm(Demb)
    
    def forward(self, x):
        """
        Input: x of shape (Nbatch, TP, SP, Demb)
        Output: shape (Nbatch, TP, SP, Demb)
        """
        Nbatch, TP, SP, Demb = x.shape
        
        # Combine batch and temporal dimensions: (Nbatch, TP, SP, Demb) -> (Nbatch*TP, SP, Demb)
        x = rearrange(x, 'b t s d -> (b t) s d')
        
        # Process in chunks if chunk_size is specified
        if self.chunk_size is not None and x.shape[0] > self.chunk_size:
            outputs = []
            for i in range(0, x.shape[0], self.chunk_size):
                chunk = x[i:i+self.chunk_size]
                
                # Apply attention blocks over spatial dimension
                for block in self.blocks:
                    chunk = block(chunk)
                
                chunk = self.ln_final(chunk)
                outputs.append(chunk)
            
            x = torch.cat(outputs, dim=0)
        else:
            # Apply attention blocks over spatial dimension
            for block in self.blocks:
                x = block(x)
            
            x = self.ln_final(x)
        
        # Rearrange back to (Nbatch, TP, SP, Demb)
        x = rearrange(x, '(b t) s d -> b t s d', b=Nbatch, t=TP)
        
        return x


class TemporalTransformer(nn.Module):
    """
    Temporal transformer that operates over temporal patches.
    Processes embeddings by combining batch and spatial dimensions to apply
    causal attention over the temporal dimension.
    """
    def __init__(
        self,
        Demb=128,
        num_temporal_blocks=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        chunk_size=None  # Process in chunks to save memory
    ):
        super().__init__()
        self.Demb = Demb
        self.num_temporal_blocks = num_temporal_blocks
        self.chunk_size = chunk_size
        
        # Stack of attention blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(Demb, num_heads, mlp_ratio, dropout)
            for _ in range(num_temporal_blocks)
        ])
        
        self.ln_final = nn.LayerNorm(Demb)
        
        # Cache for causal mask
        self.register_buffer('causal_mask', None)
        self.max_seq_len = 0
    
    def get_causal_mask(self, seq_len, device):
        """
        Create a causal attention mask that prevents attending to future tokens.
        Returns a mask of shape (seq_len, seq_len) where future positions are masked.
        """
        if self.causal_mask is None or seq_len > self.max_seq_len:
            # Create causal mask: upper triangular matrix with -inf for future positions
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            )
            self.causal_mask = mask
            self.max_seq_len = seq_len
            return mask
        else:
            return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x):
        """
        Input: x of shape (Nbatch, TP, SP, Demb) from spatial transformer
        Output: shape (Nbatch, TP, SP, Demb)
        """
        Nbatch, TP, SP, Demb = x.shape
        
        # Rearrange to combine batch and spatial dimensions: (Nbatch, TP, SP, Demb) -> (Nbatch*SP, TP, Demb)
        x = rearrange(x, 'b t s d -> (b s) t d')
        
        # Get causal mask for temporal dimension
        causal_mask = self.get_causal_mask(TP, x.device)
        
        # Process in chunks if chunk_size is specified
        if self.chunk_size is not None and x.shape[0] > self.chunk_size:
            outputs = []
            for i in range(0, x.shape[0], self.chunk_size):
                chunk = x[i:i+self.chunk_size]
                
                # Apply attention blocks over temporal dimension with causal masking
                for block in self.blocks:
                    chunk = block(chunk, attn_mask=causal_mask)
                
                chunk = self.ln_final(chunk)
                outputs.append(chunk)
            
            x = torch.cat(outputs, dim=0)
        else:
            # Apply attention blocks over temporal dimension with causal masking
            for block in self.blocks:
                x = block(x, attn_mask=causal_mask)
            
            x = self.ln_final(x)
        
        # Rearrange back to (Nbatch, TP, SP, Demb)
        x = rearrange(x, '(b s) t d -> b t s d', b=Nbatch, s=SP)
        
        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Combined spatial and temporal transformer for video processing.
    Applies spatial transformer followed by temporal transformer.
    """
    def __init__(
        self,
        Demb=128,
        num_spatial_blocks=4,
        num_temporal_blocks=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        chunk_size=32  # Process in chunks of 32 to save memory
    ):
        super().__init__()
        self.spatial_transformer = SpatialTransformer(
            Demb=Demb,
            num_spatial_blocks=num_spatial_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            chunk_size=chunk_size
        )
        
        self.temporal_transformer = TemporalTransformer(
            Demb=Demb,
            num_temporal_blocks=num_temporal_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            chunk_size=chunk_size
        )
    
    def forward(self, x, TP, SP):
        """
        Input: x from tokenizer of shape (Nbatch, TP*SP, Demb)
        TP: number of temporal patches
        SP: number of spatial patches
        Output: shape (Nbatch, TP, SP, Demb)
        """
        Nbatch = x.shape[0]
        
        # Reshape from (Nbatch, TP*SP, Demb) to (Nbatch, TP, SP, Demb)
        x = rearrange(x, 'b (t s) d -> b t s d', t=TP, s=SP)
        
        # Apply spatial transformer
        x = self.spatial_transformer(x)
        
        # Apply temporal transformer with causal masking
        x = self.temporal_transformer(x)
        
        return x