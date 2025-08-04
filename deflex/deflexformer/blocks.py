"""
Deflexformer block implementation with three-stage attention mechanism.
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import ElementWiseFFN, MultiHeadAttention


class DeflexformerBlock(nn.Module):
    """
    Single Deflexformer block implementing the three-stage process:
    1. Element-wise transformation (small FFN for each element)
    2. Frame-wise multi-head attention (same frame, different elements)  
    3. Element-wise causal multi-head attention (same element, different frames)
    
    All stages include residual connections and layer normalization.
    
    Parameters
    ----------
    embedding_dim : int
        Embedding dimension.
    n_heads : int, default=8
        Number of attention heads.
    ffn_hidden_dim : int, optional
        Hidden dimension for FFN. If None, uses 4 * embedding_dim.
    dropout : float, default=0.1
        Dropout probability.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int = 8,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.ffn_hidden_dim = ffn_hidden_dim or 4 * embedding_dim
        
        # Stage 1: Element-wise transformation
        self.element_wise_ffn = ElementWiseFFN(
            embedding_dim=embedding_dim,
            hidden_dim=self.ffn_hidden_dim,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # Stage 2: Frame-wise attention (elements within same frame)
        self.frame_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
            causal=False
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Stage 3: Element-wise causal attention (frames within same element)
        self.element_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
            causal=True
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Deflexformer block.
        
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_frames, n_elements, embedding_dim)
            Input tensor.
            
        Returns
        -------
        output : torch.Tensor
            Block output with same shape as input.
        """
        batch_size, n_frames, n_elements, embedding_dim = x.shape
        
        # Stage 1: Element-wise transformation
        # Apply FFN to each element independently
        x1 = self.element_wise_ffn(x)
        x = self.norm1(x + x1)  # Residual connection + LayerNorm
        
        # Stage 2: Frame-wise attention
        # Group by frames, attend across elements
        x_frames = x.view(batch_size * n_frames, n_elements, embedding_dim)
        x2 = self.frame_attention(x_frames, x_frames, x_frames)
        x2 = x2.view(batch_size, n_frames, n_elements, embedding_dim)
        x = self.norm2(x + x2)  # Residual connection + LayerNorm
        
        # Stage 3: Element-wise causal attention  
        # Group by elements, attend across frames (causal)
        x_elements = x.permute(0, 2, 1, 3).contiguous()  # (batch, elements, frames, dim)
        x_elements = x_elements.view(batch_size * n_elements, n_frames, embedding_dim)
        x3 = self.element_attention(x_elements, x_elements, x_elements)
        x3 = x3.view(batch_size, n_elements, n_frames, embedding_dim)
        x3 = x3.permute(0, 2, 1, 3).contiguous()  # Back to (batch, frames, elements, dim)
        x = self.norm3(x + x3)  # Residual connection + LayerNorm
        
        return x