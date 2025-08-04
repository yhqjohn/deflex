"""
Deflexformer model implementation using PyTorch.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .blocks import DeflexformerBlock


class DeflexformerModel(nn.Module):
    """
    Complete Deflexformer model consisting of multiple blocks.
    
    Parameters
    ----------
    embedding_dim : int
        Embedding dimension (maximum 64).
    n_blocks : int, default=5
        Number of Deflexformer blocks.
    n_heads : int, default=8
        Number of attention heads.
    output_dim : int, default=1
        Output dimension.
    dropout : float, default=0.1
        Dropout probability.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        n_blocks: int = 5,
        n_heads: int = 8,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if embedding_dim > 64:
            raise ValueError("embedding_dim must not exceed 64")
            
        self.embedding_dim = embedding_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.output_dim = output_dim
        
        # Input projection if needed
        self.input_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Stack of Deflexformer blocks with residual connections
        self.blocks = nn.ModuleList([
            DeflexformerBlock(
                embedding_dim=embedding_dim,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_blocks)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, output_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Deflexformer model.
        
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_frames, n_elements, embedding_dim)
            Input tensor.
            
        Returns
        -------
        output : torch.Tensor
            Model output.
        """
        # Input projection
        x = self.input_projection(x)
        
        # Pass through blocks with residual connections
        for block in self.blocks:
            x = x + block(x)  # Residual connection
            
        # Output projection
        x = self.output_norm(x)
        output = self.output_projection(x)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence modeling.
    """
    
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)