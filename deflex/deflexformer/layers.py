"""
Layer implementations for the Deflexformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ElementWiseFFN(nn.Module):
    """
    Element-wise feed-forward network applied to each element independently.
    
    Parameters
    ----------
    embedding_dim : int
        Input and output embedding dimension.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float, default=0.1
        Dropout probability.
    activation : str, default='relu'
        Activation function ('relu', 'gelu', 'swish').
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU is Swish
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply element-wise FFN.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape ending with embedding_dim.
            
        Returns
        -------
        output : torch.Tensor
            Output tensor with same shape as input.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with optional causal masking.
    
    Parameters
    ----------
    embedding_dim : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    dropout : float, default=0.1
        Dropout probability.
    causal : bool, default=False
        Whether to apply causal (triangular) masking.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        causal: bool = False
    ):
        super().__init__()
        
        if embedding_dim % n_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})")
            
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.causal = causal
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Parameters
        ----------
        query : torch.Tensor of shape (batch_size, seq_len, embedding_dim)
            Query tensor.
        key : torch.Tensor of shape (batch_size, seq_len, embedding_dim)
            Key tensor.
        value : torch.Tensor of shape (batch_size, seq_len, embedding_dim)
            Value tensor.
        mask : torch.Tensor, optional
            Attention mask.
            
        Returns
        -------
        output : torch.Tensor
            Attention output with shape (batch_size, seq_len, embedding_dim).
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections and reshape for multi-head
        Q = self.q_linear(query).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )
        output = self.out_linear(attention)
        
        return output
        
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        """
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask if needed
        if self.causal:
            seq_len = scores.size(-1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(scores.device)
            scores.masked_fill_(causal_mask, float('-inf'))
            
        # Apply additional mask if provided
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
            
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention = torch.matmul(attention_weights, V)
        
        return attention


class GradientClipping(nn.Module):
    """
    Gradient clipping layer for training stability.
    
    Parameters
    ----------
    clip_value : float
        Maximum gradient norm.
    """
    
    def __init__(self, clip_value: float = 1.0):
        super().__init__()
        self.clip_value = clip_value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient clipping during backward pass."""
        if self.training:
            return GradientClippingFunction.apply(x, self.clip_value)
        return x


class GradientClippingFunction(torch.autograd.Function):
    """
    Custom autograd function for gradient clipping.
    """
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, clip_value: float) -> torch.Tensor:
        ctx.clip_value = clip_value
        return input_tensor
        
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        grad_norm = torch.norm(grad_output)
        if grad_norm > ctx.clip_value:
            grad_output = grad_output * (ctx.clip_value / grad_norm)
        return grad_output, None