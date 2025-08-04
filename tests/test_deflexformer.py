"""
Tests for Deflexformer neural network components.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Conditional import for PyTorch
torch = pytest.importorskip("torch", reason="PyTorch not available")

from deflex.deflexformer import DeflexformerEstimator, DeflexformerModel, DeflexformerBlock
from deflex.deflexformer.layers import ElementWiseFFN, MultiHeadAttention


class TestDeflexformerEstimator:
    """Test suite for DeflexformerEstimator."""
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test valid parameters
        estimator = DeflexformerEstimator(
            n_blocks=3,
            embedding_dim=32,
            n_heads=8
        )
        assert estimator.n_blocks == 3
        assert estimator.embedding_dim == 32
        
        # Test invalid embedding_dim
        with pytest.raises(ValueError):
            DeflexformerEstimator(embedding_dim=128)  # > 64
            
    def test_basic_fit_predict(self):
        """Test basic fit and predict functionality."""
        X = np.random.randn(20, 5, 16).astype(np.float32)
        y = np.random.randn(20, 5, 1).astype(np.float32)
        
        estimator = DeflexformerEstimator(
            n_blocks=2,
            embedding_dim=16,
            n_heads=4
        )
        
        # Test fitting
        estimator.fit(X, y, epochs=2, batch_size=10)
        assert hasattr(estimator, 'is_fitted_')
        
        # Test prediction
        predictions = estimator.predict(X)
        assert predictions.shape == y.shape
        
    def test_expand_model(self):
        """Test model expansion from single block to multiple blocks."""
        X = np.random.randn(10, 3, 8).astype(np.float32)
        y = np.random.randn(10, 3, 1).astype(np.float32)
        
        # Train single block
        estimator = DeflexformerEstimator(
            n_blocks=1,
            embedding_dim=8,
            n_heads=2
        )
        estimator.fit(X, y, epochs=1)
        
        # Expand to multiple blocks
        estimator.expand_to_full_model(3)
        assert estimator.n_blocks == 3
        assert len(estimator.model_.blocks) == 3
        
    def test_extract_block_outputs(self):
        """Test extraction of intermediate block outputs."""
        X = np.random.randn(5, 3, 8).astype(np.float32)
        y = np.random.randn(5, 3, 1).astype(np.float32)
        
        estimator = DeflexformerEstimator(
            n_blocks=2,
            embedding_dim=8,
            n_heads=2
        )
        estimator.fit(X, y, epochs=1)
        
        # Extract block outputs
        outputs = estimator.extract_block_outputs(X)
        expected_shape = (X.shape[0], X.shape[1], 8 * 2)  # 2 blocks * 8 dim each
        assert outputs.shape == expected_shape


class TestDeflexformerModel:
    """Test suite for DeflexformerModel."""
    
    def test_model_creation(self):
        """Test model creation and forward pass."""
        model = DeflexformerModel(
            embedding_dim=16,
            n_blocks=2,
            n_heads=4,
            output_dim=1
        )
        
        # Test forward pass
        X = torch.randn(2, 10, 5, 16)  # (batch, frames, elements, embedding)
        output = model(X)
        
        expected_shape = (2, 10, 5, 1)
        assert output.shape == expected_shape
        
    def test_invalid_embedding_dim(self):
        """Test validation of embedding dimension."""
        with pytest.raises(ValueError):
            DeflexformerModel(embedding_dim=128)  # > 64


class TestDeflexformerBlock:
    """Test suite for DeflexformerBlock."""
    
    def test_block_forward_pass(self):
        """Test forward pass through a single block."""
        block = DeflexformerBlock(
            embedding_dim=16,
            n_heads=4
        )
        
        # Test with 3D input
        X = torch.randn(2, 8, 6, 16)  # (batch, frames, elements, embedding)
        output = block(X)
        
        assert output.shape == X.shape  # Should preserve shape
        
    def test_attention_mechanisms(self):
        """Test that attention mechanisms are properly applied."""
        block = DeflexformerBlock(
            embedding_dim=8,
            n_heads=2
        )
        
        # Small input for testing
        X = torch.randn(1, 4, 3, 8)
        
        # Forward pass should not raise errors
        output = block(X)
        assert output.shape == X.shape


class TestLayers:
    """Test suite for individual layer components."""
    
    def test_element_wise_ffn(self):
        """Test ElementWiseFFN layer."""
        ffn = ElementWiseFFN(
            embedding_dim=16,
            hidden_dim=32
        )
        
        X = torch.randn(2, 10, 5, 16)
        output = ffn(X)
        
        assert output.shape == X.shape
        
    def test_multi_head_attention(self):
        """Test MultiHeadAttention layer."""
        attention = MultiHeadAttention(
            embedding_dim=16,
            n_heads=4
        )
        
        # Test non-causal attention
        X = torch.randn(2, 8, 16)  # (batch, seq_len, embedding)
        output = attention(X, X, X)
        
        assert output.shape == X.shape
        
    def test_causal_attention(self):
        """Test causal attention masking."""
        attention = MultiHeadAttention(
            embedding_dim=8,
            n_heads=2,
            causal=True
        )
        
        X = torch.randn(1, 5, 8)
        output = attention(X, X, X)
        
        assert output.shape == X.shape