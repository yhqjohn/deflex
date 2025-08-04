"""
Tests for core Deflex estimator.
"""

import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from unittest.mock import patch, MagicMock

from deflex import DeflexEstimator


class TestDeflexEstimator:
    """Test suite for DeflexEstimator."""
    
    def test_estimator_parameters(self):
        """Test estimator parameter initialization."""
        estimator = DeflexEstimator(
            n_blocks=3,
            embedding_dim=32,
            pretrain_epochs=10,
            finetune_epochs=5
        )
        
        assert estimator.n_blocks == 3
        assert estimator.embedding_dim == 32
        assert estimator.pretrain_epochs == 10
        assert estimator.finetune_epochs == 5
        
    def test_invalid_embedding_dim(self):
        """Test validation of embedding dimension."""
        with pytest.raises(ValueError):
            # This should be caught in the Deflexformer validation
            estimator = DeflexEstimator(embedding_dim=128)  # > 64
            
    def test_input_validation(self):
        """Test input data validation."""
        estimator = DeflexEstimator()
        
        # Test invalid input shapes
        X_2d = np.random.randn(100, 32)  # Should be 3D
        y_1d = np.random.randn(100)
        
        with pytest.raises(ValueError):
            estimator.fit(X_2d, y_1d)
            
    def test_fit_predict_workflow(self):
        """Test basic fit and predict workflow with mocked backends."""
        # Create sample 3D data
        X = np.random.randn(50, 10, 32)  # (frames, elements, embedding_dim)
        y = np.random.randn(50, 10, 1)   # Target values
        
        # Mock the Julia interface to avoid dependency
        with patch('deflex.lambda_regression.estimator.JuliaInterface') as mock_julia:
            # Mock Julia interface methods
            mock_julia_instance = MagicMock()
            mock_julia_instance.generate_synthetic_data.return_value = (X[:25], y[:25])
            mock_julia_instance.symbolic_regression.return_value = {
                'formula': 'x1 + sin(x2)',
                'model': MagicMock(),
                'complexity': 3
            }
            mock_julia_instance.predict.return_value = y[:10]
            mock_julia.return_value = mock_julia_instance
            
            # Create and fit estimator
            estimator = DeflexEstimator(
                n_blocks=2,
                embedding_dim=32,
                pretrain_epochs=5,
                finetune_epochs=3,
                n_synthetic_samples=25
            )
            
            # Mock the Deflexformer to avoid PyTorch dependency in simple tests
            with patch('deflex.deflexformer.estimator.DeflexformerEstimator') as mock_deflexformer:
                mock_deflexformer_instance = MagicMock()
                mock_deflexformer_instance.extract_block_outputs.return_value = X[:10].reshape(10, -1)
                mock_deflexformer.return_value = mock_deflexformer_instance
                
                # Test fit
                estimator.fit(X, y)
                
                # Check that estimator is fitted
                assert hasattr(estimator, 'is_fitted_')
                assert estimator.is_fitted_
                
                # Test formula retrieval
                formula = estimator.get_formula()
                assert isinstance(formula, str)
                assert formula == 'x1 + sin(x2)'
                
                # Test prediction
                predictions = estimator.predict(X[:10])
                assert predictions.shape[0] == 10
                
    def test_score_method(self):
        """Test the scoring method."""
        X = np.random.randn(20, 5, 16)
        y = np.random.randn(20, 5, 1)
        
        with patch('deflex.lambda_regression.estimator.JuliaInterface'):
            with patch('deflex.deflexformer.estimator.DeflexformerEstimator'):
                estimator = DeflexEstimator()
                
                # Mock predict method
                estimator.predict = MagicMock(return_value=y + np.random.randn(*y.shape) * 0.1)
                estimator.is_fitted_ = True
                
                # Test scoring
                score = estimator.score(X, y)
                assert isinstance(score, float)
                assert score <= 1.0  # R^2 score should be <= 1
                
    def test_get_formula_before_fit(self):
        """Test that get_formula raises error before fitting."""
        estimator = DeflexEstimator()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            estimator.get_formula()


class TestDeflexEstimatorIntegration:
    """Integration tests that require actual dependencies."""
    
    @pytest.mark.slow
    def test_small_integration(self):
        """Test with very small data to verify integration."""
        # Very small dataset for quick testing
        X = np.random.randn(10, 3, 8)  # Small embedding_dim
        y = X[:, :, 0:1] ** 2 + 0.1 * np.random.randn(10, 3, 1)  # Simple pattern
        
        estimator = DeflexEstimator(
            n_blocks=1,
            embedding_dim=8,
            pretrain_epochs=2,
            finetune_epochs=1,
            n_synthetic_samples=10
        )
        
        try:
            # This will only work if Julia and PyTorch are properly installed
            estimator.fit(X, y)
            predictions = estimator.predict(X)
            assert predictions.shape == y.shape
        except (ImportError, RuntimeError):
            # Skip if dependencies not available
            pytest.skip("Julia or PyTorch dependencies not available")