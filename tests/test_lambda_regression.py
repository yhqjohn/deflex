"""
Tests for LambdaRegression symbolic regression components.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from deflex.lambda_regression import LambdaRegressionEstimator, OperatorSet
from deflex.lambda_regression.operators import (
    DefaultOperators, BasicOperators, ExtendedOperators, CustomOperators
)


class TestLambdaRegressionEstimator:
    """Test suite for LambdaRegressionEstimator."""
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        estimator = LambdaRegressionEstimator(
            population_size=50,
            generations=20,
            max_complexity=8
        )
        
        assert estimator.population_size == 50
        assert estimator.generations == 20
        assert estimator.max_complexity == 8
        
    def test_operator_set_validation(self):
        """Test operator set validation."""
        # Test with different operator sets
        estimator1 = LambdaRegressionEstimator(operator_set=BasicOperators())
        estimator2 = LambdaRegressionEstimator(operator_set=['sin', 'cos', '+', '-'])
        
        assert isinstance(estimator1.operator_set, OperatorSet)
        assert isinstance(estimator2.operator_set, list)
        
    @patch('deflex.lambda_regression.estimator.JuliaInterface')
    def test_fit_predict_workflow(self, mock_julia_interface):
        """Test fit and predict workflow with mocked Julia."""
        # Mock Julia interface
        mock_julia = MagicMock()
        mock_julia.symbolic_regression.return_value = {
            'formula': 'x1^2 + x2',
            'model': MagicMock(),
            'complexity': 3
        }
        mock_julia.predict.return_value = np.array([1.0, 2.0, 3.0])
        mock_julia_interface.return_value = mock_julia
        
        # Test data
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        
        estimator = LambdaRegressionEstimator()
        
        # Test fitting
        estimator.fit(X, y)
        assert hasattr(estimator, 'is_fitted_')
        assert estimator.formula_ == 'x1^2 + x2'
        
        # Test prediction
        predictions = estimator.predict(X[:3])
        np.testing.assert_array_equal(predictions, [1.0, 2.0, 3.0])
        
    @patch('deflex.lambda_regression.estimator.JuliaInterface')
    def test_synthetic_data_generation(self, mock_julia_interface):
        """Test synthetic data generation."""
        # Mock Julia interface
        mock_julia = MagicMock()
        mock_julia.generate_synthetic_data.return_value = {
            'y': np.array([0.5, 1.2, 2.1, 0.8])
        }
        mock_julia_interface.return_value = mock_julia
        
        estimator = LambdaRegressionEstimator()
        
        X, y = estimator.generate_synthetic_data(n_samples=4, n_features=2)
        
        assert X.shape == (4, 2)
        assert y.shape == (4,)
        
    def test_fallback_synthetic_data(self):
        """Test fallback synthetic data generation without Julia."""
        estimator = LambdaRegressionEstimator()
        
        # Simulate Julia failure by directly calling fallback
        X_template = np.random.randn(10, 3)
        params = {'noise_level': 0.1, 'random_state': 42}
        
        result = estimator.julia_._fallback_synthetic_data(X_template, params)
        
        assert 'y' in result
        assert len(result['y']) == 10
        
    def test_get_formula_before_fit(self):
        """Test that get_formula raises error before fitting."""
        estimator = LambdaRegressionEstimator()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            estimator.get_formula()


class TestOperatorSets:
    """Test suite for operator set classes."""
    
    def test_default_operators(self):
        """Test DefaultOperators class."""
        ops = DefaultOperators()
        
        julia_ops = ops.to_julia_format()
        assert '+' in julia_ops
        assert 'sin' in julia_ops
        
        arity_map = ops.get_arity_map()
        assert arity_map['+'] == 2
        assert arity_map['sin'] == 1
        
    def test_basic_operators(self):
        """Test BasicOperators class."""
        ops = BasicOperators()
        
        julia_ops = ops.to_julia_format()
        assert '+' in julia_ops
        assert 'sin' not in julia_ops  # Only arithmetic
        
    def test_extended_operators(self):
        """Test ExtendedOperators class."""
        ops = ExtendedOperators()
        
        julia_ops = ops.to_julia_format()
        assert 'sinh' in julia_ops
        assert 'tanh' in julia_ops
        assert 'max' in julia_ops
        
    def test_custom_operators(self):
        """Test CustomOperators class."""
        ops = CustomOperators(
            binary_operators=['+', '-', '*'],
            unary_operators=['sin', 'exp']
        )
        
        # Test adding operators
        ops.add_binary_operator('/')
        ops.add_unary_operator('log')
        
        julia_ops = ops.to_julia_format()
        assert '/' in julia_ops
        assert 'log' in julia_ops
        
        # Test removing operators
        ops.remove_operator('*')
        julia_ops = ops.to_julia_format()
        assert '*' not in julia_ops
        
    def test_operator_symbols(self):
        """Test μ and other symbol usage."""
        from deflex.lambda_regression.operators import μ_DEFAULT, μ_ARITHMETIC
        
        assert isinstance(μ_DEFAULT, DefaultOperators)
        assert isinstance(μ_ARITHMETIC, BasicOperators)
        
    def test_get_all_operator_sets(self):
        """Test getting all predefined operator sets."""
        from deflex.lambda_regression.operators import get_all_operator_sets
        
        all_sets = get_all_operator_sets()
        
        assert 'default' in all_sets
        assert 'arithmetic' in all_sets
        assert 'extended' in all_sets
        
        # Test that all values are OperatorSet instances
        for op_set in all_sets.values():
            assert isinstance(op_set, OperatorSet)


class TestJuliaInterface:
    """Test suite for JuliaInterface (mocked)."""
    
    @patch('deflex.lambda_regression.julia_interface.Julia')
    def test_julia_initialization(self, mock_julia):
        """Test Julia interface initialization."""
        from deflex.lambda_regression.julia_interface import JuliaInterface
        
        # Mock Julia setup
        mock_julia_instance = MagicMock()
        mock_julia.return_value = mock_julia_instance
        
        interface = JuliaInterface()
        
        # Should attempt to initialize Julia
        mock_julia.assert_called_once()
        
    def test_julia_failure_handling(self):
        """Test handling of Julia initialization failures."""
        from deflex.lambda_regression.julia_interface import JuliaInterface
        
        with patch('deflex.lambda_regression.julia_interface.Julia') as mock_julia:
            # Simulate Julia initialization failure
            mock_julia.side_effect = Exception("Julia not found")
            
            # Should not raise exception but set initialized to False
            interface = JuliaInterface()
            assert not interface._initialized