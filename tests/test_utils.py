"""
Tests for utility functions.
"""

import pytest
import numpy as np

from deflex.utils.data import (
    validate_tensor_shape, prepare_data_for_training,
    reshape_for_symbolic_regression, create_time_windows
)
from deflex.utils.validation import (
    check_deflex_input, validate_operator_set,
    validate_training_parameters, check_random_state
)
from deflex.lambda_regression.operators import DefaultOperators, BasicOperators


class TestDataUtils:
    """Test suite for data utilities."""
    
    def test_validate_tensor_shape(self):
        """Test tensor shape validation."""
        # Valid 3D tensor
        X_valid = np.random.randn(10, 5, 32)
        validate_tensor_shape(X_valid)  # Should not raise
        
        # Invalid dimensions
        X_2d = np.random.randn(10, 32)
        with pytest.raises(ValueError):
            validate_tensor_shape(X_2d)
            
        # Embedding dimension too large
        X_large_embed = np.random.randn(10, 5, 128)
        with pytest.raises(ValueError):
            validate_tensor_shape(X_large_embed)
            
        # Too few frames
        X_no_frames = np.random.randn(0, 5, 32)
        with pytest.raises(ValueError):
            validate_tensor_shape(X_no_frames, min_frames=1)
            
    def test_prepare_data_for_training(self):
        """Test data preparation for training."""
        X = np.random.randn(100, 10, 16)
        y = np.random.randn(100, 10, 1)
        
        # Test with validation split
        X_train, y_train, X_val, y_val = prepare_data_for_training(
            X, y, validation_split=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_val) == 20
        assert X_train.shape[1:] == X.shape[1:]
        assert y_train.shape[1:] == y.shape[1:]
        
        # Test without validation split
        X_train2, y_train2, X_val2, y_val2 = prepare_data_for_training(
            X, y, validation_split=0.0
        )
        
        assert X_val2 is None
        assert y_val2 is None
        assert len(X_train2) == 100
        
    def test_reshape_for_symbolic_regression(self):
        """Test reshaping for symbolic regression."""
        # 3D tensor
        X_3d = np.random.randn(20, 8, 6, 4)
        X_reshaped = reshape_for_symbolic_regression(X_3d)
        
        expected_shape = (20, 8 * 6 * 4)
        assert X_reshaped.shape == expected_shape
        
        # 2D tensor (should pass through)
        X_2d = np.random.randn(20, 32)
        X_reshaped_2d = reshape_for_symbolic_regression(X_2d)
        assert X_reshaped_2d.shape == X_2d.shape
        
        # Invalid dimensions
        X_1d = np.random.randn(20)
        with pytest.raises(ValueError):
            reshape_for_symbolic_regression(X_1d)
            
    def test_create_time_windows(self):
        """Test time window creation."""
        X = np.random.randn(20, 5, 8)  # 20 frames
        
        # Create windows of size 5
        windows = create_time_windows(X, window_size=5, step_size=2)
        
        expected_n_windows = (20 - 5) // 2 + 1  # 8 windows
        assert windows.shape == (expected_n_windows, 5, 5, 8)
        
        # Test with step size 1
        windows_step1 = create_time_windows(X, window_size=3, step_size=1)
        expected_n_windows_step1 = 20 - 3 + 1  # 18 windows
        assert windows_step1.shape[0] == expected_n_windows_step1
        
        # Test invalid window size
        with pytest.raises(ValueError):
            create_time_windows(X, window_size=25)  # Larger than data


class TestValidationUtils:
    """Test suite for validation utilities."""
    
    def test_check_deflex_input(self):
        """Test Deflex input validation."""
        # Valid inputs
        X = np.random.randn(10, 5, 32).astype(np.float32)
        y = np.random.randn(10, 5, 1).astype(np.float32)
        
        X_val, y_val = check_deflex_input(X, y)
        assert X_val.shape == X.shape
        assert y_val.shape == y.shape
        
        # Test with only X
        X_val_only, y_val_only = check_deflex_input(X)
        assert X_val_only.shape == X.shape
        assert y_val_only is None
        
        # Test invalid shapes
        X_2d = np.random.randn(10, 32)
        with pytest.raises(ValueError):
            check_deflex_input(X_2d, ensure_3d=True)
            
        # Test mismatched lengths
        y_short = np.random.randn(5, 5, 1)
        with pytest.raises(ValueError):
            check_deflex_input(X, y_short)
            
        # Test NaN values
        X_nan = X.copy()
        X_nan[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            check_deflex_input(X_nan)
            
    def test_validate_operator_set(self):
        """Test operator set validation."""
        # Test with OperatorSet instance
        ops = DefaultOperators()
        validated = validate_operator_set(ops)
        assert validated is ops
        
        # Test with string
        validated_str = validate_operator_set('default')
        assert isinstance(validated_str, DefaultOperators)
        
        # Test with list
        op_list = ['+', '-', 'sin', 'cos']
        validated_list = validate_operator_set(op_list)
        assert hasattr(validated_list, 'to_julia_format')
        
        # Test invalid string
        with pytest.raises(ValueError):
            validate_operator_set('invalid_set')
            
        # Test invalid type
        with pytest.raises(ValueError):
            validate_operator_set(123)
            
    def test_validate_training_parameters(self):
        """Test training parameter validation."""
        # Valid parameters
        validate_training_parameters(
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            validation_split=0.1
        )  # Should not raise
        
        # Invalid epochs
        with pytest.raises(ValueError):
            validate_training_parameters(0, 32, 0.001, 0.1)
            
        # Invalid batch size
        with pytest.raises(ValueError):
            validate_training_parameters(100, -1, 0.001, 0.1)
            
        # Invalid learning rate
        with pytest.raises(ValueError):
            validate_training_parameters(100, 32, 0.0, 0.1)
            
        # Invalid validation split
        with pytest.raises(ValueError):
            validate_training_parameters(100, 32, 0.001, 1.5)
            
    def test_check_random_state(self):
        """Test random state validation."""
        # Valid cases
        assert check_random_state(None) is None
        assert check_random_state(42) == 42
        
        # Invalid cases
        with pytest.raises(ValueError):
            check_random_state(-1)
            
        with pytest.raises(ValueError):
            check_random_state("invalid")


class TestEndToEndValidation:
    """End-to-end validation tests."""
    
    def test_complete_data_pipeline(self):
        """Test complete data processing pipeline."""
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(50, 8, 16)
        y = np.sum(X, axis=-1, keepdims=True) + 0.1 * np.random.randn(50, 8, 1)
        
        # Validate inputs
        X_val, y_val = check_deflex_input(X, y)
        
        # Prepare for training
        X_train, y_train, X_val_split, y_val_split = prepare_data_for_training(
            X_val, y_val, validation_split=0.2, normalize=True
        )
        
        # Create time windows
        windows = create_time_windows(X_train, window_size=5)
        
        # Reshape for symbolic regression
        X_symb = reshape_for_symbolic_regression(X_train)
        
        assert windows.shape[1] == 5  # Window size
        assert X_symb.shape == (len(X_train), 8 * 16)  # Flattened
        assert len(X_train) == 40  # 80% of 50
        assert len(X_val_split) == 10  # 20% of 50