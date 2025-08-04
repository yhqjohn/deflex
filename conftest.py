"""
Pytest configuration for Deflex test suite.
"""

import pytest
import numpy as np


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "julia: marks tests that require Julia"
    )
    config.addinivalue_line(
        "markers", "torch: marks tests that require PyTorch"
    )


@pytest.fixture
def sample_3d_data():
    """Provide sample 3D tensor data for testing."""
    np.random.seed(42)
    n_frames, n_elements, embedding_dim = 20, 5, 16
    
    X = np.random.randn(n_frames, n_elements, embedding_dim).astype(np.float32)
    y = np.sum(X[:, :, :4], axis=-1, keepdims=True).astype(np.float32)  # Simple target
    
    return X, y


@pytest.fixture 
def sample_spatial_data():
    """Provide sample spatial-temporal data."""
    data = [
        {"location": [0.0, 0.0, 0.0], "temperature": 20.0},
        {"location": [1.0, 1.0, 1.0], "temperature": 25.0},
        {"location": [2.0, 2.0, 2.0], "temperature": 30.0}
    ]
    return data


@pytest.fixture
def mock_julia_interface():
    """Provide a mock Julia interface for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.symbolic_regression.return_value = {
        'formula': 'x1 + x2',
        'model': MagicMock(),
        'complexity': 3
    }
    mock.generate_synthetic_data.return_value = {
        'y': np.random.randn(10)
    }
    mock.predict.return_value = np.random.randn(5)
    
    return mock