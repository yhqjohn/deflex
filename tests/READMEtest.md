# Deflex Test Suite

This directory contains comprehensive tests for the Deflex package.

## Test Structure

- `test_core.py`: Tests for the main DeflexEstimator and core functionality
- `test_deflexformer.py`: Tests for the neural network subsystem (requires PyTorch)
- `test_lambda_regression.py`: Tests for the symbolic regression subsystem (mocked Julia)
- `test_utils.py`: Tests for utility functions and data processing
- `test_data_loading.py`: Tests for loading and converting various data formats

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Test Files
```bash
# Core functionality only
pytest tests/test_core.py

# Data processing utilities
pytest tests/test_utils.py

# Data loading from CSV/JSON
pytest tests/test_data_loading.py
```

### Skip Slow Tests
```bash
pytest tests/ -m "not slow"
```

### Skip Integration Tests
```bash
pytest tests/ -m "not integration"
```

## Test Markers

- `@pytest.mark.slow`: Slow-running tests (can be skipped for quick testing)
- `@pytest.mark.integration`: Integration tests requiring multiple components
- `@pytest.mark.julia`: Tests requiring Julia installation
- `@pytest.mark.torch`: Tests requiring PyTorch installation

## Mock vs Real Dependencies

Most tests use mocked Julia and PyTorch components to avoid requiring full installations. 
Integration tests marked with appropriate markers test real functionality.

## Sample Data

The tests use sample data from the `data/` directory:
- `spatial_temperature.json`: JSON format with location and temperature data
- `synthetic_dynamics.csv`: CSV format with multi-element temporal data

## Test Coverage

Tests cover:
- Parameter validation
- Data format conversion (CSV, JSON → 3D tensors)
- Scikit-learn API compliance
- Error handling and edge cases
- Mock-based unit testing for complex dependencies