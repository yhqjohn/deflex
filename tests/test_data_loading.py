"""
Tests for data loading and processing from various formats.
"""

import pytest
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

from deflex.utils.data import validate_tensor_shape, prepare_data_for_training


class TestDataLoading:
    """Test suite for loading different data formats."""
    
    @pytest.fixture
    def data_dir(self):
        """Get the data directory path."""
        # Assuming tests are run from project root
        return Path(__file__).parent.parent / "data"
    
    def test_load_csv_data(self, data_dir):
        """Test loading CSV data and converting to 3D tensor."""
        csv_path = data_dir / "synthetic_dynamics.csv"
        
        if not csv_path.exists():
            pytest.skip("CSV data file not found")
            
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Convert to 3D tensor format
        # Assuming CSV has columns: frame, element_0_x, element_0_y, element_1_x, element_1_y, element_2_x, element_2_y
        n_frames = len(df)
        n_elements = 3
        n_features = 2  # x, y coordinates
        
        # Reshape data to (frames, elements, features)
        X = np.zeros((n_frames, n_elements, n_features))
        for i in range(n_elements):
            X[:, i, 0] = df[f'element_{i}_x']
            X[:, i, 1] = df[f'element_{i}_y']
            
        # Validate tensor shape
        validate_tensor_shape(X, expected_dims=3, max_embedding_dim=64)
        
        assert X.shape == (n_frames, n_elements, n_features)
        assert not np.isnan(X).any()
        
    def test_load_json_spatial_data(self, data_dir):
        """Test loading JSON spatial-temperature data."""
        json_path = data_dir / "spatial_temperature.json"
        
        if not json_path.exists():
            pytest.skip("JSON data file not found")
            
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Convert to Deflex format
        n_samples = len(data)
        
        # Extract locations (3D coordinates) and temperatures
        locations = np.array([item['location'] for item in data])  # (n_samples, 3)
        temperatures = np.array([item['temperature'] for item in data])  # (n_samples,)
        
        # Convert to 3D tensor format (samples, elements=1, features=4)
        # Features: [x, y, z, temperature]
        X = np.zeros((n_samples, 1, 4))
        X[:, 0, :3] = locations  # Location coordinates
        X[:, 0, 3] = temperatures  # Temperature as 4th feature
        
        # Target could be temperature prediction
        y = temperatures.reshape(-1, 1, 1)
        
        # Validate
        validate_tensor_shape(X)
        assert X.shape == (n_samples, 1, 4)
        assert y.shape == (n_samples, 1, 1)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
        
    def test_temporal_spatial_conversion(self, data_dir):
        """Test converting spatial data to temporal sequences."""
        json_path = data_dir / "spatial_temperature.json"
        
        if not json_path.exists():
            pytest.skip("JSON data file not found")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Simulate temporal data by treating each JSON entry as a time step
        n_frames = len(data)
        n_elements = 1  # One sensor/element per frame
        n_features = 4  # x, y, z, temperature
        
        # Create 3D tensor: (frames, elements, features)
        X = np.zeros((n_frames, n_elements, n_features))
        
        for t, item in enumerate(data):
            X[t, 0, :3] = item['location']  # x, y, z coordinates
            X[t, 0, 3] = item['temperature']  # temperature
            
        # Create target: predict next temperature
        y = X[1:, 0:1, 3:4]  # Next temperature as target
        X_input = X[:-1]  # Input without last frame
        
        assert X_input.shape == (n_frames - 1, n_elements, n_features)
        assert y.shape == (n_frames - 1, n_elements, 1)
        
    def test_deflex_workflow_with_json_data(self, data_dir):
        """Test complete Deflex workflow with JSON data."""
        json_path = data_dir / "spatial_temperature.json"
        
        if not json_path.exists():
            pytest.skip("JSON data file not found")
            
        # Load and prepare data
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        # Convert to tensor format
        n_samples = len(raw_data)
        X = np.zeros((n_samples, 1, 4))  # 1 element, 4 features
        
        for i, item in enumerate(raw_data):
            X[i, 0, :3] = item['location']
            X[i, 0, 3] = item['temperature']
            
        # Create synthetic target (temperature prediction from location)
        y = X[:, :, 3:4]  # Use temperature as target
        
        # Prepare for training
        X_train, y_train, X_val, y_val = prepare_data_for_training(
            X, y, validation_split=0.0, normalize=True
        )
        
        assert X_train.shape[0] == n_samples
        assert y_train.shape == (n_samples, 1, 1)
        
    def test_batch_json_processing(self):
        """Test processing multiple JSON objects in batch."""
        # Simulate batch of spatial-temporal data
        batch_data = [
            {"location": [1.0, 2.0, 3.0], "temperature": 25.0, "timestamp": 0},
            {"location": [1.1, 2.1, 3.1], "temperature": 25.5, "timestamp": 1},
            {"location": [1.2, 2.2, 3.2], "temperature": 26.0, "timestamp": 2},
        ]
        
        # Convert to Deflex format
        n_frames = len(batch_data)
        X = np.zeros((n_frames, 1, 5))  # location(3) + temperature(1) + timestamp(1)
        
        for i, item in enumerate(batch_data):
            X[i, 0, :3] = item['location']
            X[i, 0, 3] = item['temperature']
            X[i, 0, 4] = item['timestamp']
            
        # Validate
        validate_tensor_shape(X, max_embedding_dim=64)
        assert X.shape == (3, 1, 5)
        
        # Test that we can extract temporal patterns
        temp_sequence = X[:, 0, 3]  # Temperature over time
        assert len(temp_sequence) == 3
        assert temp_sequence[0] < temp_sequence[-1]  # Temperature increasing
        
    def test_multi_element_json_data(self):
        """Test handling JSON data with multiple elements per frame."""
        # Simulate multi-sensor data
        frame_data = {
            "timestamp": 0,
            "sensors": [
                {"id": "sensor_1", "location": [0, 0, 0], "temperature": 20.0},
                {"id": "sensor_2", "location": [10, 0, 0], "temperature": 22.0},
                {"id": "sensor_3", "location": [0, 10, 0], "temperature": 21.0}
            ]
        }
        
        # Convert to Deflex format
        n_elements = len(frame_data["sensors"])
        X = np.zeros((1, n_elements, 4))  # 1 frame, 3 elements, 4 features each
        
        for i, sensor in enumerate(frame_data["sensors"]):
            X[0, i, :3] = sensor['location']
            X[0, i, 3] = sensor['temperature']
            
        validate_tensor_shape(X)
        assert X.shape == (1, 3, 4)
        
        # Test element-wise processing
        avg_temp = np.mean(X[0, :, 3])
        assert abs(avg_temp - 21.0) < 0.1  # Average should be around 21


class TestDataFormatValidation:
    """Test data format validation for different input types."""
    
    def test_json_schema_validation(self):
        """Test validation of JSON data schemas."""
        # Valid schema
        valid_data = [
            {"location": [1.0, 2.0, 3.0], "temperature": 25.0},
            {"location": [4.0, 5.0, 6.0], "temperature": 30.0}
        ]
        
        # Validate structure
        for item in valid_data:
            assert "location" in item
            assert "temperature" in item
            assert len(item["location"]) == 3
            assert isinstance(item["temperature"], (int, float))
            
        # Invalid schema - missing temperature
        invalid_data = [{"location": [1.0, 2.0, 3.0]}]
        
        with pytest.raises(KeyError):
            for item in invalid_data:
                temp = item["temperature"]  # Should raise KeyError
                
    def test_coordinate_range_validation(self):
        """Test validation of coordinate ranges."""
        # Test data with extreme coordinates
        extreme_data = [
            {"location": [1000.0, -1000.0, 500.0], "temperature": 25.0}
        ]
        
        # Convert and validate
        X = np.array([[extreme_data[0]["location"] + [extreme_data[0]["temperature"]]]])
        
        # Check for reasonable coordinate ranges (this is domain-specific)
        locations = X[0, 0, :3]
        assert np.all(np.abs(locations) < 10000)  # Example reasonable range