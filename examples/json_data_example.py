"""
Example: Working with JSON spatial-temporal data in Deflex.

This example shows how to load, process, and analyze spatial-temporal
data in JSON format using the Deflex framework.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

try:
    from deflex import DeflexEstimator
    from deflex.utils.data import validate_tensor_shape, create_time_windows
    from deflex.utils.validation import check_deflex_input
except ImportError:
    print("Please install Deflex: pip install deflex")
    exit(1)


def load_spatial_temporal_json(json_path: str) -> tuple:
    """
    Load spatial-temporal data from JSON file.
    
    Expected JSON format:
    [
        {"location": [x, y, z], "temperature": value, "timestamp": t},
        ...
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    n_samples = len(data)
    
    # Determine features based on first sample
    first_item = data[0]
    location_dim = len(first_item['location'])
    
    # Features: location coordinates + temperature (+ timestamp if available)
    feature_dim = location_dim + 1  # location + temperature
    if 'timestamp' in first_item:
        feature_dim += 1
    
    # Convert to 3D tensor: (samples, elements=1, features)
    X = np.zeros((n_samples, 1, feature_dim))
    
    for i, item in enumerate(data):
        feature_idx = 0
        
        # Location coordinates
        X[i, 0, :location_dim] = item['location']
        feature_idx += location_dim
        
        # Temperature
        X[i, 0, feature_idx] = item['temperature']
        feature_idx += 1
        
        # Timestamp (if available)
        if 'timestamp' in item:
            X[i, 0, feature_idx] = item['timestamp']
    
    return X.astype(np.float32)


def create_temperature_prediction_task(X: np.ndarray) -> tuple:
    """
    Create a temperature prediction task from spatial data.
    
    Input: Location coordinates
    Target: Temperature values
    """
    # Assume last feature is temperature, others are location/time
    location_features = X[:, :, :-1]  # All except temperature
    temperature = X[:, :, -1:]        # Temperature only
    
    return location_features, temperature


def create_temporal_prediction_task(X: np.ndarray, window_size: int = 3) -> tuple:
    """
    Create a temporal prediction task using sliding windows.
    
    Input: Historical spatial-temperature data
    Target: Next timestep temperature
    """
    if len(X) < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} samples for temporal prediction")
    
    # Create windows of historical data
    X_windows = create_time_windows(X, window_size=window_size)
    
    # Target is next timestep temperature
    y_next = X[window_size:, :, -1:]  # Temperature from next timestep
    
    return X_windows, y_next


def analyze_spatial_patterns(X: np.ndarray, temperatures: np.ndarray):
    """Analyze spatial patterns in the data."""
    print("\n📊 Spatial Pattern Analysis")
    print("-" * 30)
    
    locations = X[:, 0, :3]  # Assuming first 3 features are x,y,z
    temps = temperatures[:, 0, 0]  # Temperature values
    
    print(f"Number of spatial points: {len(locations)}")
    print(f"Temperature range: {temps.min():.2f}°C to {temps.max():.2f}°C")
    print(f"Location bounds:")
    print(f"  X: [{locations[:, 0].min():.2f}, {locations[:, 0].max():.2f}]")
    print(f"  Y: [{locations[:, 1].min():.2f}, {locations[:, 1].max():.2f}]")
    print(f"  Z: [{locations[:, 2].min():.2f}, {locations[:, 2].max():.2f}]")
    
    # Calculate distance from origin
    distances = np.linalg.norm(locations, axis=1)
    print(f"Distance from origin: {distances.min():.2f} to {distances.max():.2f}")
    
    # Simple correlation analysis
    corr_x_temp = np.corrcoef(locations[:, 0], temps)[0, 1]
    corr_y_temp = np.corrcoef(locations[:, 1], temps)[0, 1]
    corr_z_temp = np.corrcoef(locations[:, 2], temps)[0, 1]
    corr_dist_temp = np.corrcoef(distances, temps)[0, 1]
    
    print(f"\nCorrelations with temperature:")
    print(f"  X coordinate: {corr_x_temp:.3f}")
    print(f"  Y coordinate: {corr_y_temp:.3f}")
    print(f"  Z coordinate: {corr_z_temp:.3f}")
    print(f"  Distance from origin: {corr_dist_temp:.3f}")


def demonstrate_json_workflow():
    """Demonstrate complete workflow with JSON data."""
    print("🌡️ JSON Spatial-Temporal Data Analysis with Deflex")
    print("=" * 60)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    json_path = data_dir / "spatial_temperature.json"
    
    if not json_path.exists():
        print(f"❌ Data file not found: {json_path}")
        print("Please run from the project root directory.")
        return
    
    print(f"📁 Loading data from: {json_path}")
    X_spatial = load_spatial_temporal_json(str(json_path))
    
    print(f"✅ Loaded data shape: {X_spatial.shape}")
    validate_tensor_shape(X_spatial)
    
    # Analyze spatial patterns
    X_loc, y_temp = create_temperature_prediction_task(X_spatial)
    analyze_spatial_patterns(X_loc, y_temp)
    
    # Demonstrate Deflex usage
    print("\n🎯 Training Deflex Model")
    print("-" * 30)
    
    try:
        # Create estimator with appropriate settings for small dataset
        estimator = DeflexEstimator(
            n_blocks=1,  # Simple for small data
            embedding_dim=X_loc.shape[-1],  # Match feature dimension
            pretrain_epochs=5,
            finetune_epochs=3,
            n_synthetic_samples=10,  # Small for demo
            random_state=42
        )
        
        print("Training model (this may take a moment)...")
        estimator.fit(X_loc, y_temp)
        
        # Get results
        formula = estimator.get_formula()
        print(f"✅ Discovered formula: {formula}")
        
        # Make predictions
        predictions = estimator.predict(X_loc)
        mse = np.mean((predictions - y_temp) ** 2)
        r2 = estimator.score(X_loc, y_temp)
        
        print(f"✅ Model performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R² score: {r2:.4f}")
        
        # Show prediction vs actual
        print(f"\n📈 Sample predictions vs actual:")
        for i in range(min(3, len(predictions))):
            pred_val = predictions[i, 0, 0]
            actual_val = y_temp[i, 0, 0]
            print(f"  Sample {i+1}: Predicted={pred_val:.2f}°C, Actual={actual_val:.2f}°C")
    
    except Exception as e:
        print(f"⚠️  Model training failed (this is expected without full dependencies):")
        print(f"   {e}")
        print("\n💡 To run full functionality, install:")
        print("   pip install torch julia")
        print("   julia -e 'using Pkg; Pkg.add(\"LambdaRegression\")'")


def create_sample_json_data():
    """Create a sample JSON file with more complex spatial-temporal data."""
    print("\n📝 Creating extended sample data...")
    
    # Generate a more interesting spatial pattern
    np.random.seed(42)
    n_points = 20
    
    # Create a spiral pattern in 3D
    t = np.linspace(0, 4*np.pi, n_points)
    data = []
    
    for i, time in enumerate(t):
        x = np.cos(time) * (1 + 0.1*time)
        y = np.sin(time) * (1 + 0.1*time)
        z = 0.2 * time
        
        # Temperature function: depends on position and time
        temp_base = 20  # Base temperature
        temp_spatial = 5 * np.sin(x) + 3 * np.cos(y) + 2 * z
        temp_temporal = 2 * np.sin(0.5 * time)
        temperature = temp_base + temp_spatial + temp_temporal + np.random.normal(0, 0.5)
        
        data.append({
            "location": [float(x), float(y), float(z)],
            "temperature": float(temperature),
            "timestamp": float(time)
        })
    
    # Save to file
    output_path = Path(__file__).parent.parent / "data" / "extended_spatial_temp.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Created extended dataset: {output_path}")
    print(f"   Points: {n_points}, Features: location(3) + temperature + timestamp")
    
    return str(output_path)


def main():
    """Run the JSON data demonstration."""
    demonstrate_json_workflow()
    
    # Optionally create extended sample data
    print("\n" + "=" * 60)
    extended_path = create_sample_json_data()
    
    print(f"\n💡 Try the analysis with extended data:")
    print(f"   python examples/json_data_example.py {extended_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Use provided JSON file
        json_file = sys.argv[1]
        if Path(json_file).exists():
            print(f"📁 Using data file: {json_file}")
            # Load and analyze the provided file
            # (Implementation would go here)
        else:
            print(f"❌ File not found: {json_file}")
    else:
        main()