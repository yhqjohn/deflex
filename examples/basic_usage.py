"""
Basic usage example for Deflex symbolic regression system.

This example demonstrates the complete three-stage training process
using synthetic data and JSON input formats.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings

# Import Deflex components
try:
    from deflex import DeflexEstimator, DeflexformerEstimator, LambdaRegressionEstimator
    from deflex.utils.data import validate_tensor_shape, prepare_data_for_training
    from deflex.lambda_regression.operators import μ_DEFAULT, μ_EXTENDED
except ImportError as e:
    print(f"Please install Deflex package: {e}")
    exit(1)


def load_json_data(json_path: str) -> tuple:
    """Load spatial-temperature data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to 3D tensor format
    n_samples = len(data)
    X = np.zeros((n_samples, 1, 4))  # (samples, elements=1, features=4)
    
    for i, item in enumerate(data):
        X[i, 0, :3] = item['location']     # x, y, z coordinates
        X[i, 0, 3] = item['temperature']   # temperature
    
    # Create target: predict temperature from location
    y = X[:, :, 3:4]  # Temperature as target
    X_features = X[:, :, :3]  # Location as input features
    
    return X_features.astype(np.float32), y.astype(np.float32)


def load_csv_data(csv_path: str) -> tuple:
    """Load multi-element dynamics data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Extract number of elements from column names
    element_cols = [col for col in df.columns if col.startswith('element_')]
    n_elements = len(set(col.split('_')[1] for col in element_cols))
    n_features = len(element_cols) // n_elements
    n_frames = len(df)
    
    # Convert to 3D tensor
    X = np.zeros((n_frames, n_elements, n_features))
    
    for i in range(n_elements):
        for j, feature in enumerate(['x', 'y']):  # Assuming x, y features
            col_name = f'element_{i}_{feature}'
            if col_name in df.columns:
                X[:, i, j] = df[col_name]
    
    # Create target: predict next timestep
    if n_frames > 1:
        y = X[1:, :, :1]  # Next x coordinate
        X = X[:-1]        # Current state
    else:
        y = X[:, :, :1]   # Same timestep prediction
    
    return X.astype(np.float32), y.astype(np.float32)


def generate_synthetic_data(n_samples: int = 100) -> tuple:
    """Generate synthetic 3D tensor data for demonstration."""
    np.random.seed(42)
    
    # Create temporal dynamics: sine waves with different phases
    t = np.linspace(0, 4*np.pi, n_samples)
    n_elements = 3
    embedding_dim = 8
    
    X = np.zeros((n_samples, n_elements, embedding_dim))
    
    for elem in range(n_elements):
        phase = elem * np.pi / 3
        # Basic sine wave with harmonics
        X[:, elem, 0] = np.sin(t + phase)
        X[:, elem, 1] = np.cos(t + phase) 
        X[:, elem, 2] = 0.5 * np.sin(2*t + phase)
        X[:, elem, 3] = 0.3 * np.cos(3*t + phase)
        
        # Add some interaction terms
        X[:, elem, 4] = X[:, elem, 0] * X[:, elem, 1]
        X[:, elem, 5] = X[:, elem, 0] ** 2
        
        # Add noise to remaining features
        X[:, elem, 6:] = 0.1 * np.random.randn(n_samples, embedding_dim - 6)
    
    # Create target: simple formula involving first few features
    y = (X[:, :, 0:1] ** 2 + 
         0.5 * X[:, :, 1:2] + 
         0.1 * np.random.randn(n_samples, n_elements, 1))
    
    return X.astype(np.float32), y.astype(np.float32)


def demonstrate_deflexformer_only():
    """Demonstrate using only the Deflexformer neural network."""
    print("\nDeflexformer Neural Network Demo")
    print("=" * 50)
    
    # Generate data
    X, y = generate_synthetic_data(50)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create Deflexformer estimator
    deflexformer = DeflexformerEstimator(
        n_blocks=2,
        embedding_dim=8,
        n_heads=4,
        random_state=42
    )
    
    print("Training Deflexformer...")
    try:
        deflexformer.fit(X, y, epochs=5, batch_size=16)
        
        # Make predictions
        predictions = deflexformer.predict(X[:10])
        mse = np.mean((predictions - y[:10]) ** 2)
        print(f"MSE on first 10 samples: {mse:.4f}")
        
    except Exception as e:
        print(f"Note: Deflexformer requires PyTorch. Error: {e}")


def demonstrate_lambda_regression_only():
    """Demonstrate using only the LambdaRegression symbolic regression."""
    print("\n🔬 LambdaRegression Symbolic Regression Demo")
    print("=" * 50)
    
    # Generate simple 2D data for symbolic regression
    np.random.seed(42)
    X_2d = np.random.randn(20, 3)
    y_2d = X_2d[:, 0]**2 + 0.5*X_2d[:, 1] + 0.1*np.random.randn(20)
    
    print(f"2D Data shape: X={X_2d.shape}, y={y_2d.shape}")
    
    # Create LambdaRegression estimator
    lambda_reg = LambdaRegressionEstimator(
        operator_set=μ_DEFAULT,
        population_size=20,
        generations=10,
        random_state=42
    )
    
    print("Running symbolic regression...")
    try:
        lambda_reg.fit(X_2d, y_2d)
        formula = lambda_reg.get_formula()
        print(f"Discovered formula: {formula}")
        
        # Make predictions
        predictions = lambda_reg.predict(X_2d[:5])
        print(f"Sample predictions: {predictions[:3]}")
        
    except Exception as e:
        print(f"Note: LambdaRegression requires Julia. Using mock. Error: {e}")


def demonstrate_full_deflex():
    """Demonstrate the complete Deflex three-stage process."""
    print("\n🎯 Complete Deflex System Demo")
    print("=" * 50)
    
    # Generate data
    X, y = generate_synthetic_data(30)  # Smaller dataset for demo
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create main Deflex estimator
    estimator = DeflexEstimator(
        n_blocks=2,
        embedding_dim=8,
        pretrain_epochs=3,    # Reduced for demo
        finetune_epochs=2,    # Reduced for demo
        n_synthetic_samples=20,  # Reduced for demo
        operator_set=μ_DEFAULT,
        random_state=42
    )
    
    print("Starting three-stage Deflex training...")
    print("Stage 1: Pre-training on synthetic data...")
    print("Stage 2: Post-training on experimental data...")
    print("Stage 3: Symbolic regression...")
    
    try:
        estimator.fit(X, y)
        
        # Get discovered formula
        formula = estimator.get_formula()
        print(f"Discovered symbolic formula: {formula}")
        
        # Make predictions
        predictions = estimator.predict(X[:5])
        actual = y[:5]
        mse = np.mean((predictions - actual) ** 2)
        print(f"MSE on test samples: {mse:.4f}")
        
        # Score the model
        score = estimator.score(X, y)
        print(f"R² score: {score:.4f}")
        
    except Exception as e:
        print(f"Note: Full Deflex requires PyTorch and Julia. Error: {e}")
        print("This is expected in a basic installation.")


def demonstrate_data_loading():
    """Demonstrate loading data from files."""
    print("\nData Loading Demo")
    print("=" * 50)
    
    # Try to load JSON data
    data_dir = Path(__file__).parent.parent / "data"
    json_path = data_dir / "spatial_temperature.json"
    csv_path = data_dir / "synthetic_dynamics.csv"
    
    if json_path.exists():
        print("Loading JSON spatial-temperature data...")
        X_json, y_json = load_json_data(str(json_path))
        print(f"JSON data shape: X={X_json.shape}, y={y_json.shape}")
        print(f"Sample location: {X_json[0, 0, :3]}")
        print(f"Sample temperature: {y_json[0, 0, 0]:.2f}°C")
    
    if csv_path.exists():
        print("\nLoading CSV dynamics data...")
        X_csv, y_csv = load_csv_data(str(csv_path))
        print(f"CSV data shape: X={X_csv.shape}, y={y_csv.shape}")
        print(f"Sample element positions: {X_csv[0, :, :]}")


def main():
    """Run all demonstration examples."""
    print("Deflex Demonstration Suite")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Run individual component demos
    demonstrate_data_loading()
    demonstrate_deflexformer_only()
    demonstrate_lambda_regression_only()
    demonstrate_full_deflex()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("\nNote: Some components require PyTorch and Julia installations.")
    print("Install with: pip install torch julia")
    print("Julia setup: julia -e 'using Pkg; Pkg.add(\"LambdaRegression\")'")


if __name__ == "__main__":
    main()