# Deflex: Deep Formula Discovery for Complex Systems

A symbolic regression system that combines neural networks (Deflexformer) and symbolic regression (LambdaRegression) to discover mathematical formulas in complex dynamical systems.

## Overview

Deflex is designed to work with 3D tensor data representing temporal dynamics:
- **Dimension 1**: Time frames
- **Dimension 2**: Elements/entities  
- **Dimension 3**: Individual embeddings

The system consists of two main components:

### Deflexformer
A compact Transformer-like neural network (≤64 embedding dimension) with specialized attention mechanisms:
- Element-wise transformations via small FFNs
- Frame-wise multi-head attention (across elements within same frame)
- Element-wise causal multi-head attention (across time for same element)
- Residual connections and LayerNorm throughout

### LambdaRegression  
Symbolic regression subsystem implemented via Julia FFI:
- Customizable operator sets and formula generation
- Efficient sampling and evaluation of mathematical expressions
- Scikit-learn compatible API

## Installation

```bash
pip install deflex
```

### Julia Dependencies
Deflex requires Julia with the LambdaRegression.jl package:

```bash
# Install Julia (1.8+ recommended)
# Then install required packages
julia -e 'using Pkg; Pkg.add("LambdaRegression")'
```

## Quick Start

```python
import numpy as np
from deflex import DeflexEstimator

# Generate sample 3D tensor data
# Shape: (n_frames, n_elements, embedding_dim)
X = np.random.randn(100, 10, 32)  
y = np.random.randn(100, 10, 1)

# Create and fit Deflex estimator
estimator = DeflexEstimator(
    n_blocks=5,
    embedding_dim=32,
    pretrain_epochs=50,
    finetune_epochs=25
)

# Three-stage training process
estimator.fit(X, y)

# Get discovered formula
formula = estimator.get_formula()
print(f"Discovered formula: {formula}")

# Make predictions
predictions = estimator.predict(X)
```

## Three-Stage Training Process

1. **Pre-training**: Train single Deflexformer block on synthetic data from LambdaRegression
2. **Post-training**: Stack pre-trained blocks and fine-tune on experimental data
3. **Symbolic regression**: Extract symbolic formulas from trained model

## API Reference

### Core Estimators

- `DeflexEstimator`: Main estimator combining both subsystems
- `DeflexformerEstimator`: Neural network subsystem only
- `LambdaRegressionEstimator`: Symbolic regression subsystem only

### Operator Sets

```python
from deflex.lambda_regression import DefaultOperators, ExtendedOperators

# Use different operator sets
estimator = DeflexEstimator(operator_set=ExtendedOperators())
```

## Examples

See the `examples/` directory for:
- Advanced configuration options  
- Integration with existing ML pipelines
- Custom operator definitions

## License

MIT License - see `LICENSE` file for details.

