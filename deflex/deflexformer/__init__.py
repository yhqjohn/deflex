"""
Deflexformer: Neural network subsystem for the Deflex package.

Implements a compact Transformer-like architecture for processing 
3D tensors with temporal and element-wise attention mechanisms.
"""

from .estimator import DeflexformerEstimator
from .models import DeflexformerModel
from .blocks import DeflexformerBlock

__all__ = [
    "DeflexformerEstimator",
    "DeflexformerModel", 
    "DeflexformerBlock"
]