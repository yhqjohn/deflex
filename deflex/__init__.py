"""
Deflex: Deep Formula Discovery for Complex Systems

A symbolic regression system combining neural networks (Deflexformer) 
and symbolic regression (LambdaRegression) for discovering formulas 
in complex dynamical systems.
"""

from .core.estimator import DeflexEstimator
from .deflexformer import DeflexformerEstimator
from .lambda_regression import LambdaRegressionEstimator

__version__ = "0.1.0"
__author__ = "Deflex Development Team"

# Export main estimators following scikit-learn conventions
__all__ = [
    "DeflexEstimator",
    "DeflexformerEstimator", 
    "LambdaRegressionEstimator"
]
