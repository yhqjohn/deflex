"""
LambdaRegression: Symbolic regression subsystem for the Deflex package.

Implements Julia FFI interface for symbolic regression with customizable
operator sets and formula generation capabilities.
"""

from .estimator import LambdaRegressionEstimator
from .julia_interface import JuliaInterface
from .operators import OperatorSet, DefaultOperators

__all__ = [
    "LambdaRegressionEstimator",
    "JuliaInterface", 
    "OperatorSet",
    "DefaultOperators"
]