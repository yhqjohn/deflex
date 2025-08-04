"""
Validation utilities for Deflex estimators and data.
"""

import numpy as np
from typing import Union, List, Any
from sklearn.utils.validation import check_array

from ..lambda_regression.operators import OperatorSet


def check_deflex_input(
    X: np.ndarray,
    y: np.ndarray = None,
    ensure_3d: bool = True,
    max_embedding_dim: int = 64
) -> tuple:
    """
    Validate input data for Deflex estimators.
    
    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like, optional
        Target values.
    ensure_3d : bool, default=True
        Whether to ensure X is 3D tensor.
    max_embedding_dim : int, default=64
        Maximum allowed embedding dimension.
        
    Returns
    -------
    X_validated : ndarray
        Validated input features.
    y_validated : ndarray or None
        Validated target values (None if y was None).
        
    Raises
    ------
    ValueError
        If validation fails.
    """
    # Validate X
    X = check_array(X, allow_nd=True, dtype=np.float32)
    
    if ensure_3d and X.ndim != 3:
        raise ValueError(
            f"Expected 3D tensor (frames, elements, embedding_dim), got {X.ndim}D with shape {X.shape}"
        )
        
    if X.ndim == 3:
        n_frames, n_elements, embedding_dim = X.shape
        
        if n_frames < 1:
            raise ValueError(f"Need at least 1 frame, got {n_frames}")
            
        if n_elements < 1:
            raise ValueError(f"Need at least 1 element, got {n_elements}")
            
        if embedding_dim > max_embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding_dim} exceeds maximum {max_embedding_dim}"
            )
            
    # Check for NaN or infinite values
    if not np.isfinite(X).all():
        raise ValueError("Input X contains NaN or infinite values")
        
    # Validate y if provided
    y_validated = None
    if y is not None:
        y_validated = check_array(y, allow_nd=True, dtype=np.float32)
        
        if len(X) != len(y_validated):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y_validated)}")
            
        if not np.isfinite(y_validated).all():
            raise ValueError("Target y contains NaN or infinite values")
            
    return X, y_validated


def validate_operator_set(operator_set: Union[List[str], OperatorSet, str]) -> OperatorSet:
    """
    Validate and convert operator set to OperatorSet instance.
    
    Parameters
    ----------
    operator_set : list, OperatorSet, or str
        Operator set specification.
        
    Returns
    -------
    validated_set : OperatorSet
        Validated operator set instance.
        
    Raises
    ------
    ValueError
        If operator set is invalid.
    """
    from ..lambda_regression.operators import (
        DefaultOperators, BasicOperators, ExtendedOperators, CustomOperators
    )
    
    if isinstance(operator_set, OperatorSet):
        return operator_set
    elif isinstance(operator_set, str):
        # String specification
        from ..lambda_regression.operators import get_all_operator_sets
        available_sets = get_all_operator_sets()
        if operator_set.lower() in available_sets:
            return available_sets[operator_set.lower()]
        else:
            raise ValueError(
                f"Unknown operator set '{operator_set}'. "
                f"Available: {list(available_sets.keys())}"
            )
    elif isinstance(operator_set, list):
        # List of operator strings
        return CustomOperators(
            binary_operators=[op for op in operator_set if _is_binary_operator(op)],
            unary_operators=[op for op in operator_set if _is_unary_operator(op)]
        )
    else:
        raise ValueError(
            f"operator_set must be list, OperatorSet instance, or string, got {type(operator_set)}"
        )


def _is_binary_operator(op: str) -> bool:
    """Check if operator is binary."""
    binary_ops = ['+', '-', '*', '/', '^', 'max', 'min', 'pow']
    return op in binary_ops


def _is_unary_operator(op: str) -> bool:
    """Check if operator is unary."""
    unary_ops = [
        'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
        'exp', 'log', 'log10', 'log2', 'abs', 'sqrt', 
        'square', 'sign', 'floor', 'ceil', 'round'
    ]
    return op in unary_ops


def validate_training_parameters(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float
) -> None:
    """
    Validate training parameters for neural network models.
    
    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    validation_split : float
        Validation split fraction.
        
    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"epochs must be positive integer, got {epochs}")
        
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be positive integer, got {batch_size}")
        
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive number, got {learning_rate}")
        
    if not 0 <= validation_split < 1:
        raise ValueError(f"validation_split must be in [0, 1), got {validation_split}")


def check_random_state(random_state: Any) -> Union[int, None]:
    """
    Validate random state parameter.
    
    Parameters
    ----------
    random_state : any
        Random state specification.
        
    Returns
    -------
    validated_state : int or None
        Validated random state.
        
    Raises
    ------
    ValueError
        If random_state is invalid.
    """
    if random_state is None:
        return None
    elif isinstance(random_state, int):
        if random_state < 0:
            raise ValueError("random_state must be non-negative integer")
        return random_state
    else:
        raise ValueError(f"random_state must be int or None, got {type(random_state)}")