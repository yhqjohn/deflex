"""
Data processing utilities for Deflex.
"""

import numpy as np
from typing import Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array


def validate_tensor_shape(
    X: np.ndarray, 
    expected_dims: int = 3,
    min_frames: int = 1,
    min_elements: int = 1,
    max_embedding_dim: int = 64
) -> None:
    """
    Validate that input tensor has the expected shape for Deflex.
    
    Parameters
    ----------
    X : ndarray
        Input tensor to validate.
    expected_dims : int, default=3
        Expected number of dimensions.
    min_frames : int, default=1
        Minimum number of frames (time steps).
    min_elements : int, default=1
        Minimum number of elements.
    max_embedding_dim : int, default=64
        Maximum embedding dimension.
        
    Raises
    ------
    ValueError
        If tensor shape is invalid.
    """
    if X.ndim != expected_dims:
        raise ValueError(
            f"Expected {expected_dims}D tensor, got {X.ndim}D with shape {X.shape}"
        )
        
    if expected_dims == 3:
        n_frames, n_elements, embedding_dim = X.shape
        
        if n_frames < min_frames:
            raise ValueError(f"Need at least {min_frames} frames, got {n_frames}")
            
        if n_elements < min_elements:
            raise ValueError(f"Need at least {min_elements} elements, got {n_elements}")
            
        if embedding_dim > max_embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding_dim} exceeds maximum {max_embedding_dim}"
            )


def prepare_data_for_training(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float = 0.1,
    normalize: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare data for training with optional normalization and validation split.
    
    Parameters
    ----------
    X : ndarray
        Input features.
    y : ndarray
        Target values.
    validation_split : float, default=0.1
        Fraction of data to use for validation.
    normalize : bool, default=True
        Whether to normalize the data.
    random_state : int, optional
        Random seed for shuffling.
        
    Returns
    -------
    X_train, y_train, X_val, y_val : ndarrays
        Training and validation data. Val arrays are None if validation_split=0.
    """
    # Validate inputs
    X = check_array(X, allow_nd=True)
    y = check_array(y, allow_nd=True)
    
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")
        
    # Shuffle data if random_state provided
    if random_state is not None:
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
    # Normalization
    if normalize:
        # For 3D tensors, normalize along the last dimension
        if X.ndim == 3:
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            scaler_X = StandardScaler()
            X_reshaped = scaler_X.fit_transform(X_reshaped)
            X = X_reshaped.reshape(original_shape)
        else:
            scaler_X = StandardScaler()
            X = scaler_X.fit_transform(X)
            
        # Normalize y if it's multi-dimensional
        if y.ndim > 1:
            if y.ndim == 3:
                original_shape_y = y.shape
                y_reshaped = y.reshape(-1, y.shape[-1])
                scaler_y = StandardScaler()
                y_reshaped = scaler_y.fit_transform(y_reshaped)
                y = y_reshaped.reshape(original_shape_y)
            else:
                scaler_y = StandardScaler()
                y = scaler_y.fit_transform(y)
                
    # Validation split
    if validation_split > 0:
        val_size = int(len(X) * validation_split)
        X_train = X[:-val_size]
        y_train = y[:-val_size]
        X_val = X[-val_size:]
        y_val = y[-val_size:]
    else:
        X_train = X
        y_train = y
        X_val = None
        y_val = None
        
    return X_train, y_train, X_val, y_val


def reshape_for_symbolic_regression(X: np.ndarray) -> np.ndarray:
    """
    Reshape 3D tensor data for symbolic regression (flatten to 2D).
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_frames, n_elements, embedding_dim)
        3D tensor data.
        
    Returns
    -------
    X_reshaped : ndarray of shape (n_samples, n_features)
        Flattened data for symbolic regression.
    """
    if X.ndim == 3:
        # Flatten frames and elements dimensions
        n_samples, n_frames, n_elements, embedding_dim = X.shape
        return X.reshape(n_samples, n_frames * n_elements * embedding_dim)
    elif X.ndim == 2:
        return X
    else:
        raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")


def create_time_windows(
    X: np.ndarray,
    window_size: int,
    step_size: int = 1
) -> np.ndarray:
    """
    Create sliding time windows from sequential data.
    
    Parameters
    ----------
    X : ndarray of shape (n_frames, n_elements, embedding_dim)
        Sequential data.
    window_size : int
        Size of each time window.
    step_size : int, default=1
        Step size for sliding window.
        
    Returns
    -------
    X_windowed : ndarray
        Windowed data of shape (n_windows, window_size, n_elements, embedding_dim).
    """
    if X.ndim != 3:
        raise ValueError("Input must be 3D tensor")
        
    n_frames, n_elements, embedding_dim = X.shape
    
    if window_size > n_frames:
        raise ValueError(f"Window size {window_size} larger than data length {n_frames}")
        
    # Calculate number of windows
    n_windows = (n_frames - window_size) // step_size + 1
    
    # Create windowed data
    windows = []
    for i in range(0, n_frames - window_size + 1, step_size):
        window = X[i:i + window_size]
        windows.append(window)
        
    return np.array(windows)