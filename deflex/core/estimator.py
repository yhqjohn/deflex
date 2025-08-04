"""
Main Deflex estimator implementing the three-stage training process.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from typing import Optional, Dict, Any, Union

from ..deflexformer import DeflexformerEstimator
from ..lambda_regression import LambdaRegressionEstimator


class DeflexEstimator(BaseEstimator, RegressorMixin):
    """
    Deep Formula Discovery estimator combining neural networks and symbolic regression.
    
    This estimator implements a three-stage training process:
    1. Pre-training: Train a single Deflexformer block on synthetic data
    2. Post-training: Stack pre-trained blocks and fine-tune on experimental data  
    3. Symbolic regression: Extract symbolic formulas from the trained model
    
    Parameters
    ----------
    n_blocks : int, default=5
        Number of Deflexformer blocks to use in the model.
    embedding_dim : int, default=64
        Maximum embedding dimension for the Deflexformer.
    n_synthetic_samples : int, default=10000
        Number of synthetic samples to generate for pre-training.
    pretrain_epochs : int, default=100
        Number of epochs for pre-training stage.
    finetune_epochs : int, default=50
        Number of epochs for post-training/fine-tuning stage.
    operator_set : list, optional
        Set of operators for symbolic regression. If None, uses default set.
    lambda_regression_params : dict, optional
        Additional parameters for the LambdaRegression estimator.
    random_state : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    deflexformer_ : DeflexformerEstimator
        The trained Deflexformer model.
    lambda_regression_ : LambdaRegressionEstimator  
        The symbolic regression model.
    is_fitted_ : bool
        Whether the estimator has been fitted.
    formula_ : str
        The discovered symbolic formula (available after fitting).
    """
    
    def __init__(
        self,
        n_blocks: int = 5,
        embedding_dim: int = 64,
        n_synthetic_samples: int = 10000,
        pretrain_epochs: int = 100,
        finetune_epochs: int = 50,
        operator_set: Optional[list] = None,
        lambda_regression_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None
    ):
        self.n_blocks = n_blocks
        self.embedding_dim = embedding_dim
        self.n_synthetic_samples = n_synthetic_samples
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.operator_set = operator_set
        self.lambda_regression_params = lambda_regression_params or {}
        self.random_state = random_state
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DeflexEstimator':
        """
        Fit the Deflex model using the three-stage training process.
        
        Parameters
        ----------
        X : array-like of shape (n_frames, n_elements, embedding_dim)
            Training data as 3D tensor.
        y : array-like of shape (n_frames, n_elements, output_dim)
            Target values.
            
        Returns
        -------
        self : DeflexEstimator
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y, multi_output=True, allow_nd=True)
        
        if X.ndim != 3:
            raise ValueError(f"X must be 3D tensor, got shape {X.shape}")
        if y.ndim < 2:
            raise ValueError(f"y must be at least 2D, got shape {y.shape}")
            
        # Stage 1: Pre-training on synthetic data
        self._pretrain_stage()
        
        # Stage 2: Post-training on experimental data
        self._posttrain_stage(X, y)
        
        # Stage 3: Symbolic regression
        self._symbolic_regression_stage(X, y)
        
        self.is_fitted_ = True
        return self
        
    def _pretrain_stage(self) -> None:
        """Stage 1: Pre-train single block on synthetic data."""
        print("Stage 1: Pre-training on synthetic data...")
        
        # Initialize LambdaRegression for synthetic data generation
        self.lambda_regression_ = LambdaRegressionEstimator(
            operator_set=self.operator_set,
            random_state=self.random_state,
            **self.lambda_regression_params
        )
        
        # Generate synthetic data
        X_synthetic, y_synthetic = self.lambda_regression_.generate_synthetic_data(
            n_samples=self.n_synthetic_samples
        )
        
        # Initialize single-block Deflexformer for pre-training
        self.deflexformer_ = DeflexformerEstimator(
            n_blocks=1,  # Only one block for pre-training
            embedding_dim=self.embedding_dim,
            enable_clipping=True,  # Strict clipping for synthetic data
            random_state=self.random_state
        )
        
        # Pre-train the single block
        self.deflexformer_.fit(
            X_synthetic, y_synthetic, 
            epochs=self.pretrain_epochs,
            stage="pretrain"
        )
        
    def _posttrain_stage(self, X: np.ndarray, y: np.ndarray) -> None:
        """Stage 2: Stack blocks and fine-tune on experimental data.""" 
        print("Stage 2: Post-training on experimental data...")
        
        # Expand to full architecture with pre-trained weights
        self.deflexformer_.expand_to_full_model(self.n_blocks)
        
        # Fine-tune on experimental data
        self.deflexformer_.fit(
            X, y,
            epochs=self.finetune_epochs, 
            stage="posttrain"
        )
        
    def _symbolic_regression_stage(self, X: np.ndarray, y: np.ndarray) -> None:
        """Stage 3: Extract symbolic formulas."""
        print("Stage 3: Symbolic regression...")
        
        # Extract features from trained Deflexformer blocks
        block_outputs = self.deflexformer_.extract_block_outputs(X)
        
        # Generate additional synthetic data for symbolic regression
        X_additional, y_additional = self.lambda_regression_.generate_synthetic_data(
            n_samples=self.n_synthetic_samples // 2
        )
        
        # Combine experimental and synthetic data for symbolic regression
        X_combined = np.concatenate([block_outputs, X_additional], axis=0)
        y_combined = np.concatenate([y, y_additional], axis=0)
        
        # Perform symbolic regression
        self.lambda_regression_.fit(X_combined, y_combined)
        self.formula_ = self.lambda_regression_.get_formula()
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.
        
        Parameters
        ---------- 
        X : array-like of shape (n_frames, n_elements, embedding_dim)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_frames, n_elements, output_dim)
            Predicted values.
        """
        if not hasattr(self, 'is_fitted_'):
            raise ValueError("This DeflexEstimator instance is not fitted yet.")
            
        X = check_array(X, allow_nd=True)
        
        # Use symbolic formula if available, otherwise use Deflexformer
        if hasattr(self, 'formula_') and self.formula_:
            return self.lambda_regression_.predict(X)
        else:
            return self.deflexformer_.predict(X)
            
    def get_formula(self) -> str:
        """
        Get the discovered symbolic formula.
        
        Returns
        -------
        formula : str
            The symbolic formula in human-readable format.
        """
        if not hasattr(self, 'formula_'):
            raise ValueError("Model must be fitted before getting formula.")
        return self.formula_
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_frames, n_elements, embedding_dim)
            Test samples.
        y : array-like of shape (n_frames, n_elements, output_dim)  
            True values.
            
        Returns
        -------
        score : float
            R^2 coefficient of determination.
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, multioutput='uniform_average')