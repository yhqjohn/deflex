"""
Scikit-learn style estimator for the Deflexformer model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from typing import Optional, Union, Literal

from .models import DeflexformerModel


class DeflexformerEstimator(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible estimator for the Deflexformer model.
    
    A compact Transformer-like architecture designed for 3D tensor data
    with element-wise transformations and multi-head attention mechanisms.
    
    Parameters
    ----------
    n_blocks : int, default=5
        Number of Deflexformer blocks.
    embedding_dim : int, default=64
        Embedding dimension (maximum 64).
    n_heads : int, default=8
        Number of attention heads in multi-head attention.
    dropout : float, default=0.1
        Dropout probability.
    enable_clipping : bool, default=False
        Whether to enable gradient clipping for training stability.
    clip_value : float, default=1.0
        Maximum gradient norm for clipping.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda').
    random_state : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    model_ : DeflexformerModel
        The underlying PyTorch model.
    is_fitted_ : bool
        Whether the estimator has been fitted.
    """
    
    def __init__(
        self,
        n_blocks: int = 5,
        embedding_dim: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
        enable_clipping: bool = False,
        clip_value: float = 1.0,
        device: Optional[str] = None,
        random_state: Optional[int] = None
    ):
        if embedding_dim > 64:
            raise ValueError("embedding_dim must not exceed 64")
            
        self.n_blocks = n_blocks
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.enable_clipping = enable_clipping
        self.clip_value = clip_value
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        stage: Literal["pretrain", "posttrain"] = "posttrain",
        validation_split: float = 0.1
    ) -> 'DeflexformerEstimator':
        """
        Fit the Deflexformer model.
        
        Parameters
        ----------
        X : array-like of shape (n_frames, n_elements, embedding_dim)
            Training data.
        y : array-like of shape (n_frames, n_elements, output_dim)
            Target values.
        epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=32
            Batch size for training.
        learning_rate : float, default=1e-3
            Learning rate for optimization.
        stage : {"pretrain", "posttrain"}, default="posttrain"
            Training stage identifier.
        validation_split : float, default=0.1
            Fraction of data to use for validation.
            
        Returns
        -------
        self : DeflexformerEstimator
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y, multi_output=True, allow_nd=True)
        
        if X.ndim != 3:
            raise ValueError(f"X must be 3D tensor, got shape {X.shape}")
            
        n_frames, n_elements, input_dim = X.shape
        
        # Adjust embedding dimension if needed
        if input_dim != self.embedding_dim:
            self.embedding_dim = min(input_dim, 64)
            
        # Initialize model if not exists or if architecture changed
        if not hasattr(self, 'model_') or self.model_.embedding_dim != self.embedding_dim:
            output_dim = y.shape[-1] if y.ndim > 2 else 1
            self.model_ = DeflexformerModel(
                embedding_dim=self.embedding_dim,
                n_blocks=self.n_blocks,
                n_heads=self.n_heads,
                output_dim=output_dim,
                dropout=self.dropout
            ).to(self.device)
            
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Split validation data
        val_size = int(len(X) * validation_split)
        if val_size > 0:
            X_train, X_val = X_tensor[:-val_size], X_tensor[-val_size:]
            y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]
        else:
            X_train, X_val = X_tensor, None
            y_train, y_val = y_tensor, None
            
        # Training setup
        optimizer = optim.Adam(self.model_.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model_.train()
        for epoch in range(epochs):
            # Batch training
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping if enabled
                if self.enable_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_.parameters(), 
                        self.clip_value
                    )
                
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                
            # Validation
            if X_val is not None and epoch % 10 == 0:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_val)
                    val_loss = criterion(val_outputs, y_val)
                print(f"Epoch {epoch}: Train Loss: {total_loss/n_batches:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
                self.model_.train()
                
        self.is_fitted_ = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_frames, n_elements, embedding_dim)
            Input data.
            
        Returns
        -------
        y_pred : ndarray
            Predicted values.
        """
        if not hasattr(self, 'is_fitted_'):
            raise ValueError("This DeflexformerEstimator instance is not fitted yet.")
            
        X = check_array(X, allow_nd=True)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor)
            
        return predictions.cpu().numpy()
        
    def expand_to_full_model(self, n_blocks: int) -> None:
        """
        Expand a single-block model to full architecture.
        
        Used in the pre-training to post-training transition.
        
        Parameters
        ----------
        n_blocks : int
            Number of blocks in the full model.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be initialized first.")
            
        if self.model_.n_blocks == n_blocks:
            return  # Already correct size
            
        # Save the pre-trained block
        pretrained_block = self.model_.blocks[0]
        
        # Create new full model
        self.model_ = DeflexformerModel(
            embedding_dim=self.embedding_dim,
            n_blocks=n_blocks,
            n_heads=self.n_heads,
            output_dim=self.model_.output_projection.out_features,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize all blocks with pre-trained weights
        for block in self.model_.blocks:
            block.load_state_dict(pretrained_block.state_dict())
            
        self.n_blocks = n_blocks
        
    def extract_block_outputs(self, X: np.ndarray) -> np.ndarray:
        """
        Extract intermediate outputs from each block.
        
        Used for symbolic regression stage.
        
        Parameters
        ----------
        X : array-like of shape (n_frames, n_elements, embedding_dim)
            Input data.
            
        Returns
        -------
        block_outputs : ndarray
            Concatenated outputs from all blocks.
        """
        if not hasattr(self, 'is_fitted_'):
            raise ValueError("Model must be fitted first.")
            
        X = check_array(X, allow_nd=True)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model_.eval()
        block_outputs = []
        
        with torch.no_grad():
            x = X_tensor
            for block in self.model_.blocks:
                x = block(x)
                block_outputs.append(x.cpu().numpy())
                
        # Concatenate along feature dimension
        return np.concatenate(block_outputs, axis=-1)