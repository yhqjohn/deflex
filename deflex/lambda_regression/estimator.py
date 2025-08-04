"""
Scikit-learn style estimator for symbolic regression via Julia FFI.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from typing import Optional, List, Dict, Any, Union

from .julia_interface import JuliaInterface
from .operators import DefaultOperators


class LambdaRegressionEstimator(BaseEstimator, RegressorMixin):
    """
    Symbolic regression estimator using Julia LambdaRegression.jl backend.
    
    This estimator provides a scikit-learn compatible interface to the Julia
    LambdaRegression package for discovering symbolic formulas.
    
    Parameters
    ----------
    operator_set : list or OperatorSet, optional
        Set of operators to use for symbolic regression. 
        If None, uses default operators.
    population_size : int, default=100
        Size of the genetic programming population.
    generations : int, default=50
        Number of generations for evolution.
    max_complexity : int, default=10
        Maximum complexity (depth) of generated formulas.
    parsimony_coefficient : float, default=0.01
        Penalty for formula complexity.
    tournament_size : int, default=3
        Tournament size for selection.
    mutation_rate : float, default=0.1
        Probability of mutation.
    crossover_rate : float, default=0.8
        Probability of crossover.
    random_state : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    julia_ : JuliaInterface
        Julia interface for FFI calls.
    formula_ : str
        The best discovered formula in human-readable format.
    is_fitted_ : bool
        Whether the estimator has been fitted.
    """
    
    def __init__(
        self,
        operator_set: Optional[Union[List[str], 'OperatorSet']] = None,
        population_size: int = 100,
        generations: int = 50,
        max_complexity: int = 10,
        parsimony_coefficient: float = 0.01,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        random_state: Optional[int] = None
    ):
        self.operator_set = operator_set or DefaultOperators()
        self.population_size = population_size
        self.generations = generations
        self.max_complexity = max_complexity
        self.parsimony_coefficient = parsimony_coefficient
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state
        
        # Initialize Julia interface
        self.julia_ = JuliaInterface()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LambdaRegressionEstimator':
        """
        Fit the symbolic regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.
            
        Returns
        -------
        self : LambdaRegressionEstimator
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y, multi_output=True)
        
        # Flatten input if 3D (handle Deflexformer output)
        if X.ndim > 2:
            original_shape = X.shape
            X = X.reshape(X.shape[0], -1)
            self._input_shape = original_shape
        else:
            self._input_shape = X.shape
            
        # Configure Julia parameters
        params = {
            'operator_set': self._format_operators(),
            'population_size': self.population_size,
            'generations': self.generations,
            'max_complexity': self.max_complexity,
            'parsimony_coefficient': self.parsimony_coefficient,
            'tournament_size': self.tournament_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'random_state': self.random_state
        }
        
        # Call Julia symbolic regression
        result = self.julia_.symbolic_regression(X, y, params)
        
        # Store results
        self.formula_ = result['formula']
        self._julia_model = result['model']
        self._feature_names = [f"x{i}" for i in range(X.shape[1])]
        
        self.is_fitted_ = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted symbolic model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        """
        if not hasattr(self, 'is_fitted_'):
            raise ValueError("This LambdaRegressionEstimator instance is not fitted yet.")
            
        X = check_array(X)
        
        # Reshape if needed to match training shape
        if hasattr(self, '_input_shape') and X.shape != self._input_shape[1:]:
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
                
        # Use Julia model for prediction
        predictions = self.julia_.predict(self._julia_model, X)
        
        return predictions
        
    def generate_synthetic_data(
        self, 
        n_samples: int = 1000,
        n_features: Optional[int] = None,
        noise_level: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data using random formulas.
        
        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        n_features : int, optional
            Number of features. If None, uses a reasonable default.
        noise_level : float, default=0.1
            Amount of noise to add to the data.
            
        Returns
        -------
        X : ndarray
            Generated input data.
        y : ndarray
            Generated target data.
        """
        if n_features is None:
            n_features = 3  # Default number of features
            
        # Generate random input data
        X = np.random.randn(n_samples, n_features)
        
        # Use Julia to generate synthetic formulas and evaluate
        params = {
            'operator_set': self._format_operators(),
            'n_samples': n_samples,
            'n_features': n_features,
            'noise_level': noise_level,
            'random_state': self.random_state
        }
        
        result = self.julia_.generate_synthetic_data(X, params)
        y = result['y']
        
        return X, y
        
    def get_formula(self) -> str:
        """
        Get the discovered symbolic formula.
        
        Returns
        -------
        formula : str
            Human-readable symbolic formula.
        """
        if not hasattr(self, 'formula_'):
            raise ValueError("Model must be fitted before getting formula.")
        return self.formula_
        
    def get_complexity(self) -> int:
        """
        Get the complexity of the discovered formula.
        
        Returns
        -------
        complexity : int
            Formula complexity (number of nodes in expression tree).
        """
        if not hasattr(self, '_julia_model'):
            raise ValueError("Model must be fitted before getting complexity.")
        return self.julia_.get_complexity(self._julia_model)
        
    def _format_operators(self) -> List[str]:
        """Format operator set for Julia interface."""
        if hasattr(self.operator_set, 'to_julia_format'):
            return self.operator_set.to_julia_format()
        elif isinstance(self.operator_set, list):
            return self.operator_set
        else:
            return list(self.operator_set)
            
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values.
            
        Returns
        -------
        score : float
            R^2 coefficient of determination.
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, multioutput='uniform_average')