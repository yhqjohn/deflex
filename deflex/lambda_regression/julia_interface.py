"""
Julia FFI interface for LambdaRegression symbolic regression.
"""

import numpy as np
from julia import Julia, Pkg
from typing import Dict, Any, List, Optional
import warnings


class JuliaInterface:
    """
    Interface for calling Julia LambdaRegression functions via PyJulia.
    
    This class manages the Julia environment and provides methods for
    symbolic regression operations.
    """
    
    def __init__(self, julia_project_path: Optional[str] = None):
        """
        Initialize Julia interface.
        
        Parameters
        ----------
        julia_project_path : str, optional
            Path to Julia project environment. If None, uses current directory.
        """
        self._julia = None
        self._initialized = False
        self.julia_project_path = julia_project_path or "."
        self._initialize_julia()
        
    def _initialize_julia(self) -> None:
        """Initialize Julia environment and load packages."""
        try:
            # Initialize Julia with compiled modules disabled for better compatibility
            self._julia = Julia(compiled_modules=False)
            
            # Activate project environment
            self._julia.eval(f'using Pkg; Pkg.activate("{self.julia_project_path}")')
            
            # Install and load required packages
            self._ensure_packages_installed()
            
            # Load LambdaRegression
            self._julia.eval('using LambdaRegression')
            
            self._initialized = True
            
        except Exception as e:
            warnings.warn(f"Failed to initialize Julia interface: {e}")
            self._initialized = False
            
    def _ensure_packages_installed(self) -> None:
        """Ensure required Julia packages are installed."""
        required_packages = [
            "LambdaRegression",
            "SymbolicRegression", 
            "Random",
            "Statistics"
        ]
        
        for package in required_packages:
            try:
                # Try to use the package first
                self._julia.eval(f'using {package}')
            except:
                # If that fails, try to install it
                try:
                    self._julia.eval(f'Pkg.add("{package}")')
                    self._julia.eval(f'using {package}')
                except Exception as e:
                    warnings.warn(f"Could not install Julia package {package}: {e}")
                    
    def symbolic_regression(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform symbolic regression using Julia backend.
        
        Parameters
        ----------
        X : ndarray
            Input features.
        y : ndarray  
            Target values.
        params : dict
            Parameters for symbolic regression.
            
        Returns
        -------
        result : dict
            Dictionary containing 'formula' and 'model' keys.
        """
        if not self._initialized:
            raise RuntimeError("Julia interface not properly initialized")
            
        try:
            # Convert numpy arrays to Julia arrays
            self._julia.X = X
            self._julia.y = y
            
            # Set parameters
            for key, value in params.items():
                setattr(self._julia, key, value)
                
            # Run symbolic regression
            julia_code = """
            using LambdaRegression
            
            # Configure options
            options = SymbolicRegression.Options(
                binary_operators=[+, -, *, /],
                unary_operators=[sin, cos, exp, log],
                populations=population_size,
                ncycles_per_iteration=generations,
                maxsize=max_complexity,
                parsimony=parsimony_coefficient,
                tournament_selection_p=1.0/tournament_size,
                mutation_weights=[mutation_rate, crossover_rate],
                seed=random_state
            )
            
            # Run symbolic regression
            hall_of_fame = equation_search(X, y, options=options)
            
            # Get best equation
            best_equation = hall_of_fame.members[end]
            formula_str = string_tree(best_equation.tree, options)
            
            result = Dict(
                "formula" => formula_str,
                "model" => best_equation,
                "complexity" => compute_complexity(best_equation.tree, options)
            )
            """
            
            result = self._julia.eval(julia_code)
            
            return {
                'formula': str(result['formula']),
                'model': result['model'],
                'complexity': int(result['complexity'])
            }
            
        except Exception as e:
            raise RuntimeError(f"Julia symbolic regression failed: {e}")
            
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a fitted Julia model.
        
        Parameters
        ----------
        model : Julia object
            Fitted symbolic regression model.
        X : ndarray
            Input features.
            
        Returns
        -------
        predictions : ndarray
            Predicted values.
        """
        if not self._initialized:
            raise RuntimeError("Julia interface not properly initialized")
            
        try:
            # Set variables in Julia
            self._julia.model = model
            self._julia.X_pred = X
            
            # Make predictions
            julia_code = """
            using LambdaRegression
            predictions = model(X_pred)
            """
            
            predictions = self._julia.eval(julia_code)
            return np.array(predictions)
            
        except Exception as e:
            raise RuntimeError(f"Julia prediction failed: {e}")
            
    def generate_synthetic_data(
        self, 
        X: np.ndarray, 
        params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic data using random formulas.
        
        Parameters
        ----------
        X : ndarray
            Input features template.
        params : dict
            Generation parameters.
            
        Returns
        -------
        result : dict
            Dictionary with 'y' key containing synthetic targets.
        """
        if not self._initialized:
            raise RuntimeError("Julia interface not properly initialized")
            
        try:
            # Set variables
            self._julia.X_template = X
            for key, value in params.items():
                setattr(self._julia, key, value)
                
            # Generate synthetic data
            julia_code = """
            using LambdaRegression, Random
            
            # Set random seed if provided
            if random_state !== nothing
                Random.seed!(random_state)
            end
            
            # Generate random formula
            options = SymbolicRegression.Options(
                binary_operators=[+, -, *, /],
                unary_operators=[sin, cos, exp],
                maxsize=6  # Keep formulas simple for synthetic data
            )
            
            # Create random expression tree
            tree = gen_random_tree_fixed_size(6, options, n_features, Int32)
            
            # Evaluate on input data
            y_synthetic = eval_tree_array(tree, X_template', options)
            
            # Add noise
            noise = randn(size(y_synthetic)) * noise_level
            y_synthetic = y_synthetic + noise
            
            result = Dict("y" => y_synthetic)
            """
            
            result = self._julia.eval(julia_code)
            return {'y': np.array(result['y'])}
            
        except Exception as e:
            # Fallback to simple synthetic data
            warnings.warn(f"Julia synthetic data generation failed: {e}. Using fallback.")
            return self._fallback_synthetic_data(X, params)
            
    def _fallback_synthetic_data(
        self, 
        X: np.ndarray, 
        params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Fallback synthetic data generation using numpy."""
        np.random.seed(params.get('random_state'))
        
        # Simple synthetic function: y = x1^2 + sin(x2) + x3
        if X.shape[1] >= 3:
            y = X[:, 0]**2 + np.sin(X[:, 1]) + X[:, 2]
        elif X.shape[1] == 2:
            y = X[:, 0]**2 + np.sin(X[:, 1])
        else:
            y = X[:, 0]**2
            
        # Add noise
        noise_level = params.get('noise_level', 0.1)
        y += np.random.randn(len(y)) * noise_level
        
        return {'y': y}
        
    def get_complexity(self, model: Any) -> int:
        """
        Get the complexity of a Julia model.
        
        Parameters
        ----------
        model : Julia object
            Symbolic regression model.
            
        Returns
        -------
        complexity : int
            Model complexity.
        """
        if not self._initialized:
            return 0
            
        try:
            self._julia.model = model
            complexity = self._julia.eval("compute_complexity(model.tree, options)")
            return int(complexity)
        except:
            return 0