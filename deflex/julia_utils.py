"""
Legacy Julia utilities for Deflex.

This module provides basic Julia FFI functionality. For new code,
prefer using the JuliaInterface class from lambda_regression.julia_interface.
"""

from julia import Julia
from julia import Pkg
import warnings


# Initialize Julia environment
jl = Julia(compiled_modules=False)
jl.eval('using Pkg; Pkg.activate(".")')  # Activate current project environment


def install_julia_package(package_name: str = "LambdaRegression") -> bool:
    """
    Install a Julia package.
    
    Parameters
    ----------
    package_name : str, default="LambdaRegression"
        Name of the Julia package to install.
        
    Returns
    -------
    success : bool
        Whether installation was successful.
    """
    try:
        jl.eval(f'Pkg.add("{package_name}")')
        return True
    except Exception as e:
        warnings.warn(f"Failed to install Julia package {package_name}: {e}")
        return False


def call_lambda_regression(data, formula: str = None):
    """
    Call LambdaRegression with given data.
    
    Parameters
    ----------
    data : array-like
        Input data for symbolic regression.
    formula : str, optional
        Formula to evaluate. If None, performs symbolic regression.
        
    Returns
    -------
    result : any
        Result from Julia LambdaRegression.
        
    Note
    ----
    This is a legacy function. For new code, use LambdaRegressionEstimator.
    """
    warnings.warn(
        "call_lambda_regression is deprecated. Use LambdaRegressionEstimator instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        if formula is None:
            return jl.eval('using LambdaRegression; LambdaRegression.process(data)')
        else:
            jl.data = data
            jl.formula_str = formula
            return jl.eval('using LambdaRegression; LambdaRegression.evaluate(formula_str, data)')
    except Exception as e:
        raise RuntimeError(f"Julia call failed: {e}")


def setup_julia_environment(project_path: str = ".") -> bool:
    """
    Set up Julia environment for Deflex.
    
    Parameters
    ----------
    project_path : str, default="."
        Path to Julia project.
        
    Returns
    -------
    success : bool
        Whether setup was successful.
    """
    try:
        jl.eval(f'using Pkg; Pkg.activate("{project_path}")')
        
        # Install required packages
        required_packages = ["LambdaRegression", "SymbolicRegression"]
        for package in required_packages:
            install_julia_package(package)
            
        return True
    except Exception as e:
        warnings.warn(f"Julia environment setup failed: {e}")
        return False