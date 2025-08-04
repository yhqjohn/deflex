"""
Operator sets for symbolic regression.
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod


class OperatorSet(ABC):
    """
    Abstract base class for operator sets used in symbolic regression.
    """
    
    @abstractmethod
    def to_julia_format(self) -> List[str]:
        """Convert operator set to Julia-compatible format."""
        pass
        
    @abstractmethod
    def get_arity_map(self) -> Dict[str, int]:
        """Get mapping of operators to their arity (number of arguments)."""
        pass


class DefaultOperators(OperatorSet):
    """
    Default operator set for symbolic regression.
    
    Includes basic arithmetic operations and common mathematical functions.
    """
    
    def __init__(self):
        self.binary_operators = ['+', '-', '*', '/']
        self.unary_operators = ['sin', 'cos', 'exp', 'log', 'abs', 'sqrt']
        
    def to_julia_format(self) -> List[str]:
        """Convert to Julia format."""
        julia_ops = []
        
        # Binary operators
        for op in self.binary_operators:
            if op == '+':
                julia_ops.append('+')
            elif op == '-':
                julia_ops.append('-') 
            elif op == '*':
                julia_ops.append('*')
            elif op == '/':
                julia_ops.append('/')
                
        # Unary operators  
        julia_ops.extend(self.unary_operators)
        
        return julia_ops
        
    def get_arity_map(self) -> Dict[str, int]:
        """Get operator arity mapping."""
        arity_map = {}
        
        # Binary operators have arity 2
        for op in self.binary_operators:
            arity_map[op] = 2
            
        # Unary operators have arity 1
        for op in self.unary_operators:
            arity_map[op] = 1
            
        return arity_map


class BasicOperators(OperatorSet):
    """
    Basic operator set with only arithmetic operations.
    """
    
    def __init__(self):
        self.binary_operators = ['+', '-', '*', '/']
        self.unary_operators = []
        
    def to_julia_format(self) -> List[str]:
        return self.binary_operators + self.unary_operators
        
    def get_arity_map(self) -> Dict[str, int]:
        arity_map = {}
        for op in self.binary_operators:
            arity_map[op] = 2
        for op in self.unary_operators:
            arity_map[op] = 1
        return arity_map


class ExtendedOperators(OperatorSet):
    """
    Extended operator set with more mathematical functions.
    """
    
    def __init__(self):
        self.binary_operators = ['+', '-', '*', '/', '^', 'max', 'min']
        self.unary_operators = [
            'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
            'exp', 'log', 'log10', 'abs', 'sqrt', 'square',
            'sign', 'floor', 'ceil'
        ]
        
    def to_julia_format(self) -> List[str]:
        return self.binary_operators + self.unary_operators
        
    def get_arity_map(self) -> Dict[str, int]:
        arity_map = {}
        for op in self.binary_operators:
            arity_map[op] = 2
        for op in self.unary_operators:
            arity_map[op] = 1
        return arity_map


class CustomOperators(OperatorSet):
    """
    Custom operator set allowing user-defined operators.
    """
    
    def __init__(
        self, 
        binary_operators: List[str] = None,
        unary_operators: List[str] = None
    ):
        self.binary_operators = binary_operators or ['+', '-', '*', '/']
        self.unary_operators = unary_operators or ['sin', 'cos', 'exp', 'log']
        
    def add_binary_operator(self, operator: str) -> None:
        """Add a binary operator to the set."""
        if operator not in self.binary_operators:
            self.binary_operators.append(operator)
            
    def add_unary_operator(self, operator: str) -> None:
        """Add a unary operator to the set."""
        if operator not in self.unary_operators:
            self.unary_operators.append(operator)
            
    def remove_operator(self, operator: str) -> None:
        """Remove an operator from the set."""
        if operator in self.binary_operators:
            self.binary_operators.remove(operator)
        if operator in self.unary_operators:
            self.unary_operators.remove(operator)
            
    def to_julia_format(self) -> List[str]:
        return self.binary_operators + self.unary_operators
        
    def get_arity_map(self) -> Dict[str, int]:
        arity_map = {}
        for op in self.binary_operators:
            arity_map[op] = 2
        for op in self.unary_operators:
            arity_map[op] = 1
        return arity_map


# Predefined operator sets for common use cases
μ_ARITHMETIC = BasicOperators()  # Using μ symbol as per user preference
μ_DEFAULT = DefaultOperators()
μ_EXTENDED = ExtendedOperators()

# For compatibility
ARITHMETIC_OPERATORS = μ_ARITHMETIC
DEFAULT_OPERATORS = μ_DEFAULT  
EXTENDED_OPERATORS = μ_EXTENDED


def get_all_operator_sets() -> Dict[str, OperatorSet]:
    """
    Get all available predefined operator sets.
    
    Returns
    -------
    operator_sets : dict
        Dictionary mapping names to operator set instances.
    """
    return {
        'arithmetic': μ_ARITHMETIC,
        'default': μ_DEFAULT,
        'extended': μ_EXTENDED,
        'basic': μ_ARITHMETIC,  # Alias
        'full': μ_EXTENDED      # Alias
    }