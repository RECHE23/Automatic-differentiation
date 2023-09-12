from __future__ import annotations
from functools import reduce
from typing import Union, SupportsFloat, List, Callable, Tuple, Dict
import math


class Variable:
    """
    A class representing a variable for automatic differentiation.

    Parameters:
    ----------
    value : SupportsFloat
        The initial value of the variable.
    gradient_fn : Callable[[], List[Tuple[Variable, float]]], optional
        A function that returns a list of tuples, each containing a Variable and its gradient with respect to this variable.

    Attributes:
    ----------
    value : float
        The current value of the variable.
    gradient_fn : Callable[[], List[Tuple[Variable, float]]]
        The gradient function for this variable.
    grads : Dict[Variable, float]
        A dictionary containing gradients with respect to this variable.
    """

    def __init__(self, value: SupportsFloat, gradient_fn: Callable[[], List[Tuple[Variable, float]]] = lambda: []):
        self.value = float(value)
        self.gradient_fn = gradient_fn
        self.grads = self.compute_gradients()

    def __add__(self, other: Union[Variable, SupportsFloat]) -> Variable:
        """
        Addition operator for variables.

        Parameters:
        ----------
        other : Variable or SupportsFloat
            The variable or constant to add to this variable.

        Returns:
        ----------
        Variable
            A new Variable representing the result of the addition.
        """
        other_var = Variable(other) if isinstance(other, SupportsFloat) else other
        return Variable(value=self.value + other_var.value, gradient_fn=lambda: [(self, 1.0), (other_var, 1.0)])

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        """
        Negation operator for variables.

        Returns:
        ----------
        Variable
            A new Variable representing the negation of this variable.
        """
        return Variable(value=-self.value, gradient_fn=lambda: [(self, -1.0)])

    def __sub__(self, other):
        """
        Subtraction operator for variables.

        Parameters:
        ----------
        other : Variable or SupportsFloat
            The variable or constant to subtract from this variable.

        Returns:
        ----------
        Variable
            A new Variable representing the result of the subtraction.
        """
        other_var = Variable(other) if isinstance(other, SupportsFloat) else other
        return Variable(value=self.value - other_var.value, gradient_fn=lambda: [(self, 1.0), (other_var, -1.0)])

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other: Union[Variable, SupportsFloat]) -> Variable:
        """
        Multiplication operator for variables.

        Parameters:
        ----------
        other : Variable or SupportsFloat
            The variable or constant to multiply with this variable.

        Returns:
        ----------
        Variable
            A new Variable representing the result of the multiplication.
        """
        other_var = Variable(other) if isinstance(other, SupportsFloat) else other
        return Variable(value=self.value * other_var.value, gradient_fn=lambda: [(self, other_var.value), (other_var, self.value)])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Union[Variable, SupportsFloat]) -> Variable:
        """
        Division operator for variables.

        Parameters:
        ----------
        other : Variable or SupportsFloat
            The variable or constant to divide this variable by.

        Returns:
        ----------
        Variable
            A new Variable representing the result of the division.
        """
        other_var = Variable(other) if isinstance(other, SupportsFloat) else other
        return Variable(value=self.value / other_var.value,
                        gradient_fn=lambda: [(self, 1 / other_var.value), (other_var, - self.value / other_var.value ** 2)])

    def __rtruediv__(self, other):
        other_var = Variable(other) if isinstance(other, SupportsFloat) else other
        return Variable(value=other_var.value / self.value,
                        gradient_fn=lambda: [(self, - other_var.value / self.value ** 2), (other_var, 1 / self.value)])

    def __pow__(self, exponent: Union[Variable, SupportsFloat]) -> Variable:
        """
        Exponentiation operator for variables.

        Parameters:
        ----------
        exponent : Variable or SupportsFloat
            The exponent to raise this variable to.

        Returns:
        ----------
        Variable
            A new Variable representing the result of the exponentiation.
        """
        exponent_var = Variable(exponent) if isinstance(exponent, SupportsFloat) else exponent
        return Variable(value=self.value ** exponent_var.value,
                        gradient_fn=lambda: [(self, self.value ** (exponent_var.value - 1) * exponent_var.value),
                                             (exponent_var, self.value ** exponent_var.value * math.log(self.value))])

    def __rpow__(self, base: Union[Variable, SupportsFloat]) -> Variable:
        base_var = Variable(base) if isinstance(base, SupportsFloat) else base
        return Variable(value=base_var.value ** self.value,
                        gradient_fn=lambda: [(self, base_var.value ** self.value * math.log(base_var.value)),
                                             (base_var, base_var.value ** (self.value - 1) * self.value)])

    def compute_gradients(self, *variables: Variable, backpropagation: float = 1.0) -> Dict[Variable, float]:
        """
        Compute gradients of the variable.

        Parameters:
        ----------
        variables : Variable, optional
            Variables with respect to which gradients are calculated.
        backpropagation : float, optional
            The value to backpropagate for computing gradients.

        Returns:
        ----------
        Dict[Variable, float]
            A dictionary containing gradients with respect to specified variables.
        """
        if variables:
            return {var: self.grads[var] for var in variables}

        return reduce(
            lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)},
            [var.compute_gradients(*variables, backpropagation=val * backpropagation) for var, val in self.gradient_fn()],
            {self: backpropagation}
        )
