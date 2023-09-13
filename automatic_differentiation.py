from __future__ import annotations
from functools import reduce
from typing import Set, Union, SupportsFloat, Callable, Tuple, Dict
import math


class Variable:
    """
    A class representing a variable for automatic differentiation.

    Parameters:
    ----------
    name : str
        The name of the variable.
    value : SupportsFloat, optional
        The initial value of the variable.
    variables : Set[Variable], optional
        A set of variables that this variable depends on.
    value_fn : Callable[[], float], optional
        A function that returns the current value of the variable.
    gradient_fn : Callable[[], Tuple[Tuple[Variable, float], ...]], optional
        A function that returns a tuple of (variable, gradient) pairs.

    Attributes:
    ----------
    name : str
        The name of the variable.
    variables : Set[Variable]
        A set of variables that this variable depends on.
    value : float
        The current value of the variable.
    value_fn : Callable[[], float]
        A function that returns the current value of the variable.
    gradient_fn : Callable[[], Tuple[Tuple[Variable, float], ...]]
        A function that returns a tuple of (variable, gradient) pairs.
    """

    def __init__(self, name: str, value: SupportsFloat = None, variables: Set[Variable] = None,
                 value_fn: Callable[[], float] = None, gradient_fn: Callable[[], Tuple[Tuple[Variable, float], ...]] = lambda: []):
        self.name = name
        self.variables = {self} if variables is None else variables
        self._value = value
        self.value_fn = value_fn if value_fn is not None else lambda: self.value
        self.gradient_fn = gradient_fn

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: SupportsFloat):
        self._value = value

    @property
    def grads(self) -> Dict[Variable, float]:
        """
        Get the gradients of the variable.

        Returns:
        ----------
        Dict[Variable, float]
            A dictionary containing gradients with respect to this variable.
        """
        assert all(v.value is not None for v in self.variables), "An evaluation of the formula must be done before trying to read the grads."
        return self.compute_gradients()

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
        other_var = Variable(name=str(other), value=other) if isinstance(other, SupportsFloat) else other
        variables = self.variables.union(other_var.variables)

        def value_fn():
            return self.value_fn() + other_var.value_fn()

        def gradient_fn():
            return (self, 1.0), (other_var, 1.0)

        return Variable(name=f"({self.name} + {other_var.name})", variables=variables, value_fn=value_fn, gradient_fn=gradient_fn)

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

        def value_fn():
            return -self.value_fn()

        def gradient_fn():
            return ((self, -1.0), )

        return Variable(name=f"(-{self.name})", variables=self.variables, value_fn=value_fn, gradient_fn=gradient_fn)

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
        other_var = Variable(name=str(other), value=other) if isinstance(other, SupportsFloat) else other
        variables = self.variables.union(other_var.variables)

        def value_fn():
            return self.value_fn() - other_var.value_fn()

        def gradient_fn():
            return (self, 1.0), (other_var, -1.0)

        return Variable(name=f"({self.name} - {other_var.name})", variables=variables, value_fn=value_fn, gradient_fn=gradient_fn)

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
        other_var = Variable(name=str(other), value=other) if isinstance(other, SupportsFloat) else other
        variables = self.variables.union(other_var.variables)

        def value_fn():
            return self.value_fn() * other_var.value_fn()

        def gradient_fn():
            return (self, other_var.value_fn()), (other_var, self.value_fn())

        return Variable(name=f"{self.name} * {other_var.name}", variables=variables, value_fn=value_fn, gradient_fn=gradient_fn)

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
        other_var = Variable(name=str(other), value=other) if isinstance(other, SupportsFloat) else other
        variables = self.variables.union(other_var.variables)

        def value_fn():
            return self.value_fn() / other_var.value_fn()

        def gradient_fn():
            return (self, 1 / other_var.value_fn()), (other_var, - self.value_fn() / other_var.value_fn() ** 2)

        return Variable(name=f"{self.name} / {other_var.name}", variables=variables, value_fn=value_fn, gradient_fn=gradient_fn)

    def __rtruediv__(self, other):
        other_var = Variable(name=str(other), value=other) if isinstance(other, SupportsFloat) else other
        variables = self.variables.union(other_var.variables)

        def value_fn():
            return other_var.value_fn() / self.value_fn()

        def gradient_fn():
            return (self, - other_var.value_fn() / self.value_fn() ** 2), (other_var, 1 / self.value_fn())

        return Variable(name=f"{other_var.name} / {self.name}", variables=variables, value_fn=value_fn, gradient_fn=gradient_fn)

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
        exponent_var = Variable(name=str(exponent), value=exponent) if isinstance(exponent, SupportsFloat) else exponent
        variables = self.variables.union(exponent_var.variables)

        def value_fn():
            return self.value_fn() ** exponent_var.value_fn()

        def gradient_fn():
            return ((self, self.value_fn() ** (exponent_var.value_fn() - 1) * exponent_var.value_fn()),
                    (exponent_var, self.value_fn() ** exponent_var.value_fn() * math.log(self.value_fn())))

        return Variable(name=f"{self.name} ** {exponent_var.name}", variables=variables, value_fn=value_fn, gradient_fn=gradient_fn)

    def __rpow__(self, base: Union[Variable, SupportsFloat]) -> Variable:
        base_var = Variable(name=str(base), value=base) if isinstance(base, SupportsFloat) else base
        variables = self.variables.union(base_var.variables)

        def value_fn():
            return base_var.value_fn() ** self.value_fn()

        def gradient_fn():
            return ((self, base_var.value_fn() ** self.value_fn() * math.log(base_var.value_fn())),
                    (base_var, base_var.value_fn() ** (self.value_fn() - 1) * self.value_fn()))

        return Variable(name=f"{base_var.name} ** {self.name}", variables=variables, value_fn=value_fn, gradient_fn=gradient_fn)

    def evaluate(self, variable_assignments: Dict[Variable, SupportsFloat]) -> float:
        """
        Evaluate the variable with given variable assignments.

        Parameters:
        ----------
        variable_assignments : Dict[Variable, SupportsFloat]
            A dictionary containing variable assignments.

        Returns:
        ----------
        float
            The result of the evaluation.
        """
        for k, v in variable_assignments.items():
            k.value = v

        return self.value_fn()

    def compute_gradients(self, variables: Dict[Variable, float] = None, backpropagation: float = 1.0) -> Dict[Variable, float]:
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
        if variables is not None:
            self.evaluate(variables)

        return reduce(
            lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)},
            [var.compute_gradients(variables, backpropagation=val * backpropagation) for var, val in self.gradient_fn()],
            {self: backpropagation}
        )


# Example usage
if __name__ == "__main__":
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')

    formula = (x + y) * (x - y) / (x ** z)
    print(f"f(x, y, z) = {formula}")                               # Displays the formula

    evaluation = formula.evaluate({x: 2, y: 3, z: 4})
    print(f"f({x.value}, {y.value}, {z.value}) = {evaluation}")    # Evaluation of the expression

    grads = formula.grads
    print(f"∂f({x.value}, {y.value}, {z.value})/∂x = {grads[x]}")  # Gradient with respect to x
    print(f"∂f({x.value}, {y.value}, {z.value})/∂y = {grads[y]}")  # Gradient with respect to y
    print(f"∂f({x.value}, {y.value}, {z.value})/∂z = {grads[z]}")  # Gradient with respect to z
