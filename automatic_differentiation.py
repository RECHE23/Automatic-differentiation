from __future__ import annotations

import math
import re
from functools import reduce, partial
from typing import List, Optional, Set, SupportsFloat, Callable, Tuple, Dict

import numpy as np
from opt_einsum import contract, parser

# This allows the declaration of the functions in OPERATIONS at runtime:
global erf, neg, erfc, sinh, asin, log10, log, atan, sin, asinh, acos, cos, sqrt, acosh, abs, tan, cosh, tanh, exp, cbrt, atanh

OPERATIONS = {
    # Unary mathematical operations:
    'unary': {
        'neg': (np.negative, lambda val: -np.ones_like(val)),
        'abs': (np.abs, lambda val: np.sign(val)),
        'exp': (np.exp, lambda val: np.exp(val)),
        'log': (np.log, lambda val: 1.0 / val),
        'log10': (np.log10, lambda val: 1.0 / (val * np.log(10.0))),
        'sin': (np.sin, lambda val: np.cos(val)),
        'asin': (np.arcsin, lambda val: 1.0 / np.sqrt(1 - val ** 2)),
        'cos': (np.cos, lambda val: -np.sin(val)),
        'acos': (np.arccos, lambda val: -1.0 / np.sqrt(1.0 - val ** 2.0)),
        'tan': (np.tan, lambda val: 1.0 / np.cos(val) ** 2.0),
        'atan': (np.arctan, lambda val: 1.0 / (1.0 + val ** 2.0)),
        'sinh': (np.sinh, lambda val: np.cosh(val)),
        'asinh': (np.arcsinh, lambda val: 1.0 / np.sqrt(1.0 + val ** 2.0)),
        'cosh': (np.cosh, lambda val: np.sinh(val)),
        'acosh': (np.arccosh, lambda val: 1.0 / np.sqrt(val ** 2.0 - 1.0)),
        'tanh': (np.tanh, lambda val: 1.0 / np.cosh(val) ** 2.0),
        'atanh': (np.arctanh, lambda val: 1.0 / (1.0 - val ** 2.0)),
        'sqrt': (np.sqrt, lambda val: 0.5 / np.sqrt(val)),
        'cbrt': (np.cbrt, lambda val: 1.0 / (3.0 * val ** (2.0 / 3.0))),
        'erf': (np.vectorize(math.erf), lambda val: 2.0 * np.exp(-val ** 2.0) / np.sqrt(np.pi)),
        'erfc': (np.vectorize(math.erfc), lambda val: -2.0 * np.exp(-val ** 2.0) / np.sqrt(np.pi))
    },
    # Binary mathematical operations:
    'binary': {
        '+': (np.add, lambda l_val, r_val: (np.ones_like(l_val), np.ones_like(r_val))),
        '-': (np.subtract, lambda l_val, r_val: (np.ones_like(l_val), -np.ones_like(r_val))),
        '*': (np.multiply, lambda l_val, r_val: (r_val, l_val)),
        '/': (np.divide, lambda l_val, r_val: (1 / r_val, - l_val / r_val ** 2)),
        '@': (np.matmul, lambda l_val, r_val: (r_val.T, l_val.T)),
        '**': (np.power, lambda l_val, r_val: (l_val ** (r_val - 1) * r_val, l_val ** r_val * np.log(l_val))),
    },
    # Priority of operations (0 being the highest priority):
    'priority': {
        '**': 0,
        'neg': 1,
        'abs': 1,
        'exp': 1,
        'log': 1,
        'log10': 1,
        'sin': 1,
        'asin': 1,
        'cos': 1,
        'acos': 1,
        'tan': 1,
        'atan': 1,
        'sinh': 1,
        'asinh': 1,
        'cosh': 1,
        'acosh': 1,
        'tanh': 1,
        'atanh': 1,
        'sqrt': 1,
        'cbrt': 1,
        'erf': 1,
        'erfc': 1,
        'einsum': 1,
        '*': 2,
        '/': 2,
        '@': 2,
        '+': 3,
        '-': 3}
}


class Variable:
    """
    Represents a variable in a computational graph.

    Parameters
    ----------
    name : str
        The name of the variable.
    value : float | np.ndarray, optional
        The initial value of the variable, by default None.
    value_fn : Callable[[], float | np.ndarray], optional
        A function that computes the value of the variable, by default None.
    gradient_fn : Callable[[float | np.ndarray], Tuple[Tuple[Variable, float | np.ndarray], ...]], optional
        A function that computes gradients, by default None.

    Attributes
    ----------
    variables : set
        A set containing all the variables appearing in the graph. This excludes instances of Node and Constant.
    name : str
        The name of the variable.
    id : str
        A unique identifier for the variable.

    Methods
    -------
    __add__(self, other)
        Perform addition with another variable or constant.
    __sub__(self, other)
        Perform subtraction with another variable or constant.
    __mul__(self, other)
        Perform multiplication with another variable or constant.
    __truediv__(self, other)
        Perform true division with another variable or constant.
    __matmul__(self, other)
        Perform matrix multiplication with another variable or constant.
    __pow__(self, other)
        Perform exponentiation with another variable or constant.
    __neg__(self)
        Negate the variable.
    __abs__(self)
        Compute the absolute value of the variable.
    evaluate_at(**variable_assignments)
        Evaluate the variable's value with specific variable assignments.
    compute_gradients(variable_assignments=None, backpropagation=None)
        Compute gradients for the variable.

    See Also
    --------
    Constant : Represents a constant variable.
    Node : Represents a node in the computation graph.
    Einsum : Represents an einsum operation.
    """
    variables: Set[Variable]
    name: str
    value_fn: Callable[[], float | np.ndarray]
    gradient_fn: Callable[[float | np.ndarray], Tuple[Tuple[Variable, float | np.ndarray], ...]]
    id: str

    def __init__(self, name: str, value: float | np.ndarray = None,
                 value_fn: Callable[[], float | np.ndarray] = None,
                 gradient_fn: Callable[[float | np.ndarray], Tuple[Tuple[Variable, float | np.ndarray], ...]] = None):
        self.variables = {self}
        self.name = name
        self._value = value
        self.value_fn = value_fn if value_fn is not None else lambda: self.value
        self.gradient_fn = gradient_fn if gradient_fn is not None else lambda backpropagation: []
        self.id = f"var_{id(self):x}"

    def __repr__(self) -> str:
        value_txt = f", value={self.value}" if self.value is not None else ""
        name_value_text = f"{self.__class__.__name__}(name='{self.name}'{value_txt}"
        if isinstance(self, Node):
            operands = [f"{n.name}" if isinstance(n, Constant) else f"'{n.name}'" for n in self.operands] or ", "
            operands_txt = ", ".join(operands)
            return f"{name_value_text}, operation='{self.operation}', operands=({operands_txt}))"
        return f"{name_value_text})"

    def __str__(self) -> str:
        return self.name

    @property
    def _graph(self) -> str:
        if isinstance(self, Node):
            shape = "octagon" if isinstance(self, Einsum) else "square"
            graph_text = f'  {self.id} [style=filled, shape={shape}, fillcolor=lavenderblush3, label="{self.operation}"];\n'
            graph_text += f"".join([f'  {self.id} -> {c.id};\n' for c in self.operands])
            graph_text += f"".join([c._graph for c in self.operands])
            return graph_text
        else:
            shape = ("circle" if len(self.name) <= 3 else "ellipse") if isinstance(self, Constant) else "doublecircle"
            fillcolor = "ivory3" if isinstance(self, Constant) else "lightsteelblue"
            return f'  {self.id} [style=filled, shape={shape}, fillcolor={fillcolor}, label="{self.name}"];\n'

    @property
    def graph(self) -> str:
        return f"digraph {{\n" \
               f"  fontsize=15\n" \
               f"  labelloc=\"t\"\n" \
               f"  label=\"Computational graph\"\n" \
               f"{self._graph}}}"

    @property
    def shape(self) -> Tuple[int, ...]:
        if not isinstance(self, Node) and self.value is not None and hasattr(self.value, 'shape'):
            return self.value.shape
        return self._shape if hasattr(self, '_shape') else ()

    @property
    def ndim(self) -> int:
        if not isinstance(self, Node) and self.value is not None and hasattr(self.value, 'ndim'):
            return self.value.ndim
        return self._ndim if hasattr(self, '_ndim') else 1

    @property
    def size(self) -> int:
        if not isinstance(self, Node) and self.value is not None and hasattr(self.value, 'size'):
            return self.value.size
        return self._size if hasattr(self, '_size') else 1

    @property
    def value(self) -> Optional[float | np.ndarray]:
        if self._value is None:
            return None
        if isinstance(self._value, np.ndarray):
            return self._value
        return float(self._value)

    @value.setter
    def value(self, value: float | np.ndarray) -> None:
        self._value = value
        if isinstance(value, np.ndarray):
            self._shape = value.shape

    @property
    def grads(self) -> Dict[Variable, float]:
        assert all(v.value is not None for v in self.variables),\
            "An evaluation of the formula must be done before trying to read the grads."
        return self.compute_gradients()

    def __add__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, "+")

    def __radd__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, "+")

    def __sub__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, "-")

    def __rsub__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, "-")

    def __mul__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, "*")

    def __rmul__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, "*")

    def __truediv__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, "/")

    def __rtruediv__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, "/")

    def __matmul__(self, other: Variable | np.ndarray) -> Node:
        return Node.binary_operation(self, other, "@")

    def __rmatmul__(self, other: Variable | np.ndarray) -> Node:
        return Node.binary_operation(other, self, "@")

    def __pow__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, "**")

    def __rpow__(self, other: Variable | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, "**")

    def __neg__(self) -> Node:
        return Node.unary_operation(self, "neg")

    def __abs__(self) -> Node:
        return Node.unary_operation(self, "abs")

    def evaluate_at(self, **variable_assignments: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the value of the variable with specific variable assignments.

        Parameters
        ----------
        **variable_assignments : float | np.ndarray
            Keyword arguments where the keys are variable names and the values are the assigned values.

        Returns
        -------
        float | np.ndarray
            The evaluated value of the variable after applying the variable assignments.

        Notes
        -----
        This method allows you to evaluate the value of the variable within a specific context by providing
        variable assignments as keyword arguments. It computes the value of the variable by substituting
        the assigned values for the corresponding variables in its expression.

        Examples
        --------
        >>> x, y, z = set_variables('x', 'y', 'z')
        >>> formula = exp((x + y) * (x - y) / (x ** z))
        >>> result = formula.evaluate_at(x=2, y=3, z=4)
        >>> print(result)
        """
        self._apply_variable_assignments(variable_assignments)
        self.value = self.value_fn()
        return self.value

    def compute_gradients(self, variable_assignments: Dict[Variable, float | np.ndarray] = None,
                          backpropagation: float | np.ndarray = None) -> Dict[Variable, float | np.ndarray]:
        """
        Compute gradients for the variable.

        Parameters
        ----------
        variable_assignments : dict, optional
            A dictionary where keys are Variable objects or variable names (str), and values are the assigned values.
            These assignments are used to compute gradients, by default None.
        backpropagation : float | np.ndarray, optional
            The backpropagation value used for gradient computation, by default None.

        Returns
        -------
        dict
            A dictionary where keys are Variable objects, and values are the computed gradients.

        Notes
        -----
        This method computes gradients of the variable with respect to all other variables in the computation graph.
        It allows you to perform backpropagation to compute gradients in a reverse mode.

        Examples
        --------
        >>> x, y, z = set_variables('x', 'y', 'z')
        >>> formula = exp((x + y) * (x - y) / (x ** z))
        >>> grads = formula.compute_gradients(variable_assignments={'x': 2, 'y': 3, 'z': 4})
        >>> print(grads[x])  # Gradient with respect to x
        >>> print(grads[y])  # Gradient with respect to y
        >>> print(grads[z])  # Gradient with respect to z
        """
        self._apply_variable_assignments(variable_assignments)

        if backpropagation is None:
            backpropagation = np.ones_like(self.value_fn())

        return reduce(
            lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)},
            [var.compute_gradients(variable_assignments, backpropagation=val) for var, val in self.gradient_fn(backpropagation)],
            {self: backpropagation}
        )

    def _apply_variable_assignments(self, variable_assignments: Dict[Variable | str, float | np.ndarray]) -> None:
        if variable_assignments is not None:
            if all(isinstance(k, Variable) for k in variable_assignments.keys()):
                # The dictionary is of type Dict[Variable, float | np.ndarray]:
                assert all(v.name in variable_assignments for v in self.variables)
            else:
                # The dictionary is of type Dict[str, float | np.ndarray]:
                assert len(set(variable_assignments.keys()).difference(set(v.name for v in self.variables))) == 0
                variable_assignments = {v: variable_assignments[v.name] for v in self.variables}

            for k, v in variable_assignments.items():
                k.value = v
        if isinstance(self, Node):
            self._validate_operands()

    @staticmethod
    def _ensure_is_a_variable(other: Variable | np.ndarray | SupportsFloat):
        return other if isinstance(other, Variable) else Constant(other)


class Constant(Variable):
    """
    Represents a constant variable in a computational graph.

    Parameters
    ----------
    value : float | np.ndarray
        The constant value.
    name : str, optional
        The name of the constant, by default None.
    """

    def __init__(self, value: float | np.ndarray, name: str = None):
        if name is None:
            if isinstance(value, np.ndarray):
                name = value.__class__.__name__
                name += str(value.shape)
            else:
                name = str(value)
        super().__init__(name, value)
        self.variables = set()


class Node(Variable):
    """
    Represents a node in a computational graph. A node is formed by applying an operation on instances of Variable or Constant.

    Parameters
    ----------
    name : str
        The name of the node.
    operation : str
        The operation performed by the node.
    operands : Tuple[Variable, ...]
        The operands used in the operation.
    value_fn : Callable[[], float | np.ndarray], optional
        A function that computes the value of the node, by default None.
    gradient_fn : Callable[[float | np.ndarray], Tuple[Tuple[Variable, float | np.ndarray], ...]], optional
        A function that computes gradients, by default None.
    """
    operation: str
    operands: Tuple[Variable, ...]

    def __init__(
            self, name: str, operation: str, operands: Tuple[Variable, ...],
            value_fn: Callable[[], float | np.ndarray] = None,
            gradient_fn: Callable[[float | np.ndarray], Tuple[Tuple[Variable, float | np.ndarray], ...]] = lambda: []
    ):
        super().__init__(name=name, value_fn=value_fn, gradient_fn=gradient_fn)
        self.operation = operation
        self.operands = tuple(Variable._ensure_is_a_variable(operand) for operand in operands)
        self.variables = set.union(*[operand.variables for operand in self.operands])

        self._validate_operands()

    def _validate_operands(self) -> None:
        n = len(self.operands)

        if n == 1:
            if self.operands[0].shape:
                self._shape = self.operands[0].shape
        elif n == 2:
            left, right = self.operands
            if left.shape and right.shape:
                if self.operation == '@':
                    if left.shape[1] != right.shape[0]:
                        raise ValueError(
                            f"Matrix dimensions do not align for matrix multiplication: {left.shape} and {right.shape}.")
                    self._shape = left.shape[1], right.shape[0]
                else:
                    if left.shape != right.shape:
                        raise ValueError(
                            f"Matrix dimensions do not align for itemwise operation: {left.shape} and {right.shape}.")
                    self._shape = right.shape
        else:
            raise NotImplementedError

    @staticmethod
    def _apply_parenthesis_if_needed(item: Variable, op: str, right: bool = False) -> str:
        if isinstance(item, Node):
            if op == 'neg' and item.operation in ('+', '-'):
                # This prevents the conversion of -(x + y) to become -x + y:
                return f"({item.name})"
            if right and op == '/' and item.operation in ('+', '-', '*', '/'):
                # This prevents the conversion of 1 / (x * y) to become 1 / x * y:
                return f"({item.name})"
            if OPERATIONS['priority'][item.operation] > OPERATIONS['priority'][op]:
                # This covers the other cases where priority of operations requires parenthesis:
                return f"({item.name})"
        return item.name

    @staticmethod
    def unary_operation(
            item: Variable | np.ndarray | SupportsFloat,
            operation: str
    ) -> Node:
        """
        Perform a unary operation on a variable or constant.

        Parameters
        ----------
        item : Variable | np.ndarray | SupportsFloat
            The input variable, constant, or value.
        operation : str
            The unary operation to perform.

        Returns
        -------
        Node
            A new Node representing the result of the unary operation.

        Raises
        ------
        AssertionError
            If the specified unary operation is unsupported.

        See Also
        --------
        Node : Represents a node in the computation graph.
        """
        assert operation in OPERATIONS['unary'], f"Unsupported unary operation: {operation}"

        item = Variable._ensure_is_a_variable(item)
        operands = (item,)
        if operation == 'neg':
            item_name = Node._apply_parenthesis_if_needed(item, operation)
            name = f"-{item_name}"
        else:
            name = f"{operation}({item.name})"

        def value_fn() -> float | np.ndarray:
            return OPERATIONS['unary'][operation][0](item.value_fn())

        def gradient_fn(backpropagation: float | np.ndarray) -> Tuple[Tuple[Variable, float | np.ndarray], ...]:
            grad = OPERATIONS['unary'][operation][1](item.value_fn()) * backpropagation
            return (item, grad),

        return Node(name=name, operation=operation, operands=operands, value_fn=value_fn, gradient_fn=gradient_fn)

    @staticmethod
    def binary_operation(
            left: Variable | np.ndarray | SupportsFloat,
            right: Variable | SupportsFloat,
            operation: str
    ) -> Node:
        """
        Perform a binary operation on two variables, constants, or values.

        Parameters
        ----------
        left : Variable | np.ndarray | SupportsFloat
            The left operand.
        right : Variable | SupportsFloat
            The right operand.
        operation : str
            The binary operation to perform.

        Returns
        -------
        Node
            A new Node representing the result of the binary operation.

        Raises
        ------
        AssertionError
            If the specified binary operation is unsupported.

        See Also
        --------
        Node : Represents a node in the computation graph.
        """
        assert operation in OPERATIONS['binary'], f"Unsupported binary operation: {operation}"

        left = Variable._ensure_is_a_variable(left)
        right = Variable._ensure_is_a_variable(right)
        operands = (left, right)
        left_name = Node._apply_parenthesis_if_needed(left, operation)
        right_name = Node._apply_parenthesis_if_needed(right, operation, right=True)
        name = f"{left_name} {operation} {right_name}"

        def value_fn() -> float | np.ndarray:
            return OPERATIONS['binary'][operation][0](left.value_fn(), right.value_fn())

        def gradient_fn(backpropagation: float | np.ndarray) -> Tuple[Tuple[Variable, float | np.ndarray], ...]:

            grad_left, grad_right = OPERATIONS['binary'][operation][1](left.value_fn(), right.value_fn())

            if operation == '@':
                grad_left = np.matmul(backpropagation, grad_left)
                grad_right = np.matmul(grad_right, backpropagation)
            else:
                grad_left *= backpropagation
                grad_right *= backpropagation

            return (left, grad_left), (right, grad_right)

        return Node(name=name, operation=operation, operands=operands, value_fn=value_fn, gradient_fn=gradient_fn)


class Einsum(Node):
    """
    Represents an einsum operation in a computational graph.

    Parameters
    ----------
    subscripts : str
        The einsum subscripts string. (i.e. 'ij, jk -> ik')
    operands : Tuple[Variable]
        The operands used in the einsum operation.
    name : str, optional
        The name of the einsum operation, by default None.

    Raises
    ------
    ValueError
        If the number of operands doesn't match the einsum string or dimensions don't align.

    See Also
    --------
    Node : Represents a node in the computation graph.
    """
    subscripts: str
    subscripts_list: List[str]
    subscript_to_dim: Dict[str, int]

    def __init__(self, subscripts: str, *operands: Variable, name: str = None):
        self.operands = tuple(operands)
        self.subscripts = re.sub(r'\s+', '', subscripts)
        self.subscripts_list = re.split(r',|->', self.subscripts)
        self.subscript_to_dim = {}

        def value_fn() -> np.ndarray:
            operands_list = [operand.value_fn() if isinstance(operand, Variable) else operand for operand in self.operands]
            self.subscripts = "->".join(parser.parse_einsum_input((subscripts, *operands_list))[:2])
            self.subscripts_list = re.split(r',|->', self.subscripts)

            self._validate_operands()
            return contract(self.subscripts, *operands_list, optimize='optimal')

        def gradient_fn(backpropagation) -> Tuple[Tuple[Variable, np.ndarray], ...]:

            def partial_derivative(wrt: Variable, previous_grad: np.ndarray) -> np.ndarray:
                """ Compute the partial derivative of the einsum operation with respect to a specific variable. """
                if wrt not in self.operands:
                    return np.zeros_like(wrt.value)

                # Determine the location of the variable in the operands:
                location = self.operands.index(wrt)

                # Define the order of operands and subscripts to correctly perform the einsum:
                order = list(range(len(self.subscripts_list)))
                order[location], order[-1] = order[-1], order[location]

                # Reorder the operands and subscripts according to the computed order:
                operands_list = [operand.value for operand in self.operands] + [previous_grad]
                operands_list = [operands_list[i] for i in order]
                subscripts_list = [self.subscripts_list[i] for i in order]

                # Handle ellipsis (multiple dimensions) in the einsum subscripts:
                for i, letter in enumerate(re.findall(r'\.{3}|\S', self.subscripts_list[location])):
                    if letter not in re.findall(r'\.{3}|\S', "".join(subscripts_list[:-1])):
                        subscripts_list.insert(0, letter)
                        dim = wrt.shape[i]
                        operands_list.insert(0, np.ones(dim))

                # Construct the subscripts string for the new einsum operation:
                subscripts_ = ",".join(subscripts_list[:-1]) + "->" + subscripts_list[-1]
                return contract(subscripts_, *operands_list[:-1], optimize='optimal')

            return tuple((operand, partial_derivative(operand, backpropagation)) for operand in self.operands)

        operands_str = ", ".join(str(operand) for operand in self.operands)
        name = name if name is not None else f"einsum(subscripts='{self.subscripts}', {operands_str})"
        super().__init__(name=name, operation="einsum", operands=operands, value_fn=value_fn, gradient_fn=gradient_fn)

    def __repr__(self) -> str:
        return self.name

    def _validate_operands(self) -> None:
        if len(self.operands) + 1 != len(self.subscripts_list):
            raise ValueError("Number of operands doesn't match the einsum string!")

        for operand, operand_subscripts in zip(self.operands, self.subscripts_list[:-1]):
            if all((len(operand.shape), len(operand.shape) != len(operand_subscripts),
                    "..." not in operand_subscripts, operand_subscripts)):
                raise ValueError(f"Dimension of operand {operand} doesn't match the string! Shape: {operand.shape}, string: '{operand_subscripts}', subscripts: '{self.subscripts}'")

            operand_shape = operand.shape
            if operand_subscripts[:3] == "...":
                operand_subscripts, operand_shape = operand_subscripts[::-1], operand_shape[::-1]

            for i, letter in enumerate(re.findall(r'\.{3}|\S', operand_subscripts)):
                if i < len(operand_shape):
                    dim = operand_shape[i] if len(letter) == 1 else operand_shape[i:]
                    if self.subscript_to_dim.get(letter, dim) != dim:
                        raise ValueError("Inconsistent dimension names!")
                    self.subscript_to_dim[letter] = dim

        self._shape = tuple(self.subscript_to_dim.get(letter, 0) for letter in self.subscripts_list[-1])


def einsum(subscripts: str, *operands: Variable) -> Einsum:
    """
    Create an einsum operation.

    Parameters
    ----------
    subscripts : str
        The einsum subscripts string.
    operands : Variable
        The operands used in the einsum operation.

    Returns
    -------
    Einsum
        An Einsum object representing the einsum operation.

    Examples
    --------
    >>> x = Variable('x', np.array([[1, 2], [3, 4]]))
    >>> y = Variable('y', np.array([2, 3]))
    >>> result = einsum('ij, j -> i', x, y)
    >>> print(result)
    einsum(subscripts='ij,j->i', x, y)

    Notes
    -----
    This function creates an Einsum object that represents an einsum operation. The subscripts string specifies the
    contraction pattern, and operands are the variables or constants involved in the operation.

    See Also
    --------
    Einsum : Represents an einsum operation in a computational graph.
    """
    return Einsum(subscripts, *operands)


def set_variables(*names: str) -> Tuple[Variable, ...]:
    """
    Create and return a tuple of Variable objects with the specified names.

    Parameters
    ----------
    *names : str
        Variable names to create.

    Returns
    -------
    Tuple[Variable, ...]
        A tuple containing Variable objects initialized with the provided names.

    Examples
    --------
    >>> x, y, z = set_variables('x', 'y', 'z')
    >>> type(x)
    <class '__main__.Variable'>
    >>> type(y)
    <class '__main__.Variable'>
    >>> type(z)
    <class '__main__.Variable'>

    Notes
    -----
    This function simplifies the creation of multiple Variable objects with meaningful names in a single line.

    See Also
    --------
    Variable : Represents a variable in a computational graph.
    """
    return tuple(Variable(name) for name in names)


# Declaration of the unary mathematical functions at runtime:
for key in OPERATIONS['unary'].keys():
    if re.match(r'^[a-zA-Z_][\w_]*$', key) is not None:
        globals()[key] = partial(Node.unary_operation, operation=key)

# Declaration of the binary mathematical functions at runtime:
for key in OPERATIONS['binary'].keys():
    if re.match(r'^[a-zA-Z_][\w_]*$', key) is not None:
        globals()[key] = partial(Node.binary_operation, operation=key)

# A list of all the symbols in this module:
__all__ = ['erf', 'neg', 'erfc', 'sinh', 'asin', 'log10', 'log', 'atan', 'sin', 'asinh', 'acos', 'cos',
           'sqrt', 'acosh', 'abs', 'tan', 'cosh', 'tanh', 'exp', 'cbrt', 'atanh', 'einsum', 'Variable',
           'set_variables']


# Example usage:
if __name__ == "__main__":

    print("Example 1: Floating point variables.")
    x, y, z = set_variables('x', 'y', 'z')

    formula = exp((x + y) * (x - y) / (x ** z))
    print(f"f(x, y, z) = {formula}")                               # Displays the formula

    evaluation = formula.evaluate_at(x=2, y=3, z=4)
    print(f"f({x.value}, {y.value}, {z.value}) = {evaluation}")    # Evaluation of the expression

    grads = formula.grads
    print(f"∂f({x.value}, {y.value}, {z.value})/∂x = {grads[x]}")  # Gradient with respect to x
    print(f"∂f({x.value}, {y.value}, {z.value})/∂y = {grads[y]}")  # Gradient with respect to y
    print(f"∂f({x.value}, {y.value}, {z.value})/∂z = {grads[z]}")  # Gradient with respect to z

    print("\nExample 2: Matrix operations.")
    A, B, C = set_variables('A', 'B', 'C')
    formula = A @ B + C
    print(f"f(A, B, C) = {formula}")                               # Displays the formula

    A_val = np.diag(np.arange(10) + 1)
    B_val = np.ones((10, 5))
    C_val = np.random.randn(10, 5)
    evaluation = formula.evaluate_at(A=A_val, B=B_val, C=C_val)    # Evaluation of the expression
    print(f"f(A, B, C) = \n{evaluation}")

    grads = formula.grads
    print(f"df(A, B, C)/dA = \n{grads[A]}")                        # Gradient with respect to A
    print(f"df(A, B, C)/dB = \n{grads[B]}")                        # Gradient with respect to B
    print(f"df(A, B, C)/dB = \n{grads[C]}")                        # Gradient with respect to C
