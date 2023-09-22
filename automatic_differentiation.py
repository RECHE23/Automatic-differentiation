from __future__ import annotations

import math
import re
from itertools import chain
from collections import OrderedDict
from functools import reduce, partial
from typing import Any, Optional, Set, SupportsFloat, Callable, Tuple, Dict

import numpy as np
from opt_einsum import contract, parser

# This allows the declaration of the functions in OPERATIONS at runtime:
global erf, neg, erfc, sinh, asin, log10, log, atan, sin, asinh, acos, cos, sqrt, acosh, abs, tan, cosh, tanh, exp, cbrt, atanh, transpose

OPERATIONS = {
    # Unary mathematical operations:
    'unary': {
        'neg': (np.negative, lambda item, grad: -np.ones_like(item.value) * grad),
        'abs': (np.abs, lambda item, grad: np.sign(item.value) * grad),
        'exp': (np.exp, lambda item, grad: np.exp(item.value) * grad),
        # TODO: Handle the case where item.value == 0:
        'log': (np.log, lambda item, grad: 1.0 / item.value * grad),
        # TODO: Handle the case where item.value == 0:
        'log10': (np.log10, lambda item, grad: 1.0 / (item.value * np.log(10.0)) * grad),
        'sin': (np.sin, lambda item, grad: np.cos(item.value) * grad),
        'asin': (np.arcsin, lambda item, grad: 1.0 / np.sqrt(1 - item.value ** 2) * grad),
        'cos': (np.cos, lambda item, grad: -np.sin(item.value) * grad),
        'acos': (np.arccos, lambda item, grad: -1.0 / np.sqrt(1.0 - item.value ** 2.0) * grad),
        'tan': (np.tan, lambda item, grad: 1.0 / np.cos(item.value) ** 2.0 * grad),
        'atan': (np.arctan, lambda item, grad: 1.0 / (1.0 + item.value ** 2.0) * grad),
        'sinh': (np.sinh, lambda item, grad: np.cosh(item.value) * grad),
        'asinh': (np.arcsinh, lambda item, grad: 1.0 / np.sqrt(1.0 + item.value ** 2.0) * grad),
        'cosh': (np.cosh, lambda item, grad: np.sinh(item.value) * grad),
        'acosh': (np.arccosh, lambda item, grad: 1.0 / np.sqrt(item.value ** 2.0 - 1.0) * grad),
        'tanh': (np.tanh, lambda item, grad: 1.0 / np.cosh(item.value) ** 2.0 * grad),
        'atanh': (np.arctanh, lambda item, grad: 1.0 / (1.0 - item.value ** 2.0) * grad),
        # TODO: Handle the case where item.value <= 0:
        'sqrt': (np.sqrt, lambda item, grad: 0.5 / np.sqrt(item.value) * grad),
        # TODO: Handle the case where item.value <= 0:
        'cbrt': (np.cbrt, lambda item, grad: 1.0 / (3.0 * item.value ** (2.0 / 3.0)) * grad),
        'erf': (np.vectorize(math.erf), lambda item, grad: 2.0 * np.exp(-item.value ** 2.0) / np.sqrt(np.pi) * grad),
        'erfc': (np.vectorize(math.erfc), lambda item, grad: -2.0 * np.exp(-item.value ** 2.0) / np.sqrt(np.pi) * grad),
        'transpose': (np.transpose, lambda item, grad, axes=None: np.transpose(grad, axes=np.argsort(axes) if axes is not None else None))
    },
    # Binary mathematical operations:
    'binary': {
        'add': (np.add, lambda left, right, grad: (np.ones_like(left.value) * grad,
                                                   np.ones_like(right.value) * grad)),
        'subtract': (np.subtract, lambda left, right, grad: (np.ones_like(left.value) * grad,
                                                             -np.ones_like(right.value) * grad)),
        'multiply': (np.multiply, lambda left, right, grad: (right.value * grad,
                                                             left.value * grad)),
        # TODO: Handle the case where right.value == 0:
        'divide': (np.divide, lambda left, right, grad: (1 / right.value * grad,
                                                         - left.value / right.value ** 2 * grad)),
        'matmul': (np.matmul, lambda left, right, grad: (np.matmul(grad, np.swapaxes(right.value.conj(), -1, -2)),
                                                         np.matmul(np.swapaxes(left.value.conj(), -1, -2), grad))),
        # TODO: Handle the case where left.value <= 0:
        'power': (np.power, lambda left, right, grad: (grad * (right.value * np.power(left.value, (right.value - 1))).conj(),
                                                       grad * (np.power(left.value, right.value) * np.log(left.value)).conj()))
    },
    # Priority of operations (0 being the highest priority):
    'priority': {
        'power': 0,
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
        'transpose': 1,
        'multiply': 2,
        'divide': 2,
        'matmul': 2,
        'add': 3,
        'subtract': 3},
    # Special representation of operations when not in the format operation(*operands):
    'repr': {
        'power': lambda left, right: f"{left} ** {right}",
        'neg': lambda item: f"-{item}",
        'transpose': lambda item: f"{item}.T",
        'multiply': lambda left, right: f"{left} * {right}",
        'divide': lambda left, right: f"{left} / {right}",
        'matmul': lambda left, right: f"{left} @ {right}",
        'add': lambda left, right: f"{left} + {right}",
        'subtract': lambda left, right: f"{left} - {right}"
    }
}


class Variable:
    """
    Represents a variable in a computational graph.

    Parameters
    ----------
    name : str
        The name of the variable.
    value : np.ndarray, optional
        The initial value of the variable, by default None.
    value_fn : Callable[[], np.ndarray], optional
        A function that computes the value of the variable, by default None.
    gradient_fn : Callable[[float | np.ndarray], Tuple[Tuple[Variable, np.ndarray], ...]], optional
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
    evaluate_gradients_at(**variable_assignments)
        Evaluate the variable's gradients with specific variable assignments.
    compute_gradients(backpropagation=None)
        Compute gradients for the variable.

    See Also
    --------
    Constant : Represents a constant variable.
    Node : Represents a node in the computation graph.
    """
    variables: Set[Variable]
    name: str
    value: np.ndarray
    value_fn: Callable[[], np.ndarray]
    gradient_fn: Callable[[float | np.ndarray], Tuple[Tuple[Variable, np.ndarray], ...]]
    id: str
    graph: str
    shape: Tuple[int, ...]
    ndim: int
    size: int
    dtype: np.dtype
    grads: Dict[Variable, np.ndarray]
    at: Dict[Variable, np.ndarray]
    T: Node

    def __init__(self, name: str, value: float | np.ndarray = None,
                 value_fn: Callable[[], np.ndarray] = None,
                 gradient_fn: Callable[[float | np.ndarray], Tuple[Tuple[Variable, np.ndarray], ...]] = None):
        self.name = name
        self.value = value
        self.value_fn = value_fn if value_fn is not None else lambda: self.value
        self.gradient_fn = gradient_fn if gradient_fn is not None else lambda backpropagation: []
        self.id = f"var_{id(self):x}"
        self._modified = False
        self.parents = dict()
        self.variables = {self}

    def __repr__(self) -> str:
        if all(np.any(v.value) is not None for v in self.variables):
            value_txt = f"ndarray⟨size={self.value.shape}, dtype={self.value.dtype}⟩" if self.value.ndim > 1 else self.value.squeeze()
            value_txt = "" if np.any(self.value) is None else f", value={value_txt}"
        else:
            value_txt = ""
        name_text = f"{self.__class__.__name__}(name='{self.name}'"
        if isinstance(self, Node):
            operands = [f"{n.name}" if isinstance(n, Constant) else f"'{n.name}'" for n in self.operands] or ", "
            operands_txt = ", ".join(operands)
            return f"{name_text}, operation='{self.operation}', operands=({operands_txt}){value_txt})"
        return f"{name_text}{value_txt})"

    def __str__(self) -> str:
        return self.name

    @property
    def _graph(self) -> str:
        if isinstance(self, Node):
            shape = "octagon" if self.operation == "einsum" else ("square" if len(self.name) <= 3 else "box")
            operation = self.params['subscripts'].replace('->', ' → ') if self.operation == "einsum" else self.operation
            graph_text = f'  {self.id} [style=filled, shape={shape}, fillcolor=lavenderblush3, label="{operation}", fontname=Courier];\n'
            graph_text += f"".join([f'  {self.id} -> {c.id};\n' for c in self.operands])
            graph_text += f"".join([c._graph for c in self.operands])
            return graph_text
        else:
            shape = ("circle" if len(self.name) <= 3 else "ellipse") if isinstance(self, Constant) else "doublecircle"
            fillcolor = "ivory3" if isinstance(self, Constant) else "lightsteelblue"
            return f'  {self.id} [style=filled, shape={shape}, fillcolor={fillcolor}, label="{self.name}", fontname=Courier];\n'

    @property
    def graph(self) -> str:
        graph = '\n'.join(OrderedDict.fromkeys(self._graph.split('\n')))
        return f"digraph {{\n" \
               f"  ordering=out;\n" \
               f"  fontsize=15;\n" \
               f"  labelloc=t;\n" \
               f"  label=\"Computational graph\";\n" \
               f"{graph}}}"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def size(self) -> int:
        return self.value.size

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    @property
    def value(self) -> Optional[np.ndarray]:
        if isinstance(self, Node) and (np.any(self._value) is None or any(v._modified for v in self.variables)):
            self._value = self.value_fn()
        return self._value

    @value.setter
    def value(self, value: SupportsFloat | np.ndarray) -> None:
        self._value = np.array(value)

    @property
    def grads(self) -> Dict[Variable, np.ndarray]:
        assert all(v.value is not None for v in self.variables), \
            "An evaluation of the formula must be done before trying to read the grads."
        return self.compute_gradients()

    @property
    def at(self) -> Dict[Variable, np.ndarray]:
        return {var: var.value for var in self.variables}

    @at.setter
    def at(self, variable_assignments: SupportsFloat | np.ndarray | Dict[Variable | str, float | np.ndarray]) -> None:
        if variable_assignments is not None:
            if isinstance(self, Constant):
                # Philosophical question: should we allow a Constant to change value? TBD...
                raise TypeError("You are trying to change the value of a constant!")
            if not isinstance(self, Node) and len(self.variables) == 1 and not isinstance(variable_assignments, dict):
                # It is a Variable, which is being assigned a numerical value:
                self.value = variable_assignments
            else:
                if all(isinstance(k, Variable) for k in variable_assignments.keys()):
                    # The dictionary is of type Dict[Variable, float | np.ndarray]:
                    assert all(v.name in variable_assignments for v in self.variables)
                else:
                    # The dictionary is of type Dict[str, float | np.ndarray]:
                    assert len(set(variable_assignments.keys()).difference(set(v.name for v in self.variables))) == 0
                    variable_assignments = {v: variable_assignments[v.name] for v in self.variables}

                for k, v in variable_assignments.items():
                    if (k.value != v).any():
                        if np.any(k.value) is not None:
                            k._modified = True
                        k.value = v
        if isinstance(self, Node):
            self._update_shape()
            self.value = None

    def __array__(self):
        return self.value

    @property
    def T(self) -> Node:
        return Node.unary_operation(self, "transpose")

    def __len__(self) -> int:
        return len(self.value)

    def __hash__(self):
        if isinstance(self, Constant):
            return hash((self.name, str(self.value.data)))
        elif isinstance(self, Node):
            return hash((self.name, self.operation, self.operands))
        else:
            return hash((self.name, id(self)))

    def __eq__(self, other):
        if isinstance(other, Variable):
            return hash(self) == hash(other)
        else:
            return self.value == other

    def __add__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, 'add')

    def __radd__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, 'add')

    def __sub__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, 'subtract')

    def __rsub__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, 'subtract')

    def __mul__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, 'multiply')

    def __rmul__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, 'multiply')

    def __truediv__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, 'divide')

    def __rtruediv__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, 'divide')

    def __matmul__(self, other: Variable | np.ndarray) -> Node:
        return Node.binary_operation(self, other, 'matmul')

    def __rmatmul__(self, other: Variable | np.ndarray | np.ndarray) -> Node:
        return Node.binary_operation(other, self, 'matmul')

    def __pow__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(self, other, 'power')

    def __rpow__(self, other: Variable | np.ndarray | SupportsFloat) -> Node:
        return Node.binary_operation(other, self, 'power')

    def __neg__(self) -> Node:
        return Node.unary_operation(self, "neg")

    def __abs__(self) -> Node:
        return Node.unary_operation(self, "abs")

    def evaluate_at(self, *args: float | np.ndarray, **variable_assignments: float | np.ndarray) -> np.ndarray:
        """
        Evaluate the value of the variable with specific variable assignments.

        Parameters
        ----------
        **variable_assignments : float | np.ndarray
            Keyword arguments where the keys are variable names and the values are the assigned values.

        Returns
        -------
        np.ndarray
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
        if len(self.variables) == 1 and args:
            self.at = {list(self.variables)[0].name: args}
        else:
            self.at = variable_assignments
        return self.value

    def evaluate_gradients_at(self, *args: float | np.ndarray, **variable_assignments: float | np.ndarray) -> Dict[Variable, np.ndarray]:
        """
        Evaluate the gradients of the variable with specific variable assignments.

        Parameters
        ----------
        **variable_assignments : float | np.ndarray
            Keyword arguments where the keys are variable names, and the values are the assigned values.

        Returns
        -------
        dict
            A dictionary where keys are Variable objects, and values are the computed gradients.

        Notes
        -----
        This method allows you to evaluate the gradients of the variable with respect to all other variables
        in the computation graph within a specific context by providing variable assignments as keyword arguments.
        It computes the gradients by substituting the assigned values for the corresponding variables in its expression.

        Examples
        --------
        >>> x, y, z = set_variables('x', 'y', 'z')
        >>> formula = exp((x + y) * (x - y) / (x ** z))
        >>> gradients = formula.evaluate_gradients_at(x=2, y=3, z=4)
        >>> print(gradients[x])
        """
        if len(self.variables) == 1 and args:
            self.at = {list(self.variables)[0].name: args}
        else:
            self.at = variable_assignments
        return self.compute_gradients()

    def compute_gradients(self, backpropagation: float | np.ndarray = None) -> Dict[Variable, np.ndarray]:
        """
        Compute gradients for the variable.

        Parameters
        ----------
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
        """
        if backpropagation is None:
            backpropagation = np.ones_like(self.value)

        return reduce(
            lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)},
            [var.compute_gradients(backpropagation=val) for var, val in self.gradient_fn(backpropagation)],
            {self: backpropagation}
        )

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
                name += str(value.shape).replace('(', '⟨').replace(')', '⟩').replace(', ', '×')
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
    value_fn : Callable[[], np.ndarray], optional
        A function that computes the value of the node, by default None.
    gradient_fn : Callable[[float | np.ndarray], Tuple[Tuple[Variable, np.ndarray], ...]], optional
        A function that computes gradients, by default None.
    """
    operation: str
    operands: Tuple[Variable, ...]

    def __init__(
            self, name: str, operation: str, operands: Tuple[Variable, ...],
            value_fn: Callable[[], np.ndarray] = None,
            gradient_fn: Callable[[float | np.ndarray], Tuple[Tuple[Variable, np.ndarray], ...]] = lambda: [],
            **params: Any
    ):
        self.operation = operation
        self.operands = tuple(Variable._ensure_is_a_variable(operand) for operand in operands)
        super().__init__(name=name, value_fn=value_fn, gradient_fn=gradient_fn)
        self.variables = set.union(*[operand.variables for operand in self.operands])
        self.params = params
        self._shape = None
        self._size = None

        self._update_shape()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def dtype(self) -> np.dtype:
        return np.result_type(*self.operands)

    def _update_shape(self) -> None:
        n = len(self.operands)

        if self.operation == "einsum":
            assert 'subscripts' in self.params
            in_subscripts = self.params['subscripts'].split('->')[0].split(',')

            subscript_to_dim = {}
            for operand, operand_subscripts in zip(self.operands, in_subscripts):

                operand_shape = operand.shape
                if operand_subscripts[:3] == "...":
                    operand_subscripts, operand_shape = operand_subscripts[::-1], operand_shape[::-1]

                for i, symbol in enumerate(re.findall(r'\.{3}|\S', operand_subscripts)):
                    if i < len(operand_shape):
                        dim = operand_shape[i] if len(symbol) == 1 else operand_shape[i:]
                        symbol_dim = subscript_to_dim.get(symbol, dim)
                        if symbol != '...' and symbol_dim != dim:
                            raise ValueError(f"Inconsistent dimension names! {symbol}->{symbol_dim}, expected {dim}.")
                        subscript_to_dim[symbol] = dim

            self._shape = tuple(subscript_to_dim.get(letter, 0) for letter in in_subscripts)
        elif n == 1:
            if self.operands[0].shape:
                self._shape = self.operands[0].shape
        elif n == 2:
            left, right = self.operands
            if left.shape and right.shape:
                if self.operation == 'matmul':
                    if left.shape[1] != right.shape[0]:
                        raise ValueError(
                            f"Matrix dimensions do not align for matrix multiplication: {left.shape} and {right.shape}.")
                    self._shape = left.shape[1], right.shape[0]
                else:
                    self._shape = np.broadcast_shapes(left.shape, right.shape)
        else:
            raise NotImplementedError

    @staticmethod
    def _apply_parenthesis_if_needed(item: Variable, op: str, right: bool = False) -> str:
        if isinstance(item, Node):
            if op == 'neg' and item.operation in ('add', 'subtract'):
                # This prevents the conversion of -(x + y) to become -x + y:
                return f"({item.name})"
            if right and op == 'divide' and item.operation in ('add', 'subtract', 'multiply', 'divide'):
                # This prevents the conversion of 1 / (x * y) to become 1 / x * y:
                return f"({item.name})"
            if OPERATIONS['priority'][item.operation] > OPERATIONS['priority'][op]:
                # This covers the other cases where priority of operations requires parenthesis:
                return f"({item.name})"
        return item.name

    @staticmethod
    def unary_operation(
            item: Variable | np.ndarray | SupportsFloat,
            operation: str,
            **params: Any
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

        key = (operation, operands)
        if key in item.parents:
            return item.parents[key]

        if operation in OPERATIONS['repr']:
            item_name = Node._apply_parenthesis_if_needed(item, operation)
            name = OPERATIONS['repr'][operation](item_name)
        else:
            name = f"{operation}({item.name})"

        def value_fn() -> np.ndarray:
            return OPERATIONS['unary'][operation][0](item.value, **params)

        def gradient_fn(backpropagation: float | np.ndarray) -> Tuple[Tuple[Variable, np.ndarray], ...]:
            grad = OPERATIONS['unary'][operation][1](item, backpropagation, **params)
            return (item, grad),

        node = Node(name=name, operation=operation, operands=operands, value_fn=value_fn, gradient_fn=gradient_fn, **params)
        item.parents[key] = node
        return node

    @staticmethod
    def binary_operation(
            left: Variable | np.ndarray | SupportsFloat,
            right: Variable | np.ndarray | SupportsFloat,
            operation: str,
            **params: Any
    ) -> Node:
        """
        Perform a binary operation on two variables, constants, or values.

        Parameters
        ----------
        left : Variable | np.ndarray | SupportsFloat
            The left operand.
        right : Variable | np.ndarray | SupportsFloat
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

        key = (operation, operands)
        if key in left.parents:
            return left.parents[key]
        if key in right.parents:
            return right.parents[key]

        if operation in OPERATIONS['repr']:
            left_name = Node._apply_parenthesis_if_needed(left, operation)
            right_name = Node._apply_parenthesis_if_needed(right, operation, right=True)
            name = OPERATIONS['repr'][operation](left_name, right_name)
        else:
            name = f"{operation}({left.name}, {right.name})"

        def value_fn() -> np.ndarray:
            return OPERATIONS['binary'][operation][0](left.value, right.value, **params)

        def gradient_fn(backpropagation: float | np.ndarray) -> Tuple[Tuple[Variable, np.ndarray], ...]:
            grad_left, grad_right = OPERATIONS['binary'][operation][1](left, right, backpropagation, **params)
            return (left, grad_left), (right, grad_right)

        node = Node(name=name, operation=operation, operands=operands, value_fn=value_fn, gradient_fn=gradient_fn, **params)
        left.parents[key] = node
        right.parents[key] = node
        return node

    @staticmethod
    def einsum(
            subscripts: str,
            *operands: Variable,
            optimize='optimal',
            **params: Any
    ) -> Node:
        # Ensure that all operands are instances of the Variable class:
        operands = tuple(Variable._ensure_is_a_variable(operand) for operand in operands)
        operation = "einsum"

        # Remove whitespace from subscripts:
        subscripts = re.sub(r'\s+', '', subscripts)

        key = (operation, subscripts, operands)
        for operand in operands:
            if key in operand.parents:
                return operand.parents[key]

        # Split subscripts into input and output parts:
        in_out_subscripts = subscripts.split('->')
        in_subscripts = in_out_subscripts[0].split(',')

        # Check the validity of the einsum subscripts:
        assert 1 <= len(in_out_subscripts) <= 2, f"Invalid einsum subscripts: {subscripts}"
        assert len(in_subscripts) == len(operands), \
            f"Number of input operands doesn't match input subscripts {in_subscripts} with {len(operands)} operands provided: {subscripts}."

        # Create a string representation of operands for naming purposes:
        operands_str = ", ".join(str(operand) for operand in operands)
        name = f"einsum(subscripts='{subscripts}', {operands_str})"

        # This allows the subscripts to be passed to the Node constructor:
        params['subscripts'] = subscripts

        def value_fn() -> np.ndarray:
            return contract(subscripts, *(operand.value for operand in operands), optimize=optimize)

        def gradient_fn(backpropagation):
            # This solution is based on the code from MyGrad:
            # source: https://github.com/rsokl/MyGrad/blob/133072b526966e235d70bbfcf9eb86d43d0fcfa1/src/mygrad/linalg/ops.py

            # Parse and uniformize the subscripts and their corresponding operands before evaluating the gradients:
            # (This is done using opt_einsum parser. This remove the need to deal with ellipsis (...) and exceptions.)
            operands_list = [operand.value for operand in operands]
            in_subscripts, out_subscripts, raw_operands = parser.parse_einsum_input((subscripts, *operands_list))
            raw_operands = tuple(raw_operands)
            in_subscripts = in_subscripts.split(',')

            # Get the dimension associated with a symbol:
            shapes = [op.shape for op in raw_operands]
            symbol_to_dim = {symbol: dim for symbols, dims in zip(in_subscripts, shapes) for symbol, dim in zip(symbols, dims)}

            # Iterate over the operands in order to evaluate the gradient with respect to them:
            gradients = ()
            for operand_index, operand in enumerate(operands):

                # Make a copy in order to keep the original unmodified:
                backpropagation_copy = backpropagation.copy()
                in_subscripts_copy = in_subscripts.copy()

                # Get the operand original subscripts and the subscripts without repeated symbols (i.e. 'iji' becomes 'ji'):
                operand_original_subscripts = in_subscripts_copy.pop(operand_index)
                gradient_subscripts = reduce(lambda acc, x: acc + x if x not in acc else acc, operand_original_subscripts[::-1], "")[::-1]

                # Check if a symbol is repeated in the operand original subscripts:
                a_symbol_is_repeated = len(gradient_subscripts) != len(operand_original_subscripts)

                # Evaluate the shape of the operand without any repeated symbol:
                operand_shape = tuple(symbol_to_dim[label] for label in gradient_subscripts) if a_symbol_is_repeated else operand.shape

                # Initialize the backpropagation subscripts used to evaluate the gradient w.r.t. the operand:
                backpropagation_subscripts = out_subscripts

                # Build a set composed of the symbols appearing on the left side of the arrow:
                unique_in_symbols = set(chain.from_iterable(in_subscripts_copy)) | set(backpropagation_subscripts)

                # If the right side of the subscripts contains a symbol that isn't on the left side,
                # expand the backpropagation and add the symbol to the left side at the proper position:
                if len(set(gradient_subscripts) - unique_in_symbols) > 0:
                    expanded_dims = [slice(None) for _ in range(backpropagation_copy.ndim)]
                    backpropagation_shape = list(backpropagation_copy.shape)
                    for n, symbol in enumerate(gradient_subscripts):
                        if symbol not in unique_in_symbols:
                            backpropagation_subscripts = backpropagation_subscripts[:n] + symbol + backpropagation_subscripts[n:]
                            expanded_dims.insert(n, np.newaxis)
                            backpropagation_shape.insert(n, operand_shape[n])
                    backpropagation_copy = np.broadcast_to(
                        backpropagation_copy if not backpropagation_copy.ndim else backpropagation_copy[tuple(expanded_dims)],
                        backpropagation_shape
                    )

                # Build the subscripts and sequence of operands used to evaluate the gradient w.r.t the operand:
                backpropagation_subscripts = ",".join([backpropagation_subscripts] + in_subscripts_copy) + "->" + gradient_subscripts
                out_operands = (backpropagation_copy,) + raw_operands[:operand_index] + raw_operands[operand_index + 1:]

                # If a symbol is repeated, create a view into the gradient array to map the evaluated values adequately:
                if a_symbol_is_repeated:
                    gradient = np.zeros(tuple(symbol_to_dim[i] for i in operand_original_subscripts))
                    out_view_shape = tuple(symbol_to_dim[i] for i in gradient_subscripts)

                    # Calculate the strides to map the data correctly in the view:
                    strides = tuple(
                        sum(gradient.strides[ind] for ind in (n for n, i in enumerate(operand_original_subscripts) if i == symbol))
                        for symbol in gradient_subscripts
                    )

                    # Create a view into the gradient array:
                    out_view = np.lib.stride_tricks.as_strided(gradient, shape=out_view_shape, strides=strides)

                    # Compute the gradient using the specified subscripts and operands, storing the result in the view:
                    contract(backpropagation_subscripts, *out_operands, out=out_view, optimize=optimize)
                else:
                    output_shape = operand.shape

                    # Compute the gradient using the specified subscripts and operands:
                    gradient = contract(backpropagation_subscripts, *out_operands, optimize=optimize)

                    # Check if the gradient shape matches the output shape:
                    if gradient.shape != output_shape:
                        if gradient.ndim != len(output_shape):
                            if gradient.ndim < len(output_shape):
                                raise ValueError(
                                    f"The dimensionality of the gradient of the broadcasted operation ({gradient.ndim})"
                                    f" is less than that of its associated variable ({len(output_shape)})"
                                )
                            gradient = gradient.sum(axis=tuple(range(gradient.ndim - len(output_shape))))

                        # Sum over dimensions to match the output shape if necessary:
                        keepdims = tuple(n for n, i in enumerate(gradient.shape) if i != output_shape[n])
                        if keepdims:
                            gradient = gradient.sum(axis=keepdims, keepdims=True)

                    # Broadcast the gradient to match the operand shape if necessary:
                    if operand_shape != gradient.shape:
                        gradient = np.broadcast_to(gradient, operand_shape)

                # Append the gradient and associated operand to the gradients tuple:
                gradients += ((operand, gradient),)

            return gradients

        node = Node(name=name, operation=operation, operands=operands, value_fn=value_fn, gradient_fn=gradient_fn, **params)

        for operand in operands:
            operand.parents[key] = node

        return node


def einsum(subscripts: str, *operands: Variable, optimize='optimal', **params) -> Node:
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
    Node
        An Node object representing the einsum operation.

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
    return Node.einsum(subscripts, *operands, optimize=optimize, **params)


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
    assert re.match(r'^[a-zA-Z_][\w_]*$', key) is not None, f"Invalid unary operation name: {key}"
    globals()[key] = partial(Node.unary_operation, operation=key)

# Declaration of the binary mathematical functions at runtime:
for key in OPERATIONS['binary'].keys():
    assert re.match(r'^[a-zA-Z_][\w_]*$', key) is not None, f"Invalid binary operation name: {key}"
    globals()[key] = partial(Node.binary_operation, operation=key)

# A list of all the symbols in this module:
__all__ = ['erf', 'neg', 'erfc', 'sinh', 'asin', 'log10', 'log', 'atan', 'sin', 'asinh', 'acos', 'cos',
           'sqrt', 'acosh', 'abs', 'tan', 'cosh', 'tanh', 'exp', 'cbrt', 'atanh', 'einsum', 'transpose',
           'Variable', 'Constant', 'set_variables']


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
    print(f"df(A, B, C)/dC = \n{grads[C]}")                        # Gradient with respect to C
