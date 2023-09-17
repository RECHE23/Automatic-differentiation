from __future__ import annotations
from functools import reduce, partial
from typing import Union, SupportsFloat, Callable, Tuple, Dict
import re
import math
import numpy as np
import random


OPERATIONS = {
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
    'binary': {
        '+': (np.add, lambda l_val, r_val: (np.ones_like(l_val), np.ones_like(r_val))),
        '-': (np.subtract, lambda l_val, r_val: (np.ones_like(l_val), -np.ones_like(r_val))),
        '*': (np.multiply, lambda l_val, r_val: (r_val, l_val)),
        '/': (np.divide, lambda l_val, r_val: (1 / r_val, - l_val / r_val ** 2)),
        '@': (np.matmul, lambda l_val, r_val: (r_val.T, l_val.T)),
        '**': (np.power, lambda l_val, r_val: (l_val ** (r_val - 1) * r_val, l_val ** r_val * np.log(l_val))),
    },
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

    def __init__(self, name: str, value: Union[float, np.ndarray] = None,
                 value_fn: Callable[[], Union[float, np.ndarray]] = None,
                 gradient_fn: Callable[[Union[float, np.ndarray]], Tuple[Tuple[Variable, Union[float, np.ndarray]], ...]] = None):
        self.variables = {self}
        self.name = name
        self._value = value
        self.value_fn = value_fn if value_fn is not None else lambda: self.value
        self.gradient_fn = gradient_fn if gradient_fn is not None else lambda backpropagation: []
        self.constant = not self.variables
        self.id = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(16))

    def __repr__(self):
        value_txt = f", value={self.value}" if self.value is not None else ""
        name_value_text = f"{self.__class__.__name__}(name='{self.name}'{value_txt}"
        if isinstance(self, Node):
            operands = [f"{n.name}" if n.constant else f"'{n.name}'" for n in self.operands] or ", "
            operands_txt = ", ".join(operands)
            return f"{name_value_text}, operation='{self.operation}', operands=({operands_txt}))"
        return f"{name_value_text})"

    def __str__(self):
        return self.name

    @property
    def _graph(self):
        if isinstance(self, Node):
            graph_text = f'  {self.id} [shape=box, label="{self.operation}"];\n'
            graph_text += f"".join([f'  {self.id} -> {c.id};\n' for c in self.operands])
            graph_text += f"".join([c._graph for c in self.operands])
            return graph_text
        else:
            return f'  {self.id} [label="{self.name}"];\n'

    @property
    def graph(self):
        return f"digraph {{\n" \
               f"labelloc=\"t\"" \
               f"label=\"Evaluation graph\"" \
               f"{self._graph}}}"

    @property
    def shape(self):
        if not isinstance(self, Node) and self.value is not None and hasattr(self.value, 'shape'):
            return self.value.shape
        return self._shape if hasattr(self, '_shape') else ()

    @property
    def value(self):
        if self._value is None:
            return None
        if isinstance(self._value, np.ndarray):
            return self._value
        return float(self._value)

    @value.setter
    def value(self, value: Union[float, np.ndarray]):
        self._value = value
        if isinstance(value, np.ndarray):
            self._shape = value.shape

    @property
    def grads(self) -> Dict[Variable, float]:
        assert all(v.value is not None for v in self.variables), "An evaluation of the formula must be done before trying to read the grads."
        return self.compute_gradients()

    def __add__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(self, other, "+")

    def __radd__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(other, self, "+")

    def __sub__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(self, other, "-")

    def __rsub__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(other, self, "-")

    def __mul__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(self, other, "*")

    def __rmul__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(other, self, "*")

    def __truediv__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(self, other, "/")

    def __rtruediv__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(other, self, "/")

    def __matmul__(self, other: Union[Variable, np.ndarray]) -> Node:
        return Node.binary_operation(self, other, "@")

    def __rmatmul__(self, other: Union[Variable, np.ndarray]) -> Node:
        return Node.binary_operation(other, self, "@")

    def __pow__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(self, other, "**")

    def __rpow__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(other, self, "**")

    def __neg__(self) -> Node:
        return Node.unary_operation(self, "neg")

    def __abs__(self) -> Node:
        return Node.unary_operation(self, "abs")

    def evaluate_at(self, **variable_assignments) -> Union[float, np.ndarray]:
        self._apply_variable_assignments(variable_assignments)
        self.value = self.value_fn()
        return self.value

    def compute_gradients(self, variable_assignments: Dict[Variable, Union[float, np.ndarray]] = None,
                          backpropagation: Union[float, np.ndarray] = None) -> Dict[Variable, Union[float, np.ndarray]]:
        self._apply_variable_assignments(variable_assignments)

        if backpropagation is None:
            backpropagation = np.ones_like(self.value_fn())

        return reduce(
            lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)},
            [var.compute_gradients(variable_assignments, backpropagation=val) for var, val in self.gradient_fn(backpropagation)],
            {self: backpropagation}
        )

    def _apply_variable_assignments(self, variable_assignments):
        if variable_assignments is not None:
            if all(isinstance(k, Variable) for k in variable_assignments.keys()):
                # The dictionary is of type Dict[Variable, Union[float, np.ndarray]]:
                assert all(v.name in variable_assignments for v in self.variables)
            else:
                # The dictionary is of type Dict[str, Union[float, np.ndarray]]:
                assert len(set(variable_assignments.keys()).difference(set(v.name for v in self.variables))) == 0
                variable_assignments = {v: variable_assignments[v.name] for v in self.variables}

            for k, v in variable_assignments.items():
                k.value = v

    @staticmethod
    def _ensure_is_a_variable(other: Union[Variable, np.ndarray, SupportsFloat]):
        return other if isinstance(other, Variable) else Constant(other)


class Constant(Variable):

    def __init__(self, value: Union[float, np.ndarray], name: str = None):
        if name is None:
            if isinstance(value, np.ndarray):
                name = value.__class__.__name__
                name += str(value.shape)
            else:
                name = str(value)
        super().__init__(name, value)
        self.variables = set()
        self.constant = True


class Node(Variable):

    def __init__(self, name: str, operation: str, operands: Tuple[Variable, ...],
                 value_fn: Callable[[], Union[float, np.ndarray]] = None,
                 gradient_fn: Callable[[Union[float, np.ndarray]], Tuple[Tuple[Variable, Union[float, np.ndarray]], ...]] = lambda: []):
        super().__init__(name=name, value_fn=value_fn, gradient_fn=gradient_fn)
        self.operation = operation
        self.operands = tuple(Variable._ensure_is_a_variable(operand) for operand in operands)
        self.variables = set.union(*[operand.variables for operand in self.operands])

        self._validate_operands()

    def _apply_variable_assignments(self, variable_assignments):
        super()._apply_variable_assignments(variable_assignments)
        self._validate_operands()

    def _validate_operands(self):
        n = len(self.operands)

        if n == 1:
            if self.operands[0].shape:
                self._shape = self.operands[0].shape
        elif n == 2:
            left, right = self.operands
            if left.shape and right.shape:
                if self.operation == '@':
                    if left.shape[1] != right.shape[0]:
                        raise ValueError(f"Matrix dimensions do not align for matrix multiplication: {left.shape} and {right.shape}.")
                    self._shape = left.shape[1], right.shape[0]
                else:
                    if left.shape != right.shape:
                        raise ValueError(f"Matrix dimensions do not align for itemwise operation: {left.shape} and {right.shape}.")
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
    def unary_operation(item: Union[Variable, np.ndarray, SupportsFloat], op: str) -> Node:
        assert op in OPERATIONS['unary']

        item = Variable._ensure_is_a_variable(item)
        operands = (item,)
        if op == 'neg':
            item_name = Node._apply_parenthesis_if_needed(item, op)
            name = f"-{item_name}"
        else:
            name = f"{op}({item.name})"

        def value_fn():
            return OPERATIONS['unary'][op][0](item.value_fn())

        def gradient_fn(backpropagation):
            grad = OPERATIONS['unary'][op][1](item.value_fn()) * backpropagation
            return (item, grad),

        return Node(name=name, operation=op, operands=operands, value_fn=value_fn, gradient_fn=gradient_fn)

    @staticmethod
    def binary_operation(left: Union[Variable, np.ndarray, SupportsFloat], right: Union[Variable, SupportsFloat], op: str) -> Node:
        assert op in OPERATIONS['binary']

        left = Variable._ensure_is_a_variable(left)
        right = Variable._ensure_is_a_variable(right)
        operands = (left, right)
        left_name = Node._apply_parenthesis_if_needed(left, op)
        right_name = Node._apply_parenthesis_if_needed(right, op, right=True)
        name = f"{left_name} {op} {right_name}"

        def value_fn():
            return OPERATIONS['binary'][op][0](left.value_fn(), right.value_fn())

        def gradient_fn(backpropagation):

            grad_left, grad_right = OPERATIONS['binary'][op][1](left.value_fn(), right.value_fn())

            if op == '@':
                grad_left = np.matmul(backpropagation, grad_left)
                grad_right = np.matmul(grad_right, backpropagation)
            else:
                grad_left *= backpropagation
                grad_right *= backpropagation

            return (left, grad_left), (right, grad_right)

        return Node(name=name, operation=op, operands=operands, value_fn=value_fn, gradient_fn=gradient_fn)


class Einsum(Node):
    def __init__(self, subscripts: str, *operands: Variable, name: str = None):
        self.operands = list(operands)
        self.subscripts = re.sub(r'\s+', '', subscripts)
        self.subscripts_list = re.split(r',|->', self.subscripts)
        self.subscript_to_dim = {}

        self._validate_operands()

        def value_fn():
            return np.einsum(self.subscripts, *[operand.value_fn() if isinstance(operand, Variable) else operand for operand in self.operands])

        def gradient_fn(backpropagation):

            def partial_derivative(wrt, previous_grad):
                if wrt not in self.operands:
                    return 0

                location = self.operands.index(wrt)
                order = list(range(len(self.subscripts_list)))
                order[location], order[-1] = order[-1], order[location]

                operands_with_grad = list(np.array(list(self.operands) + [previous_grad], dtype=object)[order])
                opnames = list(np.array(self.subscripts_list)[order])

                for i, letter in enumerate(re.findall(r'\.{3}|\S', self.subscripts_list[location])):
                    if letter not in re.findall(r'\.{3}|\S', "".join(opnames[:-1])):
                        opnames.insert(0, letter)
                        dim = wrt.shape[i]
                        var_to_insert = self._ensure_is_a_variable(np.ones(dim))
                        operands_with_grad.insert(0, var_to_insert)

                subscripts = ",".join(opnames[:-1]) + "->" + opnames[-1]
                return Einsum(subscripts, *operands_with_grad[:-1]).value_fn()

            return tuple((operand, partial_derivative(operand, backpropagation)) for operand in self.operands)

        operands_str = ", ".join(str(operand) for operand in self.operands)
        name = name if name is not None else f"einsum(subscripts='{self.subscripts}', {operands_str})"
        super().__init__(name=name, operation="einsum", operands=operands, value_fn=value_fn, gradient_fn=gradient_fn)

    def __repr__(self):
        return self.name

    def _validate_operands(self):
        if len(self.operands) + 1 != len(self.subscripts_list):
            raise ValueError("Number of operands doesn't match the einsum string!")

        for operand, operand_subscripts in zip(self.operands, self.subscripts_list[:-1]):
            if len(operand.shape) != 0 and len(operand.shape) != len(operand_subscripts) and "..." not in operand_subscripts and operand_subscripts != "":
                raise ValueError(f"Dimension of operand {operand} doesn't match the string! Shape: {operand.shape}, string: '{operand_subscripts}'")

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


def einsum(subscripts, *operands):
    return Einsum(subscripts, *operands)


for key in OPERATIONS['unary'].keys():
    if re.match(r'^[a-zA-Z_][\w_]*$', key) is not None:
        func = partial(Node.unary_operation, op=key)
        setattr(Variable, key, func)
        globals()[key] = func

for key in OPERATIONS['binary'].keys():
    if re.match(r'^[a-zA-Z_][\w_]*$', key) is not None:
        func = partial(Node.binary_operation, op=key)
        setattr(Variable, key, func)
        globals()[key] = func

__all__ = ['erf', 'neg', 'erfc', 'sinh', 'asin', 'log10', 'log', 'atan', 'sin', 'asinh', 'acos',
           'cos', 'sqrt', 'acosh', 'abs', 'tan', 'cosh', 'tanh', 'exp', 'cbrt', 'atanh', 'einsum']


# Example usage
if __name__ == "__main__":
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')

    formula = exp((x + y) * (x - y) / (x ** z))
    print(f"f(x, y, z) = {formula}")                               # Displays the formula

    evaluation = formula.evaluate_at(x=2, y=3, z=4)
    print(f"f({x.value}, {y.value}, {z.value}) = {evaluation}")    # Evaluation of the expression

    grads = formula.grads
    print(f"∂f({x.value}, {y.value}, {z.value})/∂x = {grads[x]}")  # Gradient with respect to x
    print(f"∂f({x.value}, {y.value}, {z.value})/∂y = {grads[y]}")  # Gradient with respect to y
    print(f"∂f({x.value}, {y.value}, {z.value})/∂z = {grads[z]}")  # Gradient with respect to z

    A = Variable('A')
    B = Variable('B')
    formula = A @ B
    print(f"f(A, B) = {formula}")

    evaluation = formula.evaluate_at(A=np.diag([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), B=np.ones((10, 5)))
    print(f"f(A, B) = \n{evaluation}")

    grads = formula.grads
    print(f"df(A, B)/dA = \n{grads[A]}")
    print(f"df(A, B)/dB = \n{grads[B]}")
