from __future__ import annotations
from functools import reduce
from typing import Set, Union, SupportsFloat, Callable, Tuple, Dict
import math

unary_operations = {
    'neg': float.__neg__,
    'abs': float.__abs__,
    'exp': math.exp,
    'log': math.log,
    'log10': math.log10,
    'sin': math.sin,
    'asin': math.asin,
    'cos': math.cos,
    'acos': math.acos,
    'tan': math.tan,
    'atan': math.atan,
    'sinh': math.sinh,
    'asinh': math.asinh,
    'cosh': math.cosh,
    'acosh': math.acosh,
    'tanh': math.tanh,
    'atanh': math.atanh,
    'sqrt': math.sqrt,
    'cbrt': lambda v: v ** (1 / 3),
    'erf': math.erf,
    'erfc': math.erfc}

binary_operations = {
    '+': float.__add__,
    '-': float.__sub__,
    '*': float.__mul__,
    '/': float.__truediv__,
    '**': float.__pow__}

operation_priority = {
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
    '*': 2,
    '/': 2,
    '+': 3,
    '-': 3}


class Variable:
    def __init__(self, name: str, value: SupportsFloat = None,
                 value_fn: Callable[[], float] = None, gradient_fn: Callable[[], Tuple[Tuple[Variable, float], ...]] = lambda: []):
        self.name = name
        self.variables = {self}
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
        self._value = float(value)

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

    def __pow__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(self, other, "**")

    def __rpow__(self, other: Union[Variable, SupportsFloat]) -> Node:
        return Node.binary_operation(other, self, "**")

    def __neg__(self) -> Node:
        return Node.unary_operation(self, "neg")

    def __abs__(self) -> Node:
        return Node.unary_operation(self, "abs")

    def exp(self):
        return Node.unary_operation(self, "exp")

    def log(self):
        return Node.unary_operation(self, "log")

    def log10(self):
        return Node.unary_operation(self, "log10")

    def sin(self):
        return Node.unary_operation(self, "sin")

    def cos(self):
        return Node.unary_operation(self, "cos")

    def tan(self):
        return Node.unary_operation(self, "tan")

    def sinh(self):
        return Node.unary_operation(self, "sinh")

    def cosh(self):
        return Node.unary_operation(self, "cosh")

    def tanh(self):
        return Node.unary_operation(self, "tanh")

    def acos(self):
        return Node.unary_operation(self, "acos")

    def acosh(self):
        return Node.unary_operation(self, "acosh")

    def asin(self):
        return Node.unary_operation(self, "asin")

    def asinh(self):
        return Node.unary_operation(self, "asinh")

    def atan(self):
        return Node.unary_operation(self, "atan")

    def atanh(self):
        return Node.unary_operation(self, "atanh")

    def sqrt(self):
        return Node.unary_operation(self, "sqrt")

    def cbrt(self):
        return Node.unary_operation(self, "cbrt")

    def erf(self):
        return Node.unary_operation(self, "erf")

    def erfc(self):
        return Node.unary_operation(self, "erfc")

    def evaluate(self, variable_assignments: Dict[Variable, SupportsFloat]) -> float:
        self._apply_variable_assignments(variable_assignments)
        return self.value_fn()

    def compute_gradients(self, variable_assignments: Dict[Variable, SupportsFloat] = None, backpropagation: float = 1.0) -> Dict[Variable, float]:
        self._apply_variable_assignments(variable_assignments)

        return reduce(
            lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)},
            [var.compute_gradients(variable_assignments, backpropagation=val * backpropagation) for var, val in self.gradient_fn()],
            {self: backpropagation}
        )

    @staticmethod
    def _apply_variable_assignments(variable_assignments):
        if variable_assignments is not None:
            for k, v in variable_assignments.items():
                k.value = v

    @staticmethod
    def _ensure_is_variable(other: Union[Variable, SupportsFloat]):
        return Variable(name=str(other), value=float(other)) if not isinstance(other, Variable) else other


class Node(Variable):
    def __init__(self, name: str, variables: Set[Variable] = None, operation: str = None,
                 value_fn: Callable[[], float] = None, gradient_fn: Callable[[], Tuple[Tuple[Variable, float], ...]] = lambda: []):
        super().__init__(name=name, value_fn=value_fn, gradient_fn=gradient_fn)
        self.variables = {self} if variables is None else variables
        self.operation = operation

    @staticmethod
    def _apply_parenthesis_if_needed(item: Variable, op: str, right: bool = False) -> str:
        if isinstance(item, Node):
            if op == 'neg' and item.operation in ('+', '-'):
                # This prevents the conversion of -(x + y) to become -x + y:
                return f"({item.name})"
            if right and op == '/' and item.operation in ('+', '-', '*', '/'):
                # This prevents the conversion of 1 / (x * y) to become 1 / x * y:
                return f"({item.name})"
            if operation_priority[item.operation] > operation_priority[op]:
                # This covers the other cases where priority of operations requires parenthesis:
                return f"({item.name})"
        return item.name

    @staticmethod
    def unary_operation(item: Union[Variable, SupportsFloat], op: str) -> Node:
        item = Variable._ensure_is_variable(item)
        variables = item.variables
        if op == 'neg':
            item_name = Node._apply_parenthesis_if_needed(item, op)
            name = f"-{item_name}"
        else:
            name = f"{op}({item.name})"

        def value_fn():
            return unary_operations[op](item.value_fn())

        def gradient_fn():
            if op == 'neg':
                grad = -1.0
            elif op == 'abs':
                grad = math.copysign(1.0, item.value_fn())
            elif op == 'exp':
                grad = math.exp(item.value_fn())
            elif op == 'log':
                grad = 1.0 / item.value_fn()
            elif op == 'log10':
                grad = 1.0 / (item.value_fn() * math.log(10.0))
            elif op == 'sin':
                grad = math.cos(item.value_fn())
            elif op == 'asin':
                grad = 1.0 / math.sqrt(1 - item.value_fn() ** 2)
            elif op == 'cos':
                grad = -math.sin(item.value_fn())
            elif op == 'acos':
                grad = -1.0 / math.sqrt(1.0 - item.value_fn() ** 2.0)
            elif op == 'tan':
                grad = 1.0 / math.cos(item.value_fn()) ** 2.0
            elif op == 'atan':
                grad = 1.0 / (1.0 + item.value_fn() ** 2.0)
            elif op == 'sinh':
                grad = math.cosh(item.value_fn())
            elif op == 'asinh':
                grad = 1.0 / math.sqrt(1.0 + item.value_fn() ** 2.0)
            elif op == 'cosh':
                grad = math.sinh(item.value_fn())
            elif op == 'acosh':
                grad = 1.0 / math.sqrt(item.value_fn() ** 2.0 - 1.0)
            elif op == 'tanh':
                grad = 1.0 / math.cosh(item.value_fn()) ** 2.0
            elif op == 'atanh':
                grad = 1.0 / (1.0 - item.value_fn() ** 2.0)
            elif op == 'sqrt':
                grad = 0.5 / math.sqrt(item.value_fn())
            elif op == 'cbrt':
                grad = 1.0 / (3.0 * item.value_fn() ** (2.0 / 3.0))
            elif op == 'erf':
                grad = 2.0 * math.exp(-item.value_fn() ** 2.0) / math.sqrt(math.pi)
            elif op == 'erfc':
                grad = -2.0 * math.exp(-item.value_fn() ** 2.0) / math.sqrt(math.pi)
            else:
                raise NotImplementedError

            return (item, grad),

        return Node(name=name, variables=variables, operation=op, value_fn=value_fn, gradient_fn=gradient_fn)

    @staticmethod
    def binary_operation(left: Union[Variable, SupportsFloat], right: Union[Variable, SupportsFloat], op: str) -> Node:
        left = Variable._ensure_is_variable(left)
        right = Variable._ensure_is_variable(right)
        variables = left.variables.union(right.variables)
        left_name = Node._apply_parenthesis_if_needed(left, op)
        right_name = Node._apply_parenthesis_if_needed(right, op, right=True)
        name = f"{left_name} {op} {right_name}"

        def value_fn():
            return binary_operations[op](left.value_fn(), right.value_fn())

        def gradient_fn():
            if op == '+':
                grad_left, grad_right = 1.0, 1.0
            elif op == '-':
                grad_left, grad_right = 1.0, -1.0
            elif op == '*':
                grad_left, grad_right = right.value_fn(), left.value_fn()
            elif op == '/':
                grad_left, grad_right = 1 / right.value_fn(), - left.value_fn() / right.value_fn() ** 2
            elif op == '**':
                grad_left = left.value_fn() ** (right.value_fn() - 1) * right.value_fn()
                grad_right = left.value_fn() ** right.value_fn() * math.log(left.value_fn())
            else:
                raise NotImplementedError

            return (left, grad_left), (right, grad_right)

        return Node(name=name, variables=variables, operation=op, value_fn=value_fn, gradient_fn=gradient_fn)


def exp(variable):
    return variable.exp() if isinstance(variable, Variable) else math.exp(variable)


def log(variable):
    return variable.log() if isinstance(variable, Variable) else math.log(variable)


def log10(variable):
    return variable.log10() if isinstance(variable, Variable) else math.log10(variable)


def sin(variable):
    return variable.sin() if isinstance(variable, Variable) else math.sin(variable)


def cos(variable):
    return variable.cos() if isinstance(variable, Variable) else math.cos(variable)


def tan(variable):
    return variable.tan() if isinstance(variable, Variable) else math.tan(variable)


def sinh(variable):
    return variable.sinh() if isinstance(variable, Variable) else math.sinh(variable)


def cosh(variable):
    return variable.cosh() if isinstance(variable, Variable) else math.cosh(variable)


def tanh(variable):
    return variable.tanh() if isinstance(variable, Variable) else math.tanh(variable)


def acos(variable):
    return variable.acos() if isinstance(variable, Variable) else math.acos(variable)


def acosh(variable):
    return variable.acosh() if isinstance(variable, Variable) else math.acosh(variable)


def asin(variable):
    return variable.asin() if isinstance(variable, Variable) else math.asin(variable)


def asinh(variable):
    return variable.asinh() if isinstance(variable, Variable) else math.asinh(variable)


def atan(variable):
    return variable.atan() if isinstance(variable, Variable) else math.atan(variable)


def atanh(variable):
    return variable.atanh() if isinstance(variable, Variable) else math.atanh(variable)


def sqrt(variable):
    return variable.sqrt() if isinstance(variable, Variable) else math.sqrt(variable)


def cbrt(variable):
    return variable.cbrt() if isinstance(variable, Variable) else variable ** (1 / 3)


def erf(variable):
    return variable.erf() if isinstance(variable, Variable) else math.erf(variable)


def erfc(variable):
    return variable.erfc() if isinstance(variable, Variable) else math.erfc(variable)


# Example usage
if __name__ == "__main__":
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')

    formula = exp((x + y) * (x - y) / (x ** z))
    print(f"f(x, y, z) = {formula}")  # Displays the formula

    evaluation = formula.evaluate({x: 2, y: 3, z: 4})
    print(f"f({x.value}, {y.value}, {z.value}) = {evaluation}")  # Evaluation of the expression

    grads = formula.grads
    print(f"∂f({x.value}, {y.value}, {z.value})/∂x = {grads[x]}")  # Gradient with respect to x
    print(f"∂f({x.value}, {y.value}, {z.value})/∂y = {grads[y]}")  # Gradient with respect to y
    print(f"∂f({x.value}, {y.value}, {z.value})/∂z = {grads[z]}")  # Gradient with respect to z
