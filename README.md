# Automatic Differentiation

This Python project provides a simple implementation of automatic differentiation using the `Variable` class. Automatic differentiation (AD) is a technique widely used in machine learning and scientific computing to compute gradients of mathematical functions efficiently. This project aims to demonstrate AD principles in a clear and concise manner.

## Features

- Supports basic arithmetic operations (addition, negation, subtraction, multiplication, division and exponentiation).
- Supports mathematical functions like `sin`, `cos`, `tan`, `exp` and `sqrt`.
- Builds a computational graphs of calculations and computes evaluation and gradients of composite functions on demand.

## Example

Here's a simple example of how you can use this project:

```python
from automatic_differentiation import Variable, exp

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
```

## Contributing

While contributions are not the primary focus of this personal project, suggestions and feedback are always welcome. If you have ideas for improvements or spot any issues, feel free to create an issue or reach out.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to explore and modify the code as a learning exercise.

## Author

This notebook was composed by René Chenard, a computer scientist and mathematician with a degree from Université Laval.

You can contact the author at: [rene.chenard.1@ulaval.ca](mailto:rene.chenard.1@ulaval.ca)
