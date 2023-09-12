# Automatic Differentiation with Variable Class

This Python project provides a simple implementation of automatic differentiation using the `Variable` class. Automatic differentiation (AD) is a technique widely used in machine learning and scientific computing to compute gradients of mathematical functions efficiently. This project aims to demonstrate AD principles in a clear and concise manner.

## Features

- Supports basic arithmetic operations (addition, negation, subtraction, multiplication, division and exponentiation).
- Computes gradients of composite functions.
- Implements reverse-mode automatic differentiation (backpropagation).

## Example

Here's a simple example of how you can use this project:

```python
from automatic_differentiation import Variable
    
x = Variable(2)
y = Variable(3)
z = Variable(4)

result = (x + y) * (x - y) / (x ** z)
print(result.value)  # Evaluation of the expression

gradients = result.compute_gradients(x, y, z)
print(gradients[x])  # Gradient with respect to x
print(gradients[y])  # Gradient with respect to y
print(gradients[z])  # Gradient with respect to z
```

## Contributing

While contributions are not the primary focus of this personal project, suggestions and feedback are always welcome. If you have ideas for improvements or spot any issues, feel free to create an issue or reach out.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to explore and modify the code as a learning exercise.

## Author

This notebook was composed by René Chenard, a computer scientist and mathematician with a degree from Université Laval.

You can contact the author at: [rene.chenard.1@ulaval.ca](mailto:rene.chenard.1@ulaval.ca)
