# Automatic Differentiation

This Python project provides a simple implementation of automatic differentiation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Introduction

Automatic differentiation (AD) is a technique widely used in machine learning and scientific computing to compute gradients of mathematical functions efficiently. This project aims to demonstrate AD principles in a clear and concise manner.

## Installation

To explore and experiment with this framework, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/RECHE23/Automatic-differentiation.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Automatic-differentiation
    ```

3. Set up a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use "venv\Scripts\activate".
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Features

- Supports basic arithmetic operations (addition, subtraction, multiplication, division, and exponentiation).
- Supports trigonometric functions like `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, and more.
- Provides differentiable versions of functions like `sqrt`, `cbrt`, `exp`, `log`, `log10`, `abs`, `erf`, and more.
- Supports matrix operations and provides differentiable versions of matrix operations like `@` and `einsum`.
- Builds a computational graph of operations and computes value and gradients of functions on demand.
- Allows easy management of variables and their values using the `Variable` class and `set_variables`.

## Usage

Here's a simple example of how you can use this project:

```python
from automatic_differentiation import set_variables, exp

x, y, z = set_variables('x', 'y', 'z')

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
