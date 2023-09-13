import unittest
import math
from automatic_differentiation import Variable


class TestVariable(unittest.TestCase):
    def test_addition(self):
        x = Variable('x')
        y = Variable('y')

        formula = x + y
        self.assertEqual(str(formula), "(x + y)")

        evaluation_result = formula.evaluate({x: 5, y: 3})
        self.assertEqual(evaluation_result, 8)

        self.assertEqual(formula.grads[x], 1.0)
        self.assertEqual(formula.grads[y], 1.0)

    def test_subtraction(self):
        x = Variable('x')
        y = Variable('y')

        formula = x - y
        self.assertEqual(str(formula), "(x - y)")

        evaluation_result = formula.evaluate({x: 5, y: 3})
        self.assertEqual(evaluation_result, 2)

        self.assertEqual(formula.grads[x], 1.0)
        self.assertEqual(formula.grads[y], -1.0)

    def test_multiplication(self):
        x = Variable('x')
        y = Variable('y')

        formula = x * y
        self.assertEqual(str(formula), "x * y")

        evaluation_result = formula.evaluate({x: 5, y: 3})
        self.assertEqual(evaluation_result, 15)

        self.assertEqual(formula.grads[x], 3.0)
        self.assertEqual(formula.grads[y], 5.0)

    def test_division(self):
        x = Variable('x')
        y = Variable('y')

        formula = x / y
        self.assertEqual(str(formula), "x / y")

        evaluation_result = formula.evaluate({x: 10, y: 2})
        self.assertEqual(evaluation_result, 5)

        self.assertEqual(formula.grads[x], 0.5)
        self.assertEqual(formula.grads[y], -2.5)

    def test_power(self):
        x = Variable('x')
        y = Variable('y')

        formula = x ** y
        self.assertEqual(str(formula), "x ** y")

        evaluation_result = formula.evaluate({x: 2, y: 3})
        self.assertEqual(evaluation_result, 8)

        self.assertEqual(formula.grads[x], 12.0)
        self.assertEqual(formula.grads[y], 8.0 * math.log(2))

    def test_negative(self):
        x = Variable('x')

        formula = -x
        self.assertEqual(str(formula), "(-x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, -5)

        self.assertEqual(formula.grads[x], -1.0)

    def test_composition(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = (x + y) * (x - y) / x ** z
        self.assertEqual(str(formula), "(x + y) * (x - y) / x ** z")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertEqual(evaluation_result, -5 / 16)

        self.assertEqual(formula.grads[x], 0.875)
        self.assertEqual(formula.grads[y], -0.375)
        self.assertEqual(formula.grads[z], 5 / 16 * math.log(2))

    def test_advanced_composition(self):
        x = Variable('x')
        y = Variable('y')

        formula = ((x ** 2 + 1) - 1 / y) ** 3
        self.assertEqual(str(formula), "((x ** 2 + 1) - 1 / y) ** 3")

        evaluation_result = formula.evaluate({x: 2, y: 3})
        self.assertAlmostEqual(evaluation_result, 2744/27, places=12)

        self.assertAlmostEqual(formula.grads[x], 784/3, places=12)
        self.assertAlmostEqual(formula.grads[y], 196/27, places=12)


if __name__ == '__main__':
    unittest.main()
