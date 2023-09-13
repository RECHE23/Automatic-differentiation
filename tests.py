import unittest
import math
from automatic_differentiation import Variable


class TestVariable(unittest.TestCase):

    def test_variable_name(self):
        x = Variable('x')
        self.assertEqual(x.name, 'x')

        y = Variable('y')
        self.assertEqual(y.name, 'y')

        z = Variable('z')
        self.assertEqual(z.name, 'z')

    def test_variable_repr_str(self):
        x = Variable('x')
        self.assertEqual(repr(x), 'x')
        self.assertEqual(str(x), 'x')

    def test_variable_evaluate(self):
        x = Variable('x')
        self.assertEqual(x.evaluate({x: 5}), 5)

    def test_variable_compute_gradients(self):
        x = Variable('x')
        x.value = 5
        grads = x.compute_gradients()
        self.assertEqual(grads[x], 1.0)

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

    def test_raddition(self):
        x = Variable('x')

        formula = 2 + x
        self.assertEqual(str(formula), "(x + 2)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, 7)

        self.assertEqual(formula.grads[x], 1.0)

    def test_rsubtraction(self):
        x = Variable('x')

        formula = 2 - x
        self.assertEqual(str(formula), "((-x) + 2)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, -3)

        self.assertEqual(formula.grads[x], -1.0)

    def test_rmultiplication(self):
        x = Variable('x')

        formula = 2 * x
        self.assertEqual(str(formula), "x * 2")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, 10)

        self.assertEqual(formula.grads[x], 2.0)

    def test_rdivision(self):
        x = Variable('x')

        formula = 2 / x
        self.assertEqual(str(formula), "2 / x")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, 2 / 5)

        self.assertEqual(formula.grads[x], -0.08)

    def test_rpower(self):
        x = Variable('x')

        formula = 2 ** x
        self.assertEqual(str(formula), "2 ** x")

        evaluation_result = formula.evaluate({x: 3})
        self.assertEqual(evaluation_result, 8)

        self.assertEqual(formula.grads[x], 8 * math.log(2))

    def test_composition1(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = (x + y) * (x - y) / x ** z
        self.assertEqual(str(formula), "((x + y) * (x - y)) / (x ** z)")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertEqual(evaluation_result, -5 / 16)

        self.assertEqual(formula.grads[x], 0.875)
        self.assertEqual(formula.grads[y], -0.375)
        self.assertEqual(formula.grads[z], 5 / 16 * math.log(2))

    def test_composition2(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = x / (y * z)
        self.assertEqual(str(formula), "x / (y * z)")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertEqual(evaluation_result, 1/6)

        self.assertEqual(formula.grads[x], 1/12)
        self.assertEqual(formula.grads[y], -1/18)
        self.assertEqual(formula.grads[z], -1/24)

    def test_composition3(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = x ** ((y * z) ** 0.5)
        self.assertEqual(str(formula), "x ** ((y * z) ** 0.5)")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertEqual(evaluation_result, 4**(math.sqrt(3)))

        self.assertAlmostEqual(formula.grads[x], math.sqrt(3) * 4**(math.sqrt(3)), places=12)
        self.assertAlmostEqual(formula.grads[y], 4**(math.sqrt(3)) * math.log(2) / math.sqrt(3), places=12)
        self.assertAlmostEqual(formula.grads[z], math.sqrt(3) * 4**(math.sqrt(3) - 1) * math.log(2), places=12)

    def test_composition4(self):
        x = Variable('x')
        y = Variable('y')

        formula = ((x ** 2 + 1) - 1 / y) ** 3
        self.assertEqual(str(formula), "(((x ** 2 + 1) - 1 / y)) ** 3")

        evaluation_result = formula.evaluate({x: 2, y: 3})
        self.assertAlmostEqual(evaluation_result, 2744/27, places=12)

        self.assertAlmostEqual(formula.grads[x], 784/3, places=12)
        self.assertAlmostEqual(formula.grads[y], 196/27, places=12)


if __name__ == '__main__':
    unittest.main()
