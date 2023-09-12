import unittest
import math
from automatic_differentiation import Variable


class TestVariable(unittest.TestCase):
    def test_addition(self):
        x = Variable(5)
        y = Variable(3)

        result = x + y
        self.assertEqual(result.value, 8)

        gradients = result.compute_gradients(x, y)
        self.assertEqual(gradients[x], 1.0)
        self.assertEqual(gradients[y], 1.0)

    def test_subtraction(self):
        x = Variable(5)
        y = Variable(3)

        result = x - y
        self.assertEqual(result.value, 2)

        gradients = result.compute_gradients(x, y)
        self.assertEqual(gradients[x], 1.0)
        self.assertEqual(gradients[y], -1.0)

    def test_multiplication(self):
        x = Variable(5)
        y = Variable(3)

        result = x * y
        self.assertEqual(result.value, 15)

        gradients = result.compute_gradients(x, y)
        self.assertEqual(gradients[x], 3.0)
        self.assertEqual(gradients[y], 5.0)

    def test_division(self):
        x = Variable(10)
        y = Variable(2)

        result = x / y
        self.assertEqual(result.value, 5)

        gradients = result.compute_gradients(x, y)
        self.assertEqual(gradients[x], 0.5)
        self.assertEqual(gradients[y], -2.5)

    def test_power(self):
        x = Variable(2)
        y = Variable(3)

        result = x ** y
        self.assertEqual(result.value, 8)

        gradients = result.compute_gradients(x, y)
        self.assertEqual(gradients[x], 12.0)
        self.assertEqual(gradients[y], 8.0 * math.log(2))

    def test_negative(self):
        x = Variable(5)

        result = -x
        self.assertEqual(result.value, -5)

        gradients = result.compute_gradients(x)
        self.assertEqual(gradients[x], -1.0)

    def test_composition(self):
        x = Variable(2)
        y = Variable(3)
        z = Variable(4)

        result = (x + y) * (x - y) / (x ** z)
        self.assertEqual(result.value, -5 / 16)

        gradients = result.compute_gradients(x, y, z)
        self.assertEqual(gradients[x], 0.875)
        self.assertEqual(gradients[y], -0.375)
        self.assertEqual(gradients[z], 5 / 16 * math.log(2))

    def test_advanced_composition(self):
        x = Variable(2)
        y = Variable(3)

        result = (x**2 + 1 - 1 / y)**3
        self.assertAlmostEqual(result.value, 2744/27, places=12)

        gradients = result.compute_gradients(x, y)
        self.assertAlmostEqual(gradients[x], 784/3, places=12)
        self.assertAlmostEqual(gradients[y], 196/27, places=12)


if __name__ == '__main__':
    unittest.main()
