import unittest
import math
from automatic_differentiation import Variable, sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, cbrt


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
        self.assertEqual(evaluation_result, 8.0)

        self.assertEqual(formula.grads[x], 12.0)
        self.assertAlmostEqual(formula.grads[y], 5.545177444479561, places=12)

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
        self.assertEqual(evaluation_result, 8.0)

        self.assertAlmostEqual(formula.grads[x], 5.545177444479561, places=12)

    def test_abs(self):
        x = Variable('x')

        formula = abs(x)
        self.assertEqual(str(formula), "abs(x)")

        evaluation_result = formula.evaluate({x: -5})
        self.assertEqual(evaluation_result, 5)

        self.assertEqual(formula.grads[x], -1)

    def test_exp(self):
        x = Variable('x')

        formula = exp(x)
        self.assertEqual(str(formula), "exp(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, 148.413159102576603, places=12)

        self.assertAlmostEqual(formula.grads[x], 148.413159102576603, places=12)

    def test_log(self):
        x = Variable('x')

        formula = log(x)
        self.assertEqual(str(formula), "log(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, 1.609437912434100, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.200000000000000, places=12)

    def test_log10(self):
        x = Variable('x')

        formula = log10(x)
        self.assertEqual(str(formula), "log10(x)")

        evaluation_result = formula.evaluate({x: 100})
        self.assertEqual(evaluation_result, 2.0)

        self.assertAlmostEqual(formula.grads[x], 0.004342944819033, places=12)

    def test_sin(self):
        x = Variable('x')

        formula = sin(x)
        self.assertEqual(str(formula), "sin(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, -0.958924274663137, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.283662185463226, places=12)

    def test_cos(self):
        x = Variable('x')

        formula = cos(x)
        self.assertEqual(str(formula), "cos(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, 0.283662185463226, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.958924274663137, places=12)

    def test_tan(self):
        x = Variable('x')

        formula = tan(x)
        self.assertEqual(str(formula), "tan(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, -3.380515006246586, places=12)

        self.assertAlmostEqual(formula.grads[x], 12.427881707458353, places=12)

    def test_sinh(self):
        x = Variable('x')

        formula = sinh(x)
        self.assertEqual(str(formula), "sinh(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, 74.203210577788759, places=12)

        self.assertAlmostEqual(formula.grads[x], 74.209948524787844, places=12)

    def test_cosh(self):
        x = Variable('x')

        formula = cosh(x)
        self.assertEqual(str(formula), "cosh(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, 74.209948524787844, places=12)

        self.assertAlmostEqual(formula.grads[x], 74.203210577788759, places=12)

    def test_tanh(self):
        x = Variable('x')

        formula = tanh(x)
        self.assertEqual(str(formula), "tanh(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, 0.999909204262595, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.000181583230944, places=12)

    def test_sqrt(self):
        x = Variable('x')

        formula = sqrt(x)
        self.assertEqual(str(formula), "sqrt(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, 2.236067977499790, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.223606797749979, places=12)

    def test_cbrt(self):
        x = Variable('x')

        formula = cbrt(x)
        self.assertEqual(str(formula), "cbrt(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, 1.709975946676697, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.113998396445113, places=12)

    def test_composition1(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = (x + y) * (x - y) / x ** z
        self.assertEqual(str(formula), "((x + y) * (x - y)) / (x ** z)")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertAlmostEqual(evaluation_result, -0.312500000000000, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.875000000000000, places=12)
        self.assertAlmostEqual(formula.grads[y], -0.375000000000000, places=12)
        self.assertAlmostEqual(formula.grads[z], 0.216608493924983, places=12)

    def test_composition2(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = x / (y * z)
        self.assertEqual(str(formula), "x / (y * z)")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertAlmostEqual(evaluation_result, 0.166666666666667, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.083333333333333, places=12)
        self.assertAlmostEqual(formula.grads[y], -0.055555555555556, places=12)
        self.assertAlmostEqual(formula.grads[z], -0.041666666666667, places=12)

    def test_composition3(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = x ** ((y * z) ** 0.5)
        self.assertEqual(str(formula), "x ** ((y * z) ** 0.5)")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertAlmostEqual(evaluation_result, 11.035664635963600, places=12)

        self.assertAlmostEqual(formula.grads[x], 19.114331844780100, places=12)
        self.assertAlmostEqual(formula.grads[y], 4.416348408832160, places=12)
        self.assertAlmostEqual(formula.grads[z], 3.312261306624120, places=12)

    def test_composition4(self):
        x = Variable('x')
        y = Variable('y')

        formula = ((x ** 2 + 1) - 1 / y) ** 3
        self.assertEqual(str(formula), "(((x ** 2 + 1) - 1 / y)) ** 3")

        evaluation_result = formula.evaluate({x: 2, y: 3})
        self.assertAlmostEqual(evaluation_result, 101.629629629629630, places=12)

        self.assertAlmostEqual(formula.grads[x], 261.333333333333333, places=12)
        self.assertAlmostEqual(formula.grads[y], 7.259259259259259, places=12)

    def test_composition5(self):
        x = Variable('x')
        y = Variable('y')

        formula = ((sin(x) + 1) - 1 / tan(y)) ** 3
        self.assertEqual(str(formula), "(((sin(x) + 1) - 1 / tan(y))) ** 3")

        evaluation_result = formula.evaluate({x: 2, y: 3})
        self.assertAlmostEqual(evaluation_result, 710.818918258531804, places=12)

        self.assertAlmostEqual(formula.grads[x], -99.435280740782561, places=12)
        self.assertAlmostEqual(formula.grads[y], 11998.217252828359505, places=12)

    def test_composition6(self):
        x = Variable('x')
        y = Variable('y')

        formula = sin(x) / cos(y)
        self.assertEqual(str(formula), "sin(x) / cos(y)")

        evaluation_result = formula.evaluate({x: 7, y: 13})
        self.assertAlmostEqual(evaluation_result, 0.723994632135732, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.830795006114286, places=12)
        self.assertAlmostEqual(formula.grads[y], 0.335224814811424, places=12)

    def test_composition7(self):
        x = Variable('x')

        formula = tan(cos(sin(x)))
        self.assertEqual(str(formula), "tan(cos(sin(x)))")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 1.012959705462661, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.932878230634677, places=12)

    def test_composition8(self):
        x = Variable('x')

        formula = exp(tan(cos(sin(x))))
        self.assertEqual(str(formula), "exp(tan(cos(sin(x))))")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 2.753739222747496, places=12)

        self.assertAlmostEqual(formula.grads[x], -2.568903373745995, places=12)

    def test_composition9(self):
        x = Variable('x')

        formula = sqrt(exp(tan(cos(sin(x)))))
        self.assertEqual(str(formula), "sqrt(exp(tan(cos(sin(x)))))")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 1.659439430273819, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.774027459779628, places=12)

    def test_composition10(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = sqrt(1 / ((x ** 2 + y ** 2) + z ** 2))
        self.assertEqual(str(formula), "sqrt(1 / (((x ** 2 + y ** 2) + z ** 2)))")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertAlmostEqual(evaluation_result, 0.185695338177052, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.012806575046693, places=12)
        self.assertAlmostEqual(formula.grads[y], -0.019209862570040, places=12)
        self.assertAlmostEqual(formula.grads[z], -0.025613150093386, places=12)

    def test_composition11(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = abs(-1 / ((x ** 2 + y ** 2) - z ** 2))
        self.assertEqual(str(formula), "abs(-1 / (((x ** 2 + y ** 2) - z ** 2)))")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertAlmostEqual(evaluation_result, 1 / 3, places=12)

        self.assertAlmostEqual(formula.grads[x], 4 / 9, places=12)
        self.assertAlmostEqual(formula.grads[y], 2 / 3, places=12)
        self.assertAlmostEqual(formula.grads[z], -8 / 9, places=12)

    def test_composition12(self):
        x = Variable('x')

        formula = x * exp(x)
        self.assertEqual(str(formula), "x * exp(x)")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 7676.432108999210195, places=12)

        self.assertAlmostEqual(formula.grads[x], 8773.065267427668794, places=12)

    def test_composition13(self):
        x = Variable('x')

        formula = cbrt(x * exp(x))
        self.assertEqual(str(formula), "cbrt(x * exp(x))")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 19.726640851995720, places=12)

        self.assertAlmostEqual(formula.grads[x], 7.514910800760274, places=12)


if __name__ == '__main__':
    unittest.main()
