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
        self.assertEqual(evaluation_result, math.exp(5))

        self.assertEqual(formula.grads[x], math.exp(5))

    def test_log(self):
        x = Variable('x')

        formula = log(x)
        self.assertEqual(str(formula), "log(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, math.log(5), places=12)

        self.assertAlmostEqual(formula.grads[x], 1 / 5, places=12)

    def test_log10(self):
        x = Variable('x')

        formula = log10(x)
        self.assertEqual(str(formula), "log10(x)")

        evaluation_result = formula.evaluate({x: 100})
        self.assertAlmostEqual(evaluation_result, 2.0, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.004342944819032518, places=12)

    def test_sin(self):
        x = Variable('x')

        formula = sin(x)
        self.assertEqual(str(formula), "sin(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, math.sin(5))

        self.assertEqual(formula.grads[x], math.cos(5))

    def test_cos(self):
        x = Variable('x')

        formula = cos(x)
        self.assertEqual(str(formula), "cos(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, math.cos(5))

        self.assertEqual(formula.grads[x], -math.sin(5))

    def test_tan(self):
        x = Variable('x')

        formula = tan(x)
        self.assertEqual(str(formula), "tan(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, math.tan(5))

        self.assertEqual(formula.grads[x], 1/math.cos(5)**2)

    def test_sinh(self):
        x = Variable('x')

        formula = sinh(x)
        self.assertEqual(str(formula), "sinh(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, math.sinh(5))

        self.assertEqual(formula.grads[x], math.cosh(5))

    def test_cosh(self):
        x = Variable('x')

        formula = cosh(x)
        self.assertEqual(str(formula), "cosh(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, math.cosh(5))

        self.assertEqual(formula.grads[x], math.sinh(5))

    def test_tanh(self):
        x = Variable('x')

        formula = tanh(x)
        self.assertEqual(str(formula), "tanh(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, math.tanh(5))

        self.assertEqual(formula.grads[x], 1/math.cosh(5)**2)

    def test_sqrt(self):
        x = Variable('x')

        formula = sqrt(x)
        self.assertEqual(str(formula), "sqrt(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertEqual(evaluation_result, math.sqrt(5))

        self.assertEqual(formula.grads[x], 0.5/math.sqrt(5))

    def test_cbrt(self):
        x = Variable('x')

        formula = cbrt(x)
        self.assertEqual(str(formula), "cbrt(x)")

        evaluation_result = formula.evaluate({x: 5})
        self.assertAlmostEqual(evaluation_result, 1.70997594667669698935310887, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.11399839644511313, places=12)

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

    def test_composition5(self):
        x = Variable('x')
        y = Variable('y')

        formula = ((sin(x) + 1) - 1 / tan(y)) ** 3
        self.assertEqual(str(formula), "(((sin(x) + 1) - 1 / tan(y))) ** 3")

        evaluation_result = formula.evaluate({x: 2, y: 3})
        self.assertAlmostEqual(evaluation_result, 710.81891825853180, places=12)

        self.assertAlmostEqual(formula.grads[x], -99.43528074078256, places=12)
        self.assertAlmostEqual(formula.grads[y], 11998.217252828359, places=12)

    def test_composition6(self):
        x = Variable('x')
        y = Variable('y')

        formula = sin(x) / cos(y)
        self.assertEqual(str(formula), "sin(x) / cos(y)")

        evaluation_result = formula.evaluate({x: 7, y: 13})
        self.assertAlmostEqual(evaluation_result, 0.723994632135732319226, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.8307950061142856, places=12)
        self.assertAlmostEqual(formula.grads[y], 0.3352248148114238, places=12)

    def test_composition7(self):
        x = Variable('x')

        formula = tan(cos(sin(x)))
        self.assertEqual(str(formula), "tan(cos(sin(x)))")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 1.0129597054626613972468, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.9328782306346769, places=12)

    def test_composition8(self):
        x = Variable('x')

        formula = exp(tan(cos(sin(x))))
        self.assertEqual(str(formula), "exp(tan(cos(sin(x))))")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 2.7537392227474955665619, places=12)

        self.assertAlmostEqual(formula.grads[x], -2.5689033737459943, places=12)

    def test_composition9(self):
        x = Variable('x')

        formula = sqrt(exp(tan(cos(sin(x)))))
        self.assertEqual(str(formula), "sqrt(exp(tan(cos(sin(x)))))")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 1.6594394302738185709155, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.7740274597796282, places=12)

    def test_composition10(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = sqrt(1 / ((x ** 2 + y ** 2) + z ** 2))
        self.assertEqual(str(formula), "sqrt(1 / (((x ** 2 + y ** 2) + z ** 2)))")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertAlmostEqual(evaluation_result, 0.1856953381770518631, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.01280657504669323, places=12)
        self.assertAlmostEqual(formula.grads[y], -0.01920986257003984, places=12)
        self.assertAlmostEqual(formula.grads[z], -0.02561315009338646, places=12)

    def test_composition11(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = abs(-1 / ((x ** 2 + y ** 2) - z ** 2))
        self.assertEqual(str(formula), "abs(-1 / (((x ** 2 + y ** 2) - z ** 2)))")

        evaluation_result = formula.evaluate({x: 2, y: 3, z: 4})
        self.assertAlmostEqual(evaluation_result, 1/3, places=12)

        self.assertAlmostEqual(formula.grads[x], 4/9, places=12)
        self.assertAlmostEqual(formula.grads[y], 2/3, places=12)
        self.assertAlmostEqual(formula.grads[z], -8/9, places=12)

    def test_composition12(self):
        x = Variable('x')

        formula = x * exp(x)
        self.assertEqual(str(formula), "x * exp(x)")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 7 * math.exp(7), places=12)

        self.assertAlmostEqual(formula.grads[x], 8 * math.exp(7), places=12)

    def test_composition13(self):
        x = Variable('x')

        formula = cbrt(x * exp(x))
        self.assertEqual(str(formula), "cbrt(x * exp(x))")

        evaluation_result = formula.evaluate({x: 7})
        self.assertAlmostEqual(evaluation_result, 19.7266408519957203353551, places=12)

        self.assertAlmostEqual(formula.grads[x], 7.514910800760276, places=12)


if __name__ == '__main__':
    unittest.main()
