import unittest

import numpy as np

from automatic_differentiation import *


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
        self.assertEqual(repr(x), "Variable(name='x')")
        self.assertEqual(str(x), 'x')

    def test_nodes_and_constants_repr_str(self):
        x = Variable('x')
        f = x + 1
        self.assertEqual(repr(f), "Node(name='x + 1', operation='add', operands=('x', 1))")
        self.assertEqual(str(f), 'x + 1')
        x.at = 1
        self.assertEqual(repr(f), "Node(name='x + 1', operation='add', operands=('x', 1), value=2)")
        x.at = {'x': 2}
        self.assertEqual(repr(f), "Node(name='x + 1', operation='add', operands=('x', 1), value=3)")
        x.evaluate_at(x=3)
        self.assertEqual(repr(f), "Node(name='x + 1', operation='add', operands=('x', 1), value=4)")
        x.evaluate_at(4)
        self.assertEqual(repr(f), "Node(name='x + 1', operation='add', operands=('x', 1), value=5)")

        one = f.operands[1]
        self.assertEqual(repr(one), "Constant(name='1', value=1)")
        self.assertEqual(str(one), '1')

        with np.testing.assert_raises(TypeError):
            one.at = 2

        alpha = Constant(1)
        beta = Constant(1)
        gamma = Constant(2)
        self.assertEqual(alpha, beta)
        self.assertNotEqual(beta, gamma)

    def test_variable_evaluate(self):
        x = Variable('x')
        self.assertEqual(x.evaluate_at(x=5), 5)
        x.at = 6
        self.assertEqual(x.value, 6)

    def test_variable_compute_gradients(self):
        x = Variable('x')
        x.value = 5
        grads = x.compute_gradients()
        self.assertEqual(grads[x], 1.0)
        grads = x.evaluate_gradients_at(6)
        self.assertEqual(grads[x], 1.0)


class TestOperations(unittest.TestCase):

    def test_addition(self):
        x = Variable('x')
        y = Variable('y')

        formula = x + y
        self.assertEqual(str(formula), "x + y")

        evaluation_result = formula.evaluate_at(x=5, y=3)
        self.assertEqual(evaluation_result, 8)

        self.assertEqual(formula.grads[x], 1.0)
        self.assertEqual(formula.grads[y], 1.0)

    def test_subtraction(self):
        x = Variable('x')
        y = Variable('y')

        formula = x - y
        self.assertEqual(str(formula), "x - y")

        evaluation_result = formula.evaluate_at(x=5, y=3)
        self.assertEqual(evaluation_result, 2)

        self.assertEqual(formula.grads[x], 1.0)
        self.assertEqual(formula.grads[y], -1.0)

    def test_multiplication(self):
        x = Variable('x')
        y = Variable('y')

        formula = x * y
        self.assertEqual(str(formula), "x * y")

        evaluation_result = formula.evaluate_at(x=5, y=3)
        self.assertEqual(evaluation_result, 15)

        self.assertEqual(formula.grads[x], 3.0)
        self.assertEqual(formula.grads[y], 5.0)

    def test_division(self):
        x = Variable('x')
        y = Variable('y')

        formula = x / y
        self.assertEqual(str(formula), "x / y")

        evaluation_result = formula.evaluate_at(x=10, y=2)
        self.assertEqual(evaluation_result, 5)

        self.assertEqual(formula.grads[x], 0.5)
        self.assertEqual(formula.grads[y], -2.5)

    def test_power(self):
        x = Variable('x')
        y = Variable('y')

        formula = x ** y
        self.assertEqual(str(formula), "x ** y")

        evaluation_result = formula.evaluate_at(x=2, y=3)
        self.assertEqual(evaluation_result, 8.0)

        self.assertEqual(formula.grads[x], 12.0)
        self.assertAlmostEqual(formula.grads[y], 5.545177444479561, places=12)

    def test_negative(self):
        x = Variable('x')

        formula = -x
        self.assertEqual(str(formula), "-x")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertEqual(evaluation_result, -5)

        self.assertEqual(formula.grads[x], -1.0)

    def test_raddition(self):
        x = Variable('x')

        formula = 2 + x
        self.assertEqual(str(formula), "2 + x")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertEqual(evaluation_result, 7)

        self.assertEqual(formula.grads[x], 1.0)

    def test_rsubtraction(self):
        x = Variable('x')

        formula = 2 - x
        self.assertEqual(str(formula), "2 - x")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertEqual(evaluation_result, -3)

        self.assertEqual(formula.grads[x], -1.0)

    def test_rmultiplication(self):
        x = Variable('x')

        formula = 2 * x
        self.assertEqual(str(formula), "2 * x")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertEqual(evaluation_result, 10)

        self.assertEqual(formula.grads[x], 2.0)

    def test_rdivision(self):
        x = Variable('x')

        formula = 2 / x
        self.assertEqual(str(formula), "2 / x")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertEqual(evaluation_result, 2 / 5)

        self.assertEqual(formula.grads[x], -0.08)

    def test_rpower(self):
        x = Variable('x')

        formula = 2 ** x
        self.assertEqual(str(formula), "2 ** x")

        evaluation_result = formula.evaluate_at(x=3)
        self.assertEqual(evaluation_result, 8.0)

        self.assertAlmostEqual(formula.grads[x], 5.545177444479561, places=12)

    def test_abs(self):
        x = Variable('x')

        formula = abs(x)
        self.assertEqual(str(formula), "abs(x)")

        evaluation_result = formula.evaluate_at(x=-5)
        self.assertEqual(evaluation_result, 5)

        self.assertEqual(formula.grads[x], -1)

    def test_exp(self):
        x = Variable('x')

        formula = exp(x)
        self.assertEqual(str(formula), "exp(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, 148.413159102576603, places=12)

        self.assertAlmostEqual(formula.grads[x], 148.413159102576603, places=12)

    def test_log(self):
        x = Variable('x')

        formula = log(x)
        self.assertEqual(str(formula), "log(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, 1.609437912434100, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.200000000000000, places=12)

    def test_log10(self):
        x = Variable('x')

        formula = log10(x)
        self.assertEqual(str(formula), "log10(x)")

        evaluation_result = formula.evaluate_at(x=100)
        self.assertEqual(evaluation_result, 2.0)

        self.assertAlmostEqual(formula.grads[x], 0.004342944819033, places=12)

    def test_sin(self):
        x = Variable('x')

        formula = sin(x)
        self.assertEqual(str(formula), "sin(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, -0.958924274663137, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.283662185463226, places=12)

    def test_cos(self):
        x = Variable('x')

        formula = cos(x)
        self.assertEqual(str(formula), "cos(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, 0.283662185463226, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.958924274663137, places=12)

    def test_tan(self):
        x = Variable('x')

        formula = tan(x)
        self.assertEqual(str(formula), "tan(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, -3.380515006246586, places=12)

        self.assertAlmostEqual(formula.grads[x], 12.427881707458353, places=12)

    def test_sinh(self):
        x = Variable('x')

        formula = sinh(x)
        self.assertEqual(str(formula), "sinh(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, 74.203210577788759, places=12)

        self.assertAlmostEqual(formula.grads[x], 74.209948524787844, places=12)

    def test_cosh(self):
        x = Variable('x')

        formula = cosh(x)
        self.assertEqual(str(formula), "cosh(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, 74.209948524787844, places=12)

        self.assertAlmostEqual(formula.grads[x], 74.203210577788759, places=12)

    def test_tanh(self):
        x = Variable('x')

        formula = tanh(x)
        self.assertEqual(str(formula), "tanh(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, 0.999909204262595, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.000181583230944, places=12)

    def test_asin(self):
        x = Variable('x')

        formula = asin(x)
        self.assertEqual(str(formula), "asin(x)")

        evaluation_result = formula.evaluate_at(x=0.5)
        self.assertAlmostEqual(evaluation_result, 0.523598775598299, places=12)

        self.assertAlmostEqual(formula.grads[x], 1.154700538379250, places=12)

    def test_acos(self):
        x = Variable('x')

        formula = acos(x)
        self.assertEqual(str(formula), "acos(x)")

        evaluation_result = formula.evaluate_at(x=0.5)
        self.assertAlmostEqual(evaluation_result, 1.047197551196600, places=12)

        self.assertAlmostEqual(formula.grads[x], -1.154700538379250, places=12)

    def test_atan(self):
        x = Variable('x')

        formula = atan(x)
        self.assertEqual(str(formula), "atan(x)")

        evaluation_result = formula.evaluate_at(x=0.5)
        self.assertAlmostEqual(evaluation_result, 0.463647609000806, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.800000000000000, places=12)

    def test_asinh(self):
        x = Variable('x')

        formula = asinh(x)
        self.assertEqual(str(formula), "asinh(x)")

        evaluation_result = formula.evaluate_at(x=1.5)
        self.assertAlmostEqual(evaluation_result, 1.194763217287110, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.554700196225229, places=12)

    def test_acosh(self):
        x = Variable('x')

        formula = acosh(x)
        self.assertEqual(str(formula), "acosh(x)")

        evaluation_result = formula.evaluate_at(x=2.0)
        self.assertAlmostEqual(evaluation_result, 1.316957896924817, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.577350269189626, places=12)

    def test_atanh(self):
        x = Variable('x')

        formula = atanh(x)
        self.assertEqual(str(formula), "atanh(x)")

        evaluation_result = formula.evaluate_at(x=0.5)
        self.assertAlmostEqual(evaluation_result, 0.549306144334055, places=12)

        self.assertAlmostEqual(formula.grads[x], 1.333333333333330, places=12)

    def test_sqrt(self):
        x = Variable('x')

        formula = sqrt(x)
        self.assertEqual(str(formula), "sqrt(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, 2.236067977499790, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.223606797749979, places=12)

    def test_cbrt(self):
        x = Variable('x')

        formula = cbrt(x)
        self.assertEqual(str(formula), "cbrt(x)")

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, 1.709975946676697, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.113998396445113, places=12)

    def test_erf(self):
        x = Variable('x')

        formula = erf(x)
        self.assertEqual(str(formula), "erf(x)")

        evaluation_result = formula.evaluate_at(x=0.5)
        self.assertAlmostEqual(evaluation_result, 0.520499877813047, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.878782578935445, places=12)

    def test_erfc(self):
        x = Variable('x')

        formula = erfc(x)
        self.assertEqual(str(formula), "erfc(x)")

        evaluation_result = formula.evaluate_at(x=0.5)
        self.assertAlmostEqual(evaluation_result, 0.479500122186953, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.878782578935445, places=12)

    def test_transpose(self):
        x = Variable('x')

        formula = x.T
        self.assertEqual(str(formula), "x.T")

        evaluation_result = formula.evaluate_at(x=np.array([[0, 1], [2, 3]]))
        np.testing.assert_array_equal(evaluation_result, np.array([[0, 2], [1, 3]]))


class TestCompositions(unittest.TestCase):

    def test_composition1(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = (x + y) * (x - y) / x ** z
        self.assertEqual(str(formula), "(x + y) * (x - y) / x ** z")

        evaluation_result = formula.evaluate_at(x=2, y=3, z=4)
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

        evaluation_result = formula.evaluate_at(x=2, y=3, z=4)
        self.assertAlmostEqual(evaluation_result, 0.166666666666667, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.083333333333333, places=12)
        self.assertAlmostEqual(formula.grads[y], -0.055555555555556, places=12)
        self.assertAlmostEqual(formula.grads[z], -0.041666666666667, places=12)

    def test_composition3(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = x ** ((y * z) ** 0.5)
        self.assertEqual(str(formula), "x ** (y * z) ** 0.5")

        evaluation_result = formula.evaluate_at(x=2, y=3, z=4)
        self.assertAlmostEqual(evaluation_result, 11.035664635963600, places=12)

        self.assertAlmostEqual(formula.grads[x], 19.114331844780100, places=12)
        self.assertAlmostEqual(formula.grads[y], 4.416348408832160, places=12)
        self.assertAlmostEqual(formula.grads[z], 3.312261306624120, places=12)

    def test_composition4(self):
        x = Variable('x')
        y = Variable('y')

        formula = ((x ** 2 + 1) - 1 / y) ** 3
        self.assertEqual(str(formula), "(x ** 2 + 1 - 1 / y) ** 3")

        evaluation_result = formula.evaluate_at(x=2, y=3)
        self.assertAlmostEqual(evaluation_result, 101.629629629629630, places=12)

        self.assertAlmostEqual(formula.grads[x], 261.333333333333333, places=12)
        self.assertAlmostEqual(formula.grads[y], 7.259259259259259, places=12)

    def test_composition5(self):
        x = Variable('x')
        y = Variable('y')

        formula = ((sin(x) + 1) - 1 / tan(y)) ** 3
        self.assertEqual(str(formula), "(sin(x) + 1 - 1 / tan(y)) ** 3")

        evaluation_result = formula.evaluate_at(x=2, y=3)
        self.assertAlmostEqual(evaluation_result, 710.818918258531804, places=12)

        self.assertAlmostEqual(formula.grads[x], -99.435280740782561, places=12)
        self.assertAlmostEqual(formula.grads[y], 11998.217252828359505, places=12)

    def test_composition6(self):
        x = Variable('x')
        y = Variable('y')

        formula = sin(x) / cos(y)
        self.assertEqual(str(formula), "sin(x) / cos(y)")

        evaluation_result = formula.evaluate_at(x=7, y=13)
        self.assertAlmostEqual(evaluation_result, 0.723994632135732, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.830795006114286, places=12)
        self.assertAlmostEqual(formula.grads[y], 0.335224814811424, places=12)

    def test_composition7(self):
        x = Variable('x')

        formula = tan(cos(sin(x)))
        self.assertEqual(str(formula), "tan(cos(sin(x)))")

        evaluation_result = formula.evaluate_at(x=7)
        self.assertAlmostEqual(evaluation_result, 1.012959705462661, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.932878230634677, places=12)

    def test_composition8(self):
        x = Variable('x')

        formula = exp(tan(cos(sin(x))))
        self.assertEqual(str(formula), "exp(tan(cos(sin(x))))")

        evaluation_result = formula.evaluate_at(x=7)
        self.assertAlmostEqual(evaluation_result, 2.753739222747496, places=12)

        self.assertAlmostEqual(formula.grads[x], -2.568903373745995, places=12)

    def test_composition9(self):
        x = Variable('x')

        formula = sqrt(exp(tan(cos(sin(x)))))
        self.assertEqual(str(formula), "sqrt(exp(tan(cos(sin(x)))))")

        evaluation_result = formula.evaluate_at(x=7)
        self.assertAlmostEqual(evaluation_result, 1.659439430273819, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.774027459779628, places=12)

    def test_composition10(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = sqrt(1 / ((x ** 2 + y ** 2) + z ** 2))
        self.assertEqual(str(formula), "sqrt(1 / (x ** 2 + y ** 2 + z ** 2))")

        evaluation_result = formula.evaluate_at(x=2, y=3, z=4)
        self.assertAlmostEqual(evaluation_result, 0.185695338177052, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.012806575046693, places=12)
        self.assertAlmostEqual(formula.grads[y], -0.019209862570040, places=12)
        self.assertAlmostEqual(formula.grads[z], -0.025613150093386, places=12)

    def test_composition11(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = abs(-1 / ((x ** 2 + y ** 2) - z ** 2))
        self.assertEqual(str(formula), "abs(-1 / (x ** 2 + y ** 2 - z ** 2))")

        evaluation_result = formula.evaluate_at(x=2, y=3, z=4)
        self.assertAlmostEqual(evaluation_result, 1 / 3, places=12)

        self.assertAlmostEqual(formula.grads[x], 4 / 9, places=12)
        self.assertAlmostEqual(formula.grads[y], 2 / 3, places=12)
        self.assertAlmostEqual(formula.grads[z], -8 / 9, places=12)

    def test_composition12(self):
        x = Variable('x')

        formula = x * exp(x)
        self.assertEqual(str(formula), "x * exp(x)")

        evaluation_result = formula.evaluate_at(x=7)
        self.assertAlmostEqual(evaluation_result, 7676.432108999210195, places=12)

        self.assertAlmostEqual(formula.grads[x], 8773.065267427668794, places=12)

    def test_composition13(self):
        x = Variable('x')

        formula = cbrt(x * exp(x))
        self.assertEqual(str(formula), "cbrt(x * exp(x))")

        evaluation_result = formula.evaluate_at(x=7)
        self.assertAlmostEqual(evaluation_result, 19.726640851995720, places=12)

        self.assertAlmostEqual(formula.grads[x], 7.514910800760274, places=12)

    def test_composition14(self):
        x = Variable('x')
        y = Variable('y')

        formula = (x ** 2 + y ** 2) ** 0.5
        self.assertEqual(str(formula), "(x ** 2 + y ** 2) ** 0.5")

        evaluation_result = formula.evaluate_at(x=3, y=4)
        self.assertAlmostEqual(evaluation_result, 5.0, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.6, places=12)
        self.assertAlmostEqual(formula.grads[y], 0.8, places=12)

    def test_composition15(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = sin(x + y + z)
        self.assertEqual(str(formula), "sin(x + y + z)")

        evaluation_result = formula.evaluate_at(x=1, y=2, z=3)
        self.assertAlmostEqual(evaluation_result, -0.279415498198926, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.960170286650366, places=12)
        self.assertAlmostEqual(formula.grads[y], 0.960170286650366, places=12)
        self.assertAlmostEqual(formula.grads[z], 0.960170286650366, places=12)

    def test_composition16(self):
        x = Variable('x')
        y = Variable('y')

        formula = log(x * y)
        self.assertEqual(str(formula), "log(x * y)")

        evaluation_result = formula.evaluate_at(x=2, y=3)
        self.assertAlmostEqual(evaluation_result, 1.791759469228055, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.500000000000000, places=12)
        self.assertAlmostEqual(formula.grads[y], 0.333333333333333, places=12)

    def test_composition17(self):
        x = Variable('x')
        y = Variable('y')

        formula = cos(x + y) / (sin(x) + cos(y))
        self.assertEqual(str(formula), "cos(x + y) / (sin(x) + cos(y))")

        evaluation_result = formula.evaluate_at(x=1, y=2)
        self.assertAlmostEqual(evaluation_result, -2.327618830599548, places=12)

        self.assertAlmostEqual(formula.grads[x], 2.625051546827947, places=12)
        self.assertAlmostEqual(formula.grads[y], -5.307993516443740, places=12)

    def test_composition18(self):
        x = Variable('x')
        y = Variable('y')

        formula = sqrt(x ** 2 + y ** 2) - x
        self.assertEqual(str(formula), "sqrt(x ** 2 + y ** 2) - x")

        evaluation_result = formula.evaluate_at(x=3, y=4)
        self.assertAlmostEqual(evaluation_result, 2.000000000000000, places=12)

        self.assertAlmostEqual(formula.grads[x], -0.400000000000000, places=12)
        self.assertAlmostEqual(formula.grads[y], 0.800000000000000, places=12)

    def test_composition19(self):
        x = Variable('x')
        y = Variable('y')

        formula = (exp(x) + exp(y)) ** 2
        self.assertEqual(str(formula), "(exp(x) + exp(y)) ** 2")

        evaluation_result = formula.evaluate_at(x=2, y=3)
        self.assertAlmostEqual(evaluation_result, 754.853261731032567, places=12)

        self.assertAlmostEqual(formula.grads[x], 406.022618271441685, places=12)
        self.assertAlmostEqual(formula.grads[y], 1103.683905190623452, places=12)

    def test_composition20(self):
        x = Variable('x')
        y = Variable('y')

        formula = tanh(x) * cosh(y)
        self.assertEqual(str(formula), "tanh(x) * cosh(y)")

        evaluation_result = formula.evaluate_at(x=0.5, y=1.0)
        self.assertAlmostEqual(evaluation_result, 0.713084036383792, places=12)

        self.assertAlmostEqual(formula.grads[x], 1.213552267034070, places=12)
        self.assertAlmostEqual(formula.grads[y], 0.543080634815244, places=12)

    def test_composition21(self):
        x = Variable('x')

        formula = -(x + 1)
        self.assertEqual(str(formula), "-(x + 1)")

        evaluation_result = formula.evaluate_at(x=7)
        self.assertEqual(evaluation_result, -8)

        self.assertEqual(formula.grads[x], -1)


class TestMultipleEvaluations(unittest.TestCase):

    def test_multiple_evaluations1(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        formula = (x + y) * (x - y) / x ** z
        self.assertEqual(str(formula), "(x + y) * (x - y) / x ** z")

        evaluation_result = formula.evaluate_at(x=2, y=3, z=4)
        self.assertAlmostEqual(evaluation_result, -0.312500000000000, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.875000000000000, places=12)
        self.assertAlmostEqual(formula.grads[y], -0.375000000000000, places=12)
        self.assertAlmostEqual(formula.grads[z], 0.216608493924983, places=12)

        evaluation_result = formula.evaluate_at(x=5, y=6, z=7)
        self.assertAlmostEqual(evaluation_result, -0.000140800000000, places=12)

        self.assertAlmostEqual(formula.grads[x], 0.000325120000000, places=12)
        self.assertAlmostEqual(formula.grads[y], -0.000153600000000, places=12)
        self.assertAlmostEqual(formula.grads[z], 0.000226608858071, places=12)

    def test_multiple_evaluations2(self):
        x = Variable('x')

        formula = sqrt(exp(tan(cos(sin(x)))))
        self.assertEqual(str(formula), "sqrt(exp(tan(cos(sin(x)))))")

        evaluation_result = formula.evaluate_at(x=3)
        self.assertAlmostEqual(evaluation_result, 2.142421022934373, places=12)
        self.assertAlmostEqual(formula.grads[x], 0.495538281180751, places=12)

        evaluation_result = formula.evaluate_at(x=5)
        self.assertAlmostEqual(evaluation_result, 1.382090969428066, places=12)
        self.assertAlmostEqual(formula.grads[x], 0.227670154879172, places=12)

        evaluation_result = formula.evaluate_at(x=7)
        self.assertAlmostEqual(evaluation_result, 1.659439430273819, places=12)
        self.assertAlmostEqual(formula.grads[x], -0.774027459779628, places=12)


class TestNumpyArrayOperations(unittest.TestCase):

    def test_addition(self):
        x = Variable('x')
        y = Variable('y')

        formula = x + y
        x_val = np.array([1.0, 2.0, 3.0])
        y_val = np.array([4.0, 5.0, 6.0])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_equal(result, expected_result)

        grads = formula.grads
        self.assertTrue(np.array_equal(grads[x], np.array([1.0, 1.0, 1.0])))
        self.assertTrue(np.array_equal(grads[y], np.array([1.0, 1.0, 1.0])))

    def test_array_subtraction(self):
        x = Variable('x')
        y = Variable('y')

        formula = x - y
        x_val = np.array([1.0, 2.0, 3.0])
        y_val = np.array([4.0, 5.0, 6.0])

        result = formula.evaluate_at(x=x_val, y=y_val)
        np.testing.assert_array_equal(result, np.array([-3.0, -3.0, -3.0]))

        grads = formula.grads
        self.assertTrue(np.array_equal(grads[x], np.array([1.0, 1.0, 1.0])))
        self.assertTrue(np.array_equal(grads[y], np.array([-1.0, -1.0, -1.0])))

    def test_array_multiplication(self):
        x = Variable('x')
        y = Variable('y')

        formula = x * y
        x_val = np.array([1.0, 2.0, 3.0])
        y_val = np.array([4.0, 5.0, 6.0])

        result = formula.evaluate_at(x=x_val, y=y_val)
        np.testing.assert_array_equal(result, np.array([4.0, 10.0, 18.0]))

        grads = formula.grads
        self.assertTrue(np.array_equal(grads[x], np.array([4.0, 5.0, 6.0])))
        self.assertTrue(np.array_equal(grads[y], np.array([1.0, 2.0, 3.0])))

    def test_array_division(self):
        x = Variable('x')
        y = Variable('y')

        formula = x / y
        x_val = np.array([1.0, 2.0, 3.0])
        y_val = np.array([4.0, 5.0, 6.0])

        result = formula.evaluate_at(x=x_val, y=y_val)
        np.testing.assert_array_equal(result, np.array([1 / 4, 2 / 5, 1 / 2]))

        grads = formula.grads
        self.assertTrue(np.array_equal(grads[x], np.array([1 / 4, 1 / 5, 1 / 6])))
        self.assertTrue(np.array_equal(grads[y], np.array([-1 / 16, -2 / 25, -1 / 12])))

    def test_array_exponentiation(self):
        x = Variable('x')
        y = Variable('y')

        formula = x ** y
        x_val = np.array([2.0, 3.0, 4.0])
        y_val = np.array([2.0, 3.0, 4.0])

        result = formula.evaluate_at(x=x_val, y=y_val)
        np.testing.assert_array_equal(result, np.array([4.0, 27.0, 256.0]))

        grads = formula.grads
        self.assertTrue(np.allclose(grads[x], y_val * (x_val ** (y_val - 1))))
        self.assertTrue(np.allclose(grads[y], (x_val ** y_val) * np.log(x_val)))

    def test_sin(self):
        x = Variable('x')

        formula = sin(x)
        x_val = np.array([0.0, np.pi / 2, np.pi])

        result = formula.evaluate_at(x=x_val)
        expected_result = np.sin(x_val)
        np.testing.assert_allclose(result, expected_result)

        grads = formula.grads
        self.assertTrue(np.allclose(grads[x], np.array([1.0, 0.0, -1.0])))

    def test_exp(self):
        x = Variable('x')

        formula = exp(x)
        x_val = np.array([0.0, 1.0, 2.0])

        result = formula.evaluate_at(x=x_val)
        np.testing.assert_allclose(result, np.array([1.0, 2.718281828459045, 7.389056098930650]))

        grads = formula.grads
        self.assertTrue(np.array_equal(grads[x], np.array([1.0, 2.718281828459045, 7.389056098930650])))

    def test_matmul_operation1(self):
        x = Variable('x')
        y = Variable('y')
        formula = x @ y

        x_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_val = np.array([[2.0], [1.0]])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([[4.0], [10.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[2.0, 1.0], [2.0, 1.0]])
        expected_grad_y = np.array([[4.0], [6.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_matmul_operation2(self):
        x = Variable('x')
        y = Variable('y')
        formula = x @ y

        x_val = np.array([[1.0, 2.0, 5.0], [5.0, 3.0, 4.0], [7.0, 3.0, 11.0]])
        y_val = np.array([[2.0], [5.0], [1.0]])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([[17.0], [29.0], [40.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[2.0, 5.0, 1.0], [2.0, 5.0, 1.0], [2.0, 5.0, 1.0]])
        expected_grad_y = np.array([[13.0], [8.0], [20.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_matmul_operation3(self):
        x = Variable('x')
        y = Variable('y')
        formula = x @ y

        x_val = np.array([[1.0, 2.0, 5.0], [5.0, 3.0, 4.0], [7.0, 3.0, 11.0]])
        y_val = np.array([[3.0, 5.0, 11.0], [2.0, 7.0, 4.0], [7.0, 2.0, 10.0]])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([[42.0, 29.0, 69.0], [49.0, 54.0, 107.0], [104.0, 78.0, 199.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[19.0, 13.0, 19.0], [19.0, 13.0, 19.0], [19.0, 13.0, 19.0]])
        expected_grad_y = np.array([[13.0, 13.0, 13.0], [8.0, 8.0, 8.0], [20.0, 20.0, 20.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_matmul_operation4(self):
        x = Variable('x')
        y = Variable('y')
        formula = x @ y

        x_val = np.array([[1.0, -2.0, 5.0], [5.0, 3.0, -4.0], [7.0, 3.0, 11.0]])
        y_val = np.array([[3.0, 5.0, -11.0], [2.0, -7.0, 4.0], [7.0, 2.0, -10.0]])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([[34.0, 29.0, -69.0], [-7.0, -4.0, -3.0], [104.0, 36.0, -175.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[-3.0, -1.0, -1.0], [-3.0, -1.0, -1.0], [-3.0, -1.0, -1.0]])
        expected_grad_y = np.array([[13.0, 13.0, 13.0], [4.0, 4.0, 4.0], [12.0, 12.0, 12.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_matmul_operation5(self):
        x = Variable('x')
        y = Variable('y')
        formula = x @ y

        x_val = np.arange(3 ** 3).reshape((3, 3, 3))
        y_val = np.arange(3 ** 3).reshape((3, 3, 3))

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array(
            [[[15.0, 18.0, 21.0], [42.0, 54.0, 66.0], [69.0, 90.0, 111.0]], [[366.0, 396.0, 426.0], [474.0, 513.0, 552.0], [582.0, 630.0, 678.0]],
             [[1203.0, 1260.0, 1317.0], [1392.0, 1458.0, 1524.0], [1581.0, 1656.0, 1731.0]]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array(
            [[[3.0, 12.0, 21.0], [3.0, 12.0, 21.0], [3.0, 12.0, 21.0]], [[30.0, 39.0, 48.0], [30.0, 39.0, 48.0], [30.0, 39.0, 48.0]],
             [[57.0, 66.0, 75.0], [57.0, 66.0, 75.0], [57.0, 66.0, 75.0]]])
        expected_grad_y = np.array(
            [[[9.0, 9.0, 9.0], [12.0, 12.0, 12.0], [15.0, 15.0, 15.0]], [[36.0, 36.0, 36.0], [39.0, 39.0, 39.0], [42.0, 42.0, 42.0]],
             [[63.0, 63.0, 63.0], [66.0, 66.0, 66.0], [69.0, 69.0, 69.0]]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_transpose(self):
        x = Variable('x')
        y = Variable('y')
        formula = transpose(x, axes=(1, 0, 2)) @ y

        x_val = np.arange(3 ** 3).reshape((3, 3, 3))
        y_val = np.arange(3 ** 3).reshape((3, 3, 3))

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array(
            [[[15.0, 18.0, 21.0], [96.0, 126.0, 156.0], [177.0, 234.0, 291.0]], [[150.0, 162.0, 174.0], [474.0, 513.0, 552.0], [798.0, 864.0, 930.0]],
             [[447.0, 468.0, 489.0], [1014.0, 1062.0, 1110.0], [1581.0, 1656.0, 1731.0]]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array(
            [[[3.0, 12.0, 21.0], [30.0, 39.0, 48.0], [57.0, 66.0, 75.0]], [[3.0, 12.0, 21.0], [30.0, 39.0, 48.0], [57.0, 66.0, 75.0]],
             [[3.0, 12.0, 21.0], [30.0, 39.0, 48.0], [57.0, 66.0, 75.0]]])
        expected_grad_y = np.array(
            [[[27.0, 27.0, 27.0], [30.0, 30.0, 30.0], [33.0, 33.0, 33.0]], [[36.0, 36.0, 36.0], [39.0, 39.0, 39.0], [42.0, 42.0, 42.0]],
             [[45.0, 45.0, 45.0], [48.0, 48.0, 48.0], [51.0, 51.0, 51.0]]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation1(self):
        x = Variable('x')
        y = Variable('y')
        formula = einsum('ij, kl -> k', x, y)

        x_val = np.array([[1.0, -2.0, 5.0], [5.0, 3.0, -4.0], [7.0, 3.0, 11.0]])
        y_val = np.array([[3.0, 5.0, -11.0], [2.0, -7.0, 4.0], [7.0, 2.0, -10.0]])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([-87.0, -29.0, -29.0])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]])
        expected_grad_y = np.array([[29.0, 29.0, 29.0], [29.0, 29.0, 29.0], [29.0, 29.0, 29.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation2(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        formula = einsum('ij, kl, k -> k', x, y, z)

        x_val = np.array([[1.0, -2.0, 5.0], [5.0, 3.0, -4.0], [7.0, 3.0, 11.0]])
        y_val = np.array([[3.0, 5.0, -11.0], [2.0, -7.0, 4.0], [7.0, 2.0, -10.0]])
        z_val = np.array([3.0, 5.0, -10.0])

        result = formula.evaluate_at(x=x_val, y=y_val, z=z_val)
        expected_result = np.array([-261.0, -145.0, 290.0])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[-4.0, -4.0, -4.0], [-4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]])
        expected_grad_y = np.array([[87.0, 87.0, 87.0], [145.0, 145.0, 145.0], [-290.0, -290.0, -290.0]])
        expected_grad_z = np.array([-87.0, -29.0, -29.0])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)
        np.testing.assert_array_almost_equal(grads[z], expected_grad_z)

    def test_einsum_operation3(self):
        x = Variable('x')
        y = Variable('y')
        formula = einsum('ij, jk -> ik', x, y)

        x_val = np.array([[1.0, -2.0, 5.0], [5.0, 3.0, -4.0], [7.0, 3.0, 11.0]])
        y_val = np.array([[3.0, 2.0, 7.0], [5.0, -7.0, 2.0], [-11.0, 4.0, -10.0]])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([[-62.0, 36.0, -47.0], [74.0, -27.0, 81.0], [-85.0, 37.0, -55.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[12., 0., -17.], [12., 0., -17.], [12., 0., -17.]])
        expected_grad_y = np.array([[13., 13., 13.], [4., 4., 4.], [12., 12., 12.]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation4(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        formula = einsum('ij, kl, mn -> mnij', x, y, z)

        x_val = np.array([[1.0, -2.0], [5.0, 3.0]])
        y_val = np.array([[3.0, 2.0], [-7.0, 4.0]])
        z_val = np.array([[1.0, 0.0], [0.0, -1.0]])

        result = formula.evaluate_at(x=x_val, y=y_val, z=z_val)
        expected_result = np.array([[[[2.0, -4.0],
                                      [10.0, 6.0]],
                                     [[0.0, 0.0],
                                      [0.0, 0.0]]],
                                    [[[0.0, 0.0],
                                      [0.0, 0.0]],
                                     [[-2.0, 4.0],
                                      [-10.0, -6.0]]]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[0.0, 0.0], [0.0, 0.0]])
        expected_grad_y = np.array([[0.0, 0.0], [0.0, 0.0]])
        expected_grad_z = np.array([[14.0, 14.0], [14.0, 14.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)
        np.testing.assert_array_almost_equal(grads[z], expected_grad_z)

    def test_einsum_operation5(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        formula = einsum('iji,ki,kj->j', x, y, z)

        x_val = np.ones((3, 4, 3))
        y_val = np.ones((2, 3))
        z_val = np.ones((2, 4))

        result = formula.evaluate_at(x=x_val, y=y_val, z=z_val)
        expected_result = np.array([6.0, 6.0, 6.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                                    [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
                                    [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]])
        expected_grad_y = np.array([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]])
        expected_grad_z = np.array([[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)
        np.testing.assert_array_almost_equal(grads[z], expected_grad_z)

    def test_einsum_operation6(self):
        x = Variable('x')
        y = Variable('y')
        formula = einsum('i, i ->', x, y)

        x_val = np.array([1.0, 2.0, 3.0])
        y_val = np.array([4.0, 5.0, 6.0])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = 32.0
        np.testing.assert_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([4.0, 5.0, 6.0])
        expected_grad_y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation7(self):
        x = Variable('x')
        y = Variable('y')
        formula = einsum('i, i -> i', x, y)

        x_val = np.array([1.0, 2.0, 3.0])
        y_val = np.array([4.0, 5.0, 6.0])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([4.0, 10.0, 18.0])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([4.0, 5.0, 6.0])
        expected_grad_y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation8(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        formula = einsum('ij, jk, kl -> il', x, y, z)

        x_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_val = np.array([[2.0, 1.0], [0.0, 2.0]])
        z_val = np.array([[1.0, 3.0], [2.0, 2.0]])

        result = formula.evaluate_at(x=x_val, y=y_val, z=z_val)
        expected_result = np.array([[12.0, 16.0], [28.0, 40.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[12.0, 8.0], [12.0, 8.0]])
        expected_grad_y = np.array([[16.0, 16.0], [24.0, 24.0]])
        expected_grad_z = np.array([[8.0, 8.0], [16.0, 16.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)
        np.testing.assert_array_almost_equal(grads[z], expected_grad_z)

    def test_einsum_operation9(self):
        x = Variable('x')
        y = Variable('y')
        formula = einsum('ij, jk -> ikj', x, y)

        x_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_val = np.array([[2.0, 1.0], [0.0, 2.0]])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([[[2.0, 0.0], [1.0, 4.0]], [[6.0, 0.0], [3.0, 8.0]]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[3.0, 2.0], [3.0, 2.0]])
        expected_grad_y = np.array([[4.0, 4.0], [6.0, 6.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation10(self):
        x = Variable('x')
        y = Variable('y')
        formula = einsum('i, j -> ij', x, y)

        x_val = np.array([1.0, 2.0])
        y_val = np.array([3.0, 4.0])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([[3.0, 4.0], [6.0, 8.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([7.0, 7.0])
        expected_grad_y = np.array([3.0, 3.0])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation11(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        formula = (x ** 2 + y ** 3) * z - einsum('ij, jk -> ik', x, y)

        x_val = np.array([[5.0, 2.0], [3.0, 4.0]])
        y_val = np.array([[2.0, 6.0], [9.0, 5.0]])
        z_val = np.array([[2.0, 7.0], [5.0, 2.0]])

        result = formula.evaluate_at(x=x_val, y=y_val, z=z_val)
        expected_result = np.array([[38.0, 1500.0], [3648.0, 244.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[12.0, 14.0], [22.0, 2.0]])
        expected_grad_y = np.array([[16.0, 748.0], [1209.0, 144.0]])
        expected_grad_z = np.array([[33.0, 220.0], [738.0, 141.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)
        np.testing.assert_array_almost_equal(grads[z], expected_grad_z)

    def test_einsum_operation12(self):
        x = Variable('x')
        y = Variable('y')
        formula = einsum('ij, i -> j', x, y)

        x_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_val = np.array([2.0, 3.0])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([11.0, 16.0])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[2.0, 2.0], [3.0, 3.0]])
        expected_grad_y = np.array([3.0, 7.0])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation13(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        formula = einsum('i, ij, jk ->', x, y, z)

        x_val = np.array([1.0, 2.0])
        y_val = np.array([[2.0, 1.0], [0.0, 2.0]])
        z_val = np.array([[1.0, 3.0], [2.0, 2.0]])

        result = formula.evaluate_at(x=x_val, y=y_val, z=z_val)
        expected_result = 28.0
        np.testing.assert_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([12.0, 8.0])
        expected_grad_y = np.array([[4.0, 4.0], [8.0, 8.0]])
        expected_grad_z = np.array([[2.0, 2.0], [5.0, 5.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)
        np.testing.assert_array_almost_equal(grads[z], expected_grad_z)

    def test_einsum_operation14(self):
        x = Variable('x')
        formula = einsum('ii ->', x)

        x_val = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = formula.evaluate_at(x=x_val)
        expected_result = 5.0
        np.testing.assert_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)

    def test_einsum_operation15(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        w = Variable('w')
        formula = einsum('ij, jk, kl, lm -> im', x, y, z, w)

        x_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_val = np.array([[2.0, 1.0], [0.0, 2.0]])
        z_val = np.array([[1.0, 3.0], [2.0, 2.0]])
        w_val = np.array([[2.0, 1.0], [0.0, 2.0]])

        result = formula.evaluate_at(x=x_val, y=y_val, z=z_val, w=w_val)
        expected_result = np.array([[24.0, 44.0], [56.0, 108.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[28.0, 20.0], [28.0, 20.0]])
        expected_grad_y = np.array([[36.0, 40.0], [54.0, 60.0]])
        expected_grad_z = np.array([[24.0, 16.0], [48.0, 32.0]])
        expected_grad_w = np.array([[40.0, 40.0], [56.0, 56.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)
        np.testing.assert_array_almost_equal(grads[z], expected_grad_z)
        np.testing.assert_array_almost_equal(grads[w], expected_grad_w)

    def test_einsum_operation_with_ellipsis1(self):
        x = Variable('x')
        y = Variable('y')
        formula = einsum('...j, kl -> ...k', x, y)

        x_val = np.array([[[1.0, -2.0, 5.0], [5.0, 3.0, -4.0]], [[2.0, 1.0, -3.0], [4.0, 6.0, 2.0]]])
        y_val = np.array([[3.0, 5.0, -11.0], [2.0, -7.0, 4.0]])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([[[-12.0, -4.0], [-12.0, -4.0]], [[0.0, 0.0], [-36.0, -12.0]]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[[-4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]], [[-4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]]])
        expected_grad_y = np.array([[20.0, 20.0, 20.0], [20.0, 20.0, 20.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation_with_ellipsis2(self):
        x = Variable('x')
        y = Variable('y')
        formula = einsum('i, ...j -> ...ij', x, y)

        x_val = np.array([1.0, 2.0])
        y_val = np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

        result = formula.evaluate_at(x=x_val, y=y_val)
        expected_result = np.array([[[3.0, 4.0], [6.0, 8.0]], [[5.0, 6.0], [10.0, 12.0]], [[7.0, 8.0], [14.0, 16.0]]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([33.0, 33.0])
        expected_grad_y = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)

    def test_einsum_operation_with_ellipsis3(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        formula = einsum('...i, ...j, ...k -> ...ijk', x, y, z)

        x_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        y_val = np.array([[[2.0, 1.0], [0.0, 2.0]], [[1.0, 2.0], [2.0, 1.0]]])
        z_val = np.array([[[1.0, 3.0], [2.0, 2.0]], [[0.0, 1.0], [1.0, 0.0]]])

        result = formula.evaluate_at(x=x_val, y=y_val, z=z_val)
        expected_result = np.array([[[[[2.0, 6.0], [1.0, 3.0]], [[4.0, 12.0], [2.0, 6.0]]], [[[0.0, 0.0], [12.0, 12.0]], [[0.0, 0.0], [16.0, 16.0]]]],
                                    [[[[0.0, 5.0], [0.0, 10.0]], [[0.0, 6.0], [0.0, 12.0]]], [[[14.0, 0.0], [7.0, 0.0]], [[16.0, 0.0], [8.0, 0.0]]]]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[[12.0, 12.0], [8.0, 8.0]], [[3.0, 3.0], [3.0, 3.0]]])
        expected_grad_y = np.array([[[12.0, 12.0], [28.0, 28.0]], [[11.0, 11.0], [15.0, 15.0]]])
        expected_grad_z = np.array([[[9.0, 9.0], [14.0, 14.0]], [[33.0, 33.0], [45.0, 45.0]]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)
        np.testing.assert_array_almost_equal(grads[z], expected_grad_z)

    def test_einsum_operation_with_ellipsis4(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        w = Variable('w')
        formula = einsum('...i, ...j, ...k, l... -> ijkl', x, y, z, w)

        x_val = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        y_val = np.array([[2.0, 1.0], [0.0, 2.0]])
        z_val = np.array([[[1.0, 3.0], [2.0, 2.0]]])
        w_val = np.array([[2.0]])

        result = formula.evaluate_at(x=x_val, y=y_val, z=z_val, w=w_val)
        expected_result = np.array([[[[24.0], [72.0]], [[92.0], [116.0]]], [[[32.0], [96.0]], [[112.0], [144.0]]]])
        np.testing.assert_array_almost_equal(result, expected_result)

        grads = formula.grads
        expected_grad_x = np.array([[[[24.0, 24.0], [16.0, 16.0]], [[24.0, 24.0], [16.0, 16.0]]]])
        expected_grad_y = np.array([[112.0, 112.0], [176.0, 176.0]])
        expected_grad_z = np.array([[[84.0, 84.0], [88.0, 88.0]]])
        expected_grad_w = np.array([[344.0]])
        np.testing.assert_array_almost_equal(grads[x], expected_grad_x)
        np.testing.assert_array_almost_equal(grads[y], expected_grad_y)
        np.testing.assert_array_almost_equal(grads[z], expected_grad_z)
        np.testing.assert_array_almost_equal(grads[w], expected_grad_w)


if __name__ == '__main__':
    unittest.main()
