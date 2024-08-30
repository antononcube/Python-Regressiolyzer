import unittest
import numpy
from Regressionizer import *


def chebyshev_t_polynomials(n):
    if n == 0:
        return lambda x: 1
    elif n == 1:
        return lambda x: x
    else:
        T0 = lambda x: 1
        T1 = lambda x: x
        for i in range(2, n + 1):
            Tn = lambda x, T0=T0, T1=T1: 2 * x * T1(x) - T0(x)
            T0, T1 = T1, Tn
        return Tn


chebyshev_polynomials = [chebyshev_t_polynomials(i) for i in range(8)]


class MyTestCase(unittest.TestCase):
    x = numpy.linspace(0, 2, 400)
    y = numpy.sin(2 * numpy.pi * x) + numpy.random.normal(0, 0.4, x.shape)
    data = numpy.column_stack((x, y))
    funcs = [lambda x: 1, lambda x: x, lambda x: numpy.cos(x), lambda x: numpy.cos(3 * x), lambda x: numpy.cos(6 * x)]

    def test_quantile_regression(self):
        probs = [0.2, 0.5, 0.8]

        frac_dict = (
            Regressionizer(self.data)
            .quantile_regression(knots=10, probs=probs)
            .separate(cumulative=True, fractions=True)
            .take_value()
        )

        for prob in probs:
            self.assertAlmostEqual(frac_dict[prob], prob, delta=0.025)

    def test_quantile_regression_fit(self):
        probs = [0.2, 0.5, 0.8]

        frac_dict = (
            Regressionizer(self.data)
            .quantile_regression_fit(funcs=chebyshev_polynomials, probs=probs)
            .separate(cumulative=True, fractions=True)
            .take_value()
        )

        for prob in probs:
            self.assertAlmostEqual(frac_dict[prob], prob, delta=0.025)


if __name__ == '__main__':
    unittest.main()
