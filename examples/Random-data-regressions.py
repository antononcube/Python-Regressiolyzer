
from Regressionizer.Regressionizer import Regressionizer

import numpy
from scipy.stats import norm

numpy.random.seed(0)
x = numpy.linspace(0, 2, 240)
y = numpy.sin(2 * numpy.pi * x) + numpy.random.normal(0, 0.4, x.shape)
data = numpy.column_stack((x, y))

dist_data = numpy.array([[x, numpy.exp(-x ** 2) + norm.rvs(scale=0.15)] for x in numpy.arange(-3, 3.2, 0.2)])
funcs = [lambda x: 1, lambda x: x, lambda x: numpy.cos(x), lambda x: numpy.cos(3 * x), lambda x: numpy.cos(6 * x)]

obj = Regressionizer(data).quantile_regression(knots=5, probs=[0.2, 0.5, 0.8])

print(obj.take_value())

print(obj.outliers().take_value())

print(obj.outliers_plot(date_list_plot=True))

