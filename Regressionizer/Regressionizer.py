import math
import pickle
import warnings
from typing import Union, Optional
from QuantileRegression import QuantileRegression
import pandas
import numpy
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.subplots as sp
import scipy


# ======================================================================
# Utilities
# ======================================================================
def _is_list_of_callables(obj):
    if not isinstance(obj, (list, tuple)):
        return False
    return all(callable(item) for item in obj)


def _is_numeric_list(obj):
    return isinstance(obj, list | tuple) and all([isinstance(x, float | int) for x in obj])


def _is_list_of_probs(obj):
    if not isinstance(obj, (list, tuple)):
        return False
    return all(isinstance(item, (int, float)) and 0 <= item <= 1 for item in obj)


def _five_point_summary(arr):
    return {
        'min': numpy.min(arr),
        '25%': numpy.percentile(arr, 25),
        'median': numpy.median(arr),
        '75%': numpy.percentile(arr, 75),
        'max': numpy.max(arr)
    }


def _five_point_summary_columnwise(arr, column_names=('Regressor', 'Value')):
    if not (isinstance(column_names, list) and len(column_names) >= 2):
        ValueError("The value of column_names must be a list or tuple of length at least 2.")

    summary = {}
    for i in range(arr.shape[1]):
        summary[column_names[i]] = {
            'min': numpy.min(arr[:, i]),
            '25%': numpy.percentile(arr[:, i], 25),
            'median': numpy.median(arr[:, i]),
            '75%': numpy.percentile(arr[:, i], 75),
            'max': numpy.max(arr[:, i])
        }
    return summary


def _print_summary(summary):
    headers = ['Statistic'] + list(summary.keys())
    rows = ['min', '25%', 'median', '75%', 'max']
    print("{:<12} {}".format(headers[0], ' | '.join(headers[1:])))
    print("-" * 12 + " " + "-" * (len(headers) - 1) * 10)
    for row in rows:
        values = [summary[col][row] for col in summary]
        print("{:<12} {}".format(row, ' | '.join(f"{v:>10}" for v in values)))


# ======================================================================
# Class definition
# ======================================================================
class Regressionizer(QuantileRegression):
    _value = None

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """Creation of Regressionizer object.

        The first (optional) argument is expected to be a list of numbers, a numpy array, or a data frame.
        """
        super().__init__()
        if len(args) == 1:
            self.set_data(args[0])
        else:
            ValueError("One or no arguments are expected.")

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    def take_value(self):
        """Take the pipeline value."""
        return self._value

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------
    def set_data(self, arg):
        """Set data."""
        if isinstance(arg, pandas.DataFrame):
            columns = arg.columns.str.lower()
            has_columns = 'time' in columns and 'value' in columns
            if has_columns:
                self._data = arg[['Time', 'Value']].to_numpy()
        elif isinstance(arg, numpy.ndarray) and arg.shape[1] >= 2:
            self._data = arg
        elif isinstance(arg, numpy.ndarray) and arg.shape[1] == 1:
            self._data = numpy.column_stack((numpy.arange(len(arg)), arg))
        else:
            ValueError("The first argument is expected to be a list of numbers, a numpy array, or a data frame.")

    def set_basis_funcs(self, arg):
        """Set document-term matrix."""
        if _is_list_of_callables(arg):
            self._basis_funcs = arg
        else:
            raise TypeError("The first argument is expected to be a list of functions (callables.)")
        return self

    def set_regression_quantiles(self, arg):
        """Set regression quantiles."""
        if isinstance(arg, dict) and _is_list_of_probs(list(arg.keys())):
            self._regression_quantiles = arg
        else:
            TypeError("The first argument is expected to be a dictionary of probabilities to regression quantiles.")
        return self

    def set_value(self, arg):
        """Set pipeline value."""
        self._value = arg
        return self

    # ------------------------------------------------------------------
    # Echo summary
    # ------------------------------------------------------------------
    def echo_data_summary(self, echo: bool = True):
        """
        Echo data summary.

        Parameters:
            echo (bool): Whether to echo the data summary.
        Returns:
            Regressionizer: The instance of the Regressionizer class
        """
        self._value = _five_point_summary_columnwise(self.take_data())
        if echo:
            _print_summary(self._value)
        return self

    # ------------------------------------------------------------------
    # Rescale
    # ------------------------------------------------------------------
    def rescale(self, regressor=False, value=True):
        """
        Rescale the data.
        :param regressor: Whether to rescale the regressor column data.
        :param value: Whether to rescale the value column data.
        :return: The instance of the Regressionizer class.
        """
        if regressor:
            min_regressor = self.take_data()[:, 0].min()
            max_regressor = self.take_data()[:, 0].max()
            self._data[:, 0] = (self.take_data()[:, 0] - min_regressor) / (max_regressor - min_regressor)

        if value:
            min_value = self.take_data()[:, 1].min()
            max_value = self.take_data()[:, 1].max()
            self._data[:, 1] = (self.take_data()[:, 1] - min_value) / (max_value - min_value)

        return self

    # ------------------------------------------------------------------
    # Least Squares
    # ------------------------------------------------------------------
    def fit(self, funcs, **kwargs):
        """
        Least squares fit -- shortcut for least squares fit.
        :param funcs:
        :param kwargs:
        :return:
        """
        return self.least_squares_fit(funcs, **kwargs)

    def least_squares_fit(self, funcs, **kwargs):
        """
        Least squares fit.
        Non-linear.

        :param funcs: Functions to fit.
        :param kwargs: Additional keyword arguments for scipy.optimize.curve_fit.
        :return: The instance of the Regressionizer class.
        """

        def combined_function(x, *params):
            result = numpy.zeros_like(x)
            for func, param in zip(funcs, params):
                result += func(x) * param
            return result

        x_data = self.take_data()[:, 0]
        y_data = self.take_data()[:, 1]

        func = numpy.vectorize(combined_function)
        p0 = kwargs.pop('p0', numpy.ones(len(funcs)))
        params, pcov = scipy.optimize.curve_fit(func, x_data, y_data, p0=p0, **kwargs)

        def fitted_function(x):
            return combined_function(x, *params)

        # Result
        if isinstance(self.take_regression_quantiles(), dict):
            self._regression_quantiles = self.take_regression_quantiles() | {"mean": fitted_function}
        else:
            self._regression_quantiles = {"mean": fitted_function}

        return self

    # ------------------------------------------------------------------
    # Quantile regression
    # ------------------------------------------------------------------
    def quantile_regression_fit(self, funcs, probs=None, **opts):
        """
        Quantile regression fit using the specified functions and probabilities.

        Parameters:
        funcs (list): A list of functions to be used in the quantile regression fitting.
        probs (list, options): A list of probabilities at which to estimate the quantiles.
                               If None, defaults to a standard set of probabilities, 0.25, 0.5, and 0.75..
        **opts: Additional keyword arguments to be passed to the scipy.optimize.linprog() function.

        Returns:
        Regressionizer: The instance of the Regressionizer class with fitted regression quantiles.
        """
        super(Regressionizer, self).quantile_regression_fit(funcs, probs, **opts)
        self._regression_quantiles = dict(zip(self.take_probs(), self.take_regression_quantiles()))
        self._value = self.take_regression_quantiles()
        return self

    # ------------------------------------------------------------------
    # Quantile regression
    # ------------------------------------------------------------------
    def quantile_regression(self, knots, probs=None, order: int = 3, **opts):
        """
        Quantile regression using specified knots and probabilities.

        This method calls the quantile regression implementation from the superclass
        and stores the resulting regression quantiles in the instance variable `_value`.

        Parameters:
        knots (int or list): Either the number of regularly spaced knots or a list of
                             actual B-spline knots for the quantile regression.
        probs (None, float, or list, optional): Can be None, a single number between 0 and 1,
                                                 or a list of numbers between 0 and 1
                                                 at which to estimate the regression quantiles.
        order (int, optional): The order of the B-splines to be used in the regression.
                               Defaults to 3.
        **opts: Additional keyword arguments to be passed to the scipy.optimize.linprog() function.

        Returns:
        Regressionizer: The instance of the Regressionizer class with fitted regression quantiles.
        """
        super(Regressionizer, self).quantile_regression(knots, probs, order, **opts)
        self._regression_quantiles = dict(zip(self.take_probs(), self.take_regression_quantiles()))
        self._value = self.take_regression_quantiles()
        return self

    # ------------------------------------------------------------------
    # Generic plot (private)
    # ------------------------------------------------------------------
    def _list_plot(self,
                   data: dict,
                   title="", width=800, height=600,
                   mode: (str | dict) = "lines",
                   date_list_plot=False, epoch_start="1900-01-01",
                   **kwargs):
        fig = go.Figure()
        start_date = pandas.Timestamp(epoch_start)

        if not isinstance(data, dict):
            raise ValueError("The data argument must be a dictionary of DataFrames or numpy arrays.")

        mode_dict = mode
        if isinstance(mode_dict, str):
            mode_dict = {k: mode_dict for k, v in data.items()}

        if not isinstance(mode_dict, dict):
            raise ValueError(
                """The value of the argument "mode" must be a strings a dictionary of strings to strings.""")

        if isinstance(data, dict):
            for label, series in data.items():
                if isinstance(series, pandas.DataFrame):
                    x = series.iloc[:, 0]
                    y = series.iloc[:, 1]
                elif isinstance(series, numpy.ndarray):
                    x = series[:, 0]
                    y = series[:, 1]
                else:
                    raise ValueError("Unsupported data type in dictionary of time series.")

                if date_list_plot and numpy.issubdtype(x.dtype, numpy.number):
                    x = start_date + pandas.to_timedelta(x, unit='s')

                mode2 = "lines"
                if label in mode_dict:
                    mode2 = mode_dict[label]

                fig.add_trace(go.Scatter(x=x, y=y, mode=mode2, name=label))

        fig.update_layout(title=title, width=width, height=height, **kwargs)

        self._value = fig

        return self

    # ------------------------------------------------------------------
    # Outliers
    # ------------------------------------------------------------------
    def outliers(self):
        """
        Outliers found using regression quantiles.

        At least two regression quantiles are needed to estimate the outliers. (E.g 0.02 and 0.98.)

        :return: The instance of the Regressionizer class with found outliers.
        """
        if not (isinstance(self.take_regression_quantiles(), dict) and len(self.take_regression_quantiles()) > 1):
            ValueError("Quantile regression with at least two regression quantiles is expected.")

        topProb = sorted(self.take_regression_quantiles().keys())[-1]
        bottomProb = sorted(self.take_regression_quantiles().keys())[0]

        qr_bottom_outliers = [r for r in self.take_data() if self.take_regression_quantiles()[bottomProb](r[0]) > r[1]]
        qr_top_outliers = [r for r in self.take_data() if self.take_regression_quantiles()[topProb](r[0]) < r[1]]

        self._value = {"bottom": qr_bottom_outliers, "top": qr_top_outliers}

        return self

    # ------------------------------------------------------------------
    # CDF
    # ------------------------------------------------------------------
    def conditional_cdf(self, points):
        """
        Conditional Cumulative Distribution Functions (CDFs).
        :param points: Points to find CDFs at.
        :return: An instance of the Regressionizer class with CDFs.
        """
        if isinstance(points, float | int):
            return self.conditional_cdf([points, ])
        elif not _is_numeric_list(points):
            TypeError("A number or a list of numbers is expected as first argument.")

        if not (isinstance(self.take_regression_quantiles(), dict) and len(self.take_regression_quantiles()) > 1):
            ValueError("At least two regression quantiles is expected.")

        res = {}
        for p in points:
            rq_values = numpy.array(
                [(self.take_regression_quantiles()[prob](p), prob) for prob in sorted(self.take_regression_quantiles())])
            res = res | {p: scipy.interpolate.interp1d(x=rq_values[:, 0], y=rq_values[:, 1])}

        self._value = res
        return self

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot(self,
             title="", width=800, height=600,
             date_list_plot: bool = False, epoch_start="1900-01-01",
             **kwargs):
        """
        Plot data and regression quantiles.
        :param title: Title of the plot.
        :param width: Width of the plot.
        :param height: Height of the plot.
        :param date_list_plot: Whether to plot as a date-time series.
        :param epoch_start: Start of epoch when regressor is in seconds.
        :param kwargs: Additional keyword arguments to be passed to the plotly's update_layout.
        :return: The instance of the Regressionizer class.
        """
        start_date = pandas.Timestamp(epoch_start)
        fig = go.Figure()

        xs = self.take_data()[:, 0]
        if date_list_plot:
            xs = start_date + pandas.to_timedelta(xs, unit='s')

        # Plot data points
        fig.add_trace(go.Scatter(x=xs, y=self.take_data()[:, 1], mode="markers", name="data"))

        # Plot each regression quantile
        for i, p in enumerate(self.take_regression_quantiles().keys()):
            y_fit = [self.take_regression_quantiles()[p](xi) for xi in self.take_data()[:, 0]]
            fig.add_trace(go.Scatter(x=xs, y=y_fit, mode='lines', name=f'{p}'))

        # Layout options
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            **kwargs
        )

        # Result
        self._value = fig

        return self

    # ------------------------------------------------------------------
    # DateListPlot
    # ------------------------------------------------------------------
    def date_list_plot(self,
                       title="", width=800, height=600,
                       epoch_start="1900-01-01",
                       **kwargs):
        """
        Plot data and regression quantiles with time/date axis.
        :param title: Title of the plot.
        :param width: Width of the plot.
        :param height: Height of the plot.
        :param epoch_start: Start of epoch when regressor is in seconds.
        :param kwargs: Additional keyword arguments to be passed to the plotly's update_layout.
        :return: The instance of the Regressionizer class.
        """
        return self.plot(title=title, width=width, height=height,
                         date_list_plot=True, epoch_start=epoch_start,
                         **kwargs)

    # ------------------------------------------------------------------
    # PlotOutliers
    # ------------------------------------------------------------------
    def plot_outliers(self,
                      title="", width=800, height=600,
                      date_list_plot: bool = False, epoch_start="1900-01-01",
                      **kwargs):
        """
        Plot data and regression quantiles, and outliers.
        :param title: Title of the plot.
        :param width: Width of the plot.
        :param height: Height of the plot.
        :param date_list_plot: Whether to plot as a date-time series.
        :param epoch_start: Start of epoch when regressor is in seconds.
        :param kwargs: Additional keyword arguments to be passed to the plotly's update_layout.'
        :return: The instance of the Regressionizer class.
        """
        # Some code refactoring is possible:
        # self.outliers()

        topProb = sorted(self.take_regression_quantiles().keys())[-1]
        bottomProb = sorted(self.take_regression_quantiles().keys())[0]

        qr_bottom_outliers = [r for r in self.take_data() if self.take_regression_quantiles()[bottomProb](r[0]) > r[1]]
        qr_top_outliers = [r for r in self.take_data() if self.take_regression_quantiles()[topProb](r[0]) < r[1]]

        # Get the corresponding bottom and top regression quantiles
        bottom_regression_quantile = [(x, self.take_regression_quantiles()[bottomProb](x)) for x in self.take_data()[:, 0]]
        top_regression_quantile = [(x, self.take_regression_quantiles()[topProb](x)) for x in self.take_data()[:, 0]]

        self._list_plot({"data": self.take_data(),
                         "bottom outliers": numpy.array(qr_bottom_outliers),
                         "top outliers": numpy.array(qr_top_outliers),
                         bottomProb: numpy.array(bottom_regression_quantile),
                         topProb: numpy.array(top_regression_quantile)
                         },
                        mode={"data": "markers", "bottom outliers": "markers", "top outliers": "markers"},
                        title=title, width=width, height=height,
                        date_list_plot=date_list_plot, epoch_start=epoch_start,
                        **kwargs)
        return self

    # ------------------------------------------------------------------
    # Multi-panel plot
    # ------------------------------------------------------------------
    def _create_multi_panel_plot_with_segments(self,
                                               function_dict,
                                               title="Error plots", width=800, height=300,
                                               mode = 'markers',
                                               filling : bool = True,
                                               **kwargs):
        num_functions = len(function_dict)
        fig = sp.make_subplots(rows=num_functions, cols=1, subplot_titles=list(function_dict.keys()))

        for i, (name, points) in enumerate(function_dict.items(), start=1):
            x, y = zip(*points)
            scatter_trace = go.Scatter(x=x, y=y, mode=mode, name=name)
            fig.add_trace(scatter_trace, row=i, col=1)

            if filling:
                for xi, yi in zip(x, y):
                    line_trace_x = go.Scatter(x=[xi, xi], y=[0, yi], mode='lines', showlegend=False,
                                              line=dict(color='gray', dash='solid'))
                    line_trace_y = go.Scatter(x=[xi, xi], y=[yi, 0], mode='lines', showlegend=False,
                                              line=dict(color='gray', dash='dash'))
                    fig.add_trace(line_trace_x, row=i, col=1)
                    fig.add_trace(line_trace_y, row=i, col=1)

        fig.update_layout(title=title, width=width, height=height * num_functions, **kwargs)

        self._value = fig
        return self

    # ------------------------------------------------------------------
    # Error plots
    # ------------------------------------------------------------------
    def error_plots(self,
                    title="", width=800, height=300,
                    date_list_plot: bool = False, epoch_start="1900-01-01",
                    **kwargs):
        """
        Plot residual fitting errors for found regression quantiles.
        :param title: Title of the plot.
        :param width: Width of the plot.
        :param height: Height of the plot.
        :param date_list_plot: Whether to plot as a date-time series.
        :param epoch_start: Start of epoch when regressor is in seconds.
        :param kwargs: Additional keyword arguments to be passed to the plotly's update_layout.'
        :return: The instance of the Regressionizer class.
        """
        start_date = pandas.Timestamp(epoch_start)
        x = self.take_data()[:, 0]
        xs = self.take_data()[:, 0]
        ys = self.take_data()[:, 1]
        if date_list_plot:
            xs = start_date + pandas.to_timedelta(xs, unit='s')

        function_dict = {k: [(xs, y - f(x)) for x, xs, y in zip(x, xs, ys)] for k, f in
                         self.take_regression_quantiles().items()}
        return self._create_multi_panel_plot_with_segments(function_dict,
                                                           title=title,
                                                           width=width, height=height,
                                                           mode = 'markers',
                                                           filling = True,
                                                           **kwargs)

    # ------------------------------------------------------------------
    # Conditional CDF plots
    # ------------------------------------------------------------------
    def conditional_cdf_plot(self,
                             points,
                             title="", width=800, height=300,
                             **kwargs):
        """
        Plot conditional CDFs based on found regression quantiles.
        :param points: List of point to at which the CDFs are computed.
        :param title: Title of the plot.
        :param width: Width of the plot.
        :param height: Height of the plot.
        :param kwargs: Additional keyword arguments to be passed to the plotly's update_layout.'
        :return: The instance of the Regressionizer class.
        """
        if isinstance(points, float | int):
            return self.conditional_cdf_plot([points, ], title=title, width=width, height=height, **kwargs)
        elif not _is_numeric_list(points):
            TypeError("A number or a list of numbers is expected as first argument.")

        if not (isinstance(self.take_regression_quantiles(), dict) and len(self.take_regression_quantiles()) > 1):
            ValueError("At least two regression quantiles is expected.")

        res = {}
        for p in points:
            rq_values = numpy.array(
                [(self.take_regression_quantiles()[prob](p), prob) for prob in sorted(self.take_regression_quantiles())])
            res = res | {p: rq_values}

        return self._create_multi_panel_plot_with_segments(res,
                                                           title=title,
                                                           width=width, height=height,
                                                           mode='lines',
                                                           filling=False,
                                                           **kwargs)

    # ------------------------------------------------------------------
    # To dictionary form
    # ------------------------------------------------------------------
    def to_dict(self):
        """Convert to dictionary form.

        Returns dictionary representation of the Regessionizer object with keys:
        ['data', 'basis_funcs', 'regression_quantiles', 'value'].

        (Ideally) this function facilitates rapid conversion and serialization.
        """

        res = {"data": self.take_data(),
               "basis_funcs": self.take_basis_funcs(),
               "regression_quantiles": self.take_regression_quantiles(),
               "value": self.take_value()}
        return res

    # ------------------------------------------------------------------
    # From dictionary form
    # ------------------------------------------------------------------
    def from_dict(self, arg):
        """Convert from dictionary form.

        Creates a Regressionizer object from a dictionary representation with keys:
        ['data', 'basis_funcs', 'regression_quantiles', 'value'].

        (Ideally) this function facilitates rapid conversion and serialization.
        """
        if not (isinstance(arg, dict) and
                all([x in arg for x in ['data', 'basis_funcs', 'regression_quantiles']])):
            raise TypeError("""The first argument is expected to be a dictionary with keys:
            'data', 'basis_funcs', 'regression_quantiles'.""")

        self.set_data(arg["data"])
        self.set_regression_quantiles(arg["regression_quantiles"])
        self.set_basis_funcs(arg["basis_funcs"])
        self.set_value(arg.get("value", None))
        return self

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __str__(self):
        if isinstance(self.take_data(), numpy.ndarray):
            res = "Regressionizer object with data that has %d records" % self.take_data().shape[0]
        else:
            res = "Regressionizer object with no data"

        if isinstance(self.take_regression_quantiles(), dict) and len(self.take_regression_quantiles()) > 0:
            res = res + f" and {len(self.take_regression_quantiles())} regression quantiles for {str(list(self.take_regression_quantiles().keys()))}"
        else:
            res = res + " and no regression quantiles"

        return res + "."

    def __repr__(self):
        """Representation of Regressionizer object."""
        return str(self)
