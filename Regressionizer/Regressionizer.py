import math
import pickle
import warnings
from typing import Union, Optional
from QuantileRegression import QuantileRegression, quantile_regression_fit, quantile_regression
import pandas
import numpy
from datetime import datetime, timedelta
import plotly.graph_objects as go


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
    def take_data(self):
        """Take the data."""
        return self.data

    def take_basis_funcs(self):
        """Take the basis functions."""
        return self.basis_funcs

    def take_probs(self):
        """Take the probabilities."""
        return self.probs

    def take_regression_quantiles(self):
        """Take the regression quantiles."""
        return self.regression_quantiles

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
                self.data = arg[['Time', 'Value']].to_numpy()
        elif isinstance(arg, numpy.ndarray) and arg.shape[1] >= 2:
            self.data = arg
        elif isinstance(arg, numpy.ndarray) and arg.shape[1] == 1:
            self.data = numpy.column_stack((numpy.arange(len(arg)), arg))
        else:
            ValueError("The first argument is expected to be a list of numbers, a numpy array, or a data frame.")

    def set_basis_funcs(self, arg):
        """Set document-term matrix."""
        if _is_list_of_callables(arg):
            self.basis_funcs = arg
        else:
            raise TypeError("The first argument is expected to be a list of functions (callables.)")
        return self

    def set_probs(self, arg):
        """Set probabilities."""
        if _is_list_of_probs(arg):
            self.probs = arg
        else:
            TypeError("The first argument is expected to be a list of probabilities.")
        return self

    def set_value(self, arg):
        """Set pipeline value."""
        self._value = arg
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
        Regressionizer: The instance of the Regressionizer class with fitted quantiles.
        """
        super(Regressionizer, self).quantile_regression_fit(funcs, probs, **opts)
        self.regression_quantiles = dict(zip(self.probs, self.regression_quantiles))
        self._value = self.regression_quantiles
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
        Regressionizer: The instance of the Regressionizer class with fitted quantiles.
        """
        super(Regressionizer, self).quantile_regression(knots, probs, order, **opts)
        self.regression_quantiles = dict(zip(self.probs, self.regression_quantiles))
        self._value = self.regression_quantiles
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
        epoch_start_date = datetime.strptime(epoch_start, "%Y-%m-%d")
        print(epoch_start_date)

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

                if date_list_plot:
                    if numpy.issubdtype(x.dtype, numpy.number):
                        x = [epoch_start_date + timedelta(days=int(num)) for num in x]

                mode2 = "lines"
                if label in mode_dict:
                    mode2 = mode_dict[label]

                fig.add_trace(go.Scatter(x=x, y=y, mode=mode2, name=label))

        fig.update_layout(title=title, width=width, height=height, **kwargs)

        self._value = fig

        return self

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot(self,
             title="", width=800, height=600,
             date_list_plot=False, epoch_start="1900-01-01", **kwargs):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.data[:, 0], y=self.data[:, 1], mode="markers", name="data"))

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            **kwargs
        )

        # Plot each regression quantile
        for i, p in enumerate(self.regression_quantiles.keys()):
            y_fit = [self.regression_quantiles[p](xi) for xi in self.data[:, 0]]
            fig.add_trace(go.Scatter(x=self.data[:, 0], y=y_fit, mode='lines', name=f'{p}'))

        self._value=fig

        return self

    # ------------------------------------------------------------------
    # DateListPlot
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # PlotOutliers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __str__(self):
        if isinstance(self.data, numpy.ndarray):
            res = "Regressionizer object with data that has %d records" % self.take_data().shape()
        else:
            res = "Regressionizer object with no data"

        if isinstance(self.regression_quantiles, list) and self.regression_quantiles > 0:
            res = res + f" and {self.regression_quantiles} regression quantiles"
        elif isinstance(self.regression_quantiles, numpy.ndarray) and self.regression_quantiles.shape[0] > 0:
            res = res + f" and {self.regression_quantiles} regression quantiles"
        else:
            res = res + " and no regression quantiles"

        return res + "."

    def __repr__(self):
        """Representation of Regressionizer object."""
        return str(self)
