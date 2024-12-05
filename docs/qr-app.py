from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from Regressionizer import *
from OutlierIdentifiers import *

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sns.set_theme()

# Load the gzipped CSV file
fileName = "../Regressionizer/resources/dfTemperatureData.csv.zip"
dfTemperatureData = pd.read_csv(fileName, compression='zip')

# Convert 'Time' column from seconds since 1900-01-01 to date objects
#dfTemperatureData['Time'] = dfTemperatureData['Time'].apply(lambda x: datetime(1900, 1, 1) + timedelta(seconds=x))

# Display the DataFrame
print(dfTemperatureData)

dfData = dfTemperatureData.to_numpy()

# Generic seaborn function over Regressionizer object
def sns_plot(obj,
    title="", width=800, height=600,
    data_color: (str | None) = "grey",
    date_plot: bool = False, epoch_start="1900-01-01",
    background_color="Gainsboro",
    grid_lines=False,
    point_size=2,
    **kwargs):
    """
    Plot data and regression quantiles using seaborn.
    :param title: Title of the plot.
    :param width: Width of the plot.
    :param height: Height of the plot.
    :param data_color: Color of the data points.
    :param date_plot: Whether to plot as a date-time series.
    :param epoch_start: Start of epoch when regressor is in seconds.
    :param background_color: Background color of the plot.
    :param grid_lines: Whether to show grid lines.
    :param kwargs: Additional keyword arguments to be passed to seaborn's plotting functions.
    :return: The instance of the Regressionizer class.
    """
    start_date = pd.Timestamp(epoch_start)
    xs = obj.take_data()[:, 0]
    if date_plot:
        xs = start_date + pd.to_timedelta(xs, unit='s')

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    sns.set_theme()

    # Set background color
    ax.set_facecolor(background_color)

    # Plot data points
    sns.scatterplot(x=xs, y=obj.take_data()[:, 1], color=data_color, ax=ax, size=point_size)

    # Plot each regression quantile
    for i, p in enumerate(obj.take_regression_quantiles().keys()):
        y_fit = [obj.take_regression_quantiles()[p](xi) for xi in obj.take_data()[:, 0]]
        sns.lineplot(x=xs, y=y_fit, ax=ax, label=f'{p}', linewidth=3)

    # Set title
    ax.set_title(title)

    # Set the style to include grid lines
    if grid_lines:
        sns.set_style("whitegrid") 


    # Do not show the plot
    #plt.close(fig)

    # Result
    obj._value = fig

    return obj

# Style variables/constants
template='plotly_dark'
data_color='gray'

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# App definitions
# app_ui = ui.page_sidebar(
#     ui.sidebar(
#         ui.input_slider(
#             "knots", "Number of knots", min=2, max=100, value=10
#         ),
#         ui.input_slider(
#             "prob", "Probability", min=0, max=1, value=0.5
#         ),
#     ),
#     ui.card(
#         ui.output_plot("distPlot"),
#     ),
# )

app_ui = ui.page_fluid(
    ui.input_slider("knots", "Number of knots:", min=2, max=100, value=10),
    #ui.input_slider("prob", "Probability:", min=0, max=1, value=0.5),
    ui.input_text("probs", "Probabilities:", "0.1, 0.5, 0.9"),
    ui.input_checkbox("gridQ", "Show grid", value=False),
    ui.card(
        ui.output_plot("distPlot"),
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def probs_arr():
        return [float(num.strip()) for num in input.probs().split(",") if is_float(num.strip())]

    @render.plot
    def distPlot():
        obj = (
            Regressionizer(dfData)
            .quantile_regression(knots=input.knots(), probs=probs_arr(), order=3)
        )

        obj = sns_plot(obj,
            title="Outliers of Orlando, FL, USA, Temperature, C",
            date_plot=True, 
            template=template,
            data_color=data_color,
            #background_color = '#1F1F1F',
            grid_lines = input.gridQ(),
            point_size = 2,
            width = 800, height = 300)

        return obj.take_value()
  


app = App(app_ui, server)