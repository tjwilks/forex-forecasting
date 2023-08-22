import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta


class ForecastDataset:
    def __init__(self, y_train: List[float], y_test: List[float],
                 dates_train: List[str], dates_test: List[str]):
        """
        Initialize a ForecastDataset object.

        Parameters:
            y_train (List[float]): The target values for the training set.
            y_test (List[float]): The target values for the test set.
            dates_train (List[str]): The corresponding dates for the training
            set.
            dates_test (List[str]): The corresponding dates for the test set.
        """
        self.y_train = y_train
        self.y_test = y_test
        self.dates_train = dates_train
        self.dates_test = dates_test
        self.forecasts: Dict[str, List[float]] = {}

    def add_forecast(self, fit_model: Any, forecast: List[float]) -> None:
        """
        Add a forecast to the dataset.

        Parameters:
            fit_model (Any): The reference to the model used for the forecast.
            forecast (List[float]): The forecasted values.
        """
        self.forecasts[fit_model.get_reference()] = forecast

    def plot(self, figure_and_axes: Tuple[plt.Figure, plt.Axes] = None
             ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the training, test, and forecasted data.

        Parameters:
            figure_and_axes (Tuple[plt.Figure, plt.Axes]): Optional pre-defined
             figure and axes for plotting.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects used for
            the plot.
        """
        if not figure_and_axes:
            fig, ax = plt.subplots(figsize=(24, 6))
        else:
            fig, ax = figure_and_axes

        sns.lineplot(x=self.dates_train, y=self.y_train, color="black",
                     linewidth=1.25)

        sns.lineplot(x=[self.dates_train[-1]] + self.dates_test,
                     y=[self.y_train[-1]] + self.y_test,
                     color="black", linestyle="dashed", linewidth=1.25)

        forecasts_for_plotting = {model_ref: [self.y_train[-1], *forecast]
                                  for model_ref, forecast in self.forecasts.items()}
        forecasts_dataset = pd.DataFrame(forecasts_for_plotting)
        forecasts_dataset['date'] = [self.dates_train[-1]] + self.dates_test
        forecasts_dataset = pd.melt(forecasts_dataset, id_vars=['date'],
                                    var_name='model', value_name='forecast')

        sns.lineplot(x='date', y='forecast', hue="model",
                     data=forecasts_dataset, linewidth=1.5)

        plt.axvline(x=self.dates_train[-1], linestyle='--', color='red')

        return fig, ax

    def calculate_error(self) -> Dict[str, float]:
        """
        Calculate the mean squared error for each forecast model.

        Returns:
            Dict[str, float]: A dictionary containing the mean squared error
             for each model.
        """
        error = {}
        for model_reference, forecast in self.forecasts.items():
            error[model_reference] = sum(
                [(y_i - forecast_i) ** 2
                 for y_i, forecast_i in zip(self.y_test, forecast)]
            )
        return error


class ForecastResults:
    """
      A class to store, analyse and export forecast results.

    """

    def __init__(self, time_series_dataset: Any):
        """
        Initialize a ForecastResults object.

        Parameters:
            time_series_dataset (Any): The time series dataset used for
            forecasting.
        """
        self.time_series_dataset = time_series_dataset
        self.forecast_datasets: Dict[int, ForecastDataset] = {}

    def add_data_for_forecast(self, data_for_forecast: Dict[str, Any]) -> None:
        """
        Add data for forecasting to the ForecastResults object.

        Parameters:
            data_for_forecast (Dict[str, Any]): A dictionary containing the d
            ata for forecasting.
                It should contain 'y_train', 'y_test', 'dates_train', and '
                dates_test'.
        """
        forecast_dataset = ForecastDataset(y_train=data_for_forecast['y_train'],
                                           y_test=data_for_forecast['y_test'],
                                           dates_train=data_for_forecast['dates_train'],
                                           dates_test=data_for_forecast['dates_test'])
        self.forecast_datasets[data_for_forecast['window_index']] = forecast_dataset

    def add_forecast(self, window_index: int, fit_model: Any, forecast: List[float]) -> None:
        """
        Add a forecast to the specified window index.

        Parameters:
            window_index (int): The index of the window for which the forecast is added.
            fit_model (Any): The reference to the model used for the forecast.
            forecast (List[float]): The forecasted values.
        """
        self.forecast_datasets[window_index].add_forecast(fit_model=fit_model,
                                                          forecast=forecast)

    def plot_forecast_datasets(self, window_range: List[int]) -> None:
        """
        Plot the forecast datasets for the specified window range.

        Parameters:
            window_range (List[int]): The range of window indices to plot.
        """
        for window_index in window_range:
            fig, ax = self.forecast_datasets[window_index].plot()
            plt.show()

    def calculate_error_over_time(self, window_range: List[int] = None) -> pd.DataFrame:
        """
        Calculate the mean squared error for each forecast model over time.

        Parameters:
            window_range (List[int]): The range of window indices to consider.

        Returns:
            pd.DataFrame: A DataFrame containing the error data over time for
            each model and window index.
        """
        if window_range is None:
            window_range = range(len(self.forecast_datasets))
        error_over_time = []
        for window_index in window_range:
            error = self.forecast_datasets[window_index].calculate_error()
            predict_from = self.forecast_datasets[window_index].dates_test[0]
            error = {"window_index": window_index, "predict_from": predict_from, **error}
            error_over_time.append(error)
        return pd.DataFrame(error_over_time)

    def get_forecasts_dataset(self, window_range: List[int] = None):
        """
        Get the aggregated forecast dataset for the specified window range.


        :param window_range: The range of window indices to consider. If None, all
            available windows will be considered.

        :return: A dictionary containing aggregated forecast data for each model.
            The keys are model names, and the values are lists of forecasted values.
            The dictionary also contains 'date' and 'y_test' keys, which provide the
            corresponding dates and true target values.
        """

        if window_range is None:
            window_range = range(len(self.forecast_datasets))
        forecasts_all = self.forecast_datasets[window_range[0]].forecasts.copy()
        forecasts_all['date'] = [self.forecast_datasets[window_range[0]].dates_test[0]]
        forecasts_all['y_test'] = [self.forecast_datasets[window_range[0]].y_test[0]]
        for window_index in window_range[1:]:
            forecasts = self.forecast_datasets[window_index].forecasts
            for model_name, forecast in forecasts.items():
                forecasts_all[model_name] += forecast
            forecasts_all['date'].append(self.forecast_datasets[window_index].dates_test[0])
            forecasts_all['y_test'].append(self.forecast_datasets[window_index].y_test[0])
        forecasts_all = {
            var_name: variable[::-1] for var_name, variable in forecasts_all.items()
        }
        return forecasts_all

    def calculate_mean_model_error(self, window_range: List[int] = None) -> None:
        """
        Calculate and print the mean squared error for each forecast model.

        Parameters:
            window_range (List[int]): The range of window indices to consider.
        """
        if window_range is None:
            window_range = range(len(self.forecast_datasets))
        error_over_time = self.calculate_error_over_time(window_range)
        mean_model_error = error_over_time.drop(
            columns=['window_index', 'predict_from']
        ).mean()
        print("Model Performance")
        for model, error in zip(mean_model_error.index, mean_model_error.to_list()):
            print(f"Model: {model} - MSE: {np.round(error, 2)}")

    def plot_error_over_time(self, window_range: List[int]) -> None:
        """
        Plot the mean squared error for each model over time.

        Parameters:
            window_range (List[int]): The range of window indices to consider.
        """
        error_over_time = self.calculate_error_over_time(window_range)
        fig, ax = plt.subplots(figsize=(24, 6))
        error_over_time = pd.melt(error_over_time,
                                  id_vars=['predict_from', 'window_index'],
                                  var_name='model',
                                  value_name='error')
        sns.lineplot(x='predict_from', y='error', hue="model",
                     data=error_over_time, linewidth=1.5)
        plt.show()

    def plot_forecasts_on_whole_ts(self, window_range: List[int],
                                   n: int = None) -> None:
        """
        Plot the forecasts for the specified window range on the whole time series.

        Parameters:
            window_range (List[int]): The range of window indices to consider.
            n (int): The number of days before and after the forecast date to plot.
        """
        for window_index in window_range:
            if n:
                forecast_date = \
                self.forecast_datasets[window_index].dates_test[0]
                dates = []
                y_data = []
                for date, value in zip(self.time_series_dataset.dates,
                                       self.time_series_dataset.y_data):
                    if forecast_date - timedelta(
                            days=n) <= date <= forecast_date + timedelta(
                            days=n):
                        dates.append(date)
                        y_data.append(value)
            else:
                dates = self.time_series_dataset.dates
                y_data = self.time_series_dataset.y_data
            fig, ax = plt.subplots(figsize=(24, 6))
            sns.lineplot(x=dates, y=y_data, color="black", linewidth=1.25)
            fig, ax = self.forecast_datasets[window_index].plot(
                figure_and_axes=(fig, ax))
            plt.show()