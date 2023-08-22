from typing import List, Union, Dict
from datetime import datetime
from ForexForcasting.models import TimeSeriesModel
from ForexForcasting.results import ForecastResults
from ForexForcasting.preprocessing import Preprocessor


class BacktestWindow:

    def __init__(self,
                 n_obs: int,
                 train_window_len: int,
                 max_test_window_length: int
                 ) -> None:
        """
        Initializes a new instance of the Window class.

        :param n_obs: total number of observations
        :param train_window_len: The length of the training window.

        """
        max_time_series_index = n_obs - 1
        max_requested_time_series_index = train_window_len
        assert max_time_series_index >= max_requested_time_series_index, \
            "time series is too short to initialise window with requested " \
            "train_window_length"
        self.test_end_index = n_obs - 1
        self.test_start_index = n_obs - 1
        self.train_end_index = n_obs - 2
        self.train_start_index = n_obs - 1 - train_window_len
        self.test_window_length = 1
        self.max_test_window_length = max_test_window_length
        self.index = 0

    def is_not_last(self):
        """
        Check if the current window is not the last window in the time series.

        :return: True if the current window is not the last; otherwise, False.
        """
        return self.train_start_index != 0

    def next(self):
        """
        Move the window to the next position in the time series.

        :raises AssertionError: If the start index of the training window is
        less than 0.
        """
        assert self.is_not_last(), "Start index of training window can not " \
                                   "be less than 0"
        self.train_start_index -= 1
        self.train_end_index -= 1
        if self.test_window_length == self.max_test_window_length:
            self.test_end_index -= 1
        self.test_start_index -= 1
        self.test_window_length = self.test_end_index - self.test_start_index + 1
        self.index += 1

    def get_train_index(self):
        """
        Get the start and end indices of the training window.

        :return: A tuple of the start and end indices of the training window.
        """
        return self.train_start_index, self.train_end_index

    def get_test_index(self):
        """
        Get the start and end indices of the test window.

        :return: A tuple of the start and end indices of the test window.
        """
        return self.test_start_index, self.test_end_index


class TimeseriesBacktestDataset:
    """
    A class representing a time series backtest dataset.

    :param dates: A list of dates corresponding to the time series data.
    :param y_data: A list of target values for the time series.
    :param window: An instance of BacktestWindow representing the backtest
    window.
    :param regressor_data: A list of lists representing additional regressor
    data (optional).
    """
    def __init__(
            self,
            dates: List[datetime.date],
            y_data: List[Union[float, int]],
            window: BacktestWindow,
            regressor_data: Dict[str, List[Union[float, int]]] = None,
    ):
        assert len(dates) == len(y_data), f"dates and y_data parameters must "\
                                          f"be of same length." \
                                          f"\ndates length:" f"{len(dates)} " \
                                          f"\ny_data length {len(y_data)}"
        self.dates = dates
        self.y_data = y_data
        self.regressor_data = regressor_data
        self.window = window

    def _get_start_and_end_index(self, train_or_test):
        """
        Get the start and end indices of the training or test window.

        :param train_or_test: Either "train" or "test" to specify the window
        type.
        :return: A tuple of the start and end indices of the window.
        :raises ValueError: If an invalid value for train_or_test is provided.
        """
        if train_or_test == "train":
            return self.window.get_train_index()
        elif train_or_test == "test":
            return self.window.get_test_index()
        else:
            raise ValueError("train_or_test must be either 'train' or 'test'")

    def get_data(self, requested_data, train_or_test):
        """
        Get the requested data (target values, dates, or regressors) for the
        specified window.

        :param requested_data: The type of data to retrieve ("y", "dates", or
         "regressors").
        :param train_or_test: Either "train" or "test" to specify the window
        type.
        :return: The requested data for the specified window.
        :raises ValueError: If an invalid value for requested_data or
        train_or_test is provided.
        """
        start_index, end_index = self._get_start_and_end_index(train_or_test)
        if requested_data == "y":
            return self.y_data[start_index: end_index+1]
        elif requested_data == "dates":
            return self.dates[start_index: end_index+1]
        elif requested_data == "regressors":
            if self.regressor_data:
                return {regressor_name: regressor[start_index: end_index+1]
                        for regressor_name, regressor in self.regressor_data.items()}
            else:
                return None
        else:
            raise ValueError("requested_data must be 'y', 'dates' "
                             "or regressors")

    def get_data_for_forecast(self):
        """
        Get the data required for forecasting for the current window.

        :return: A dictionary containing the data required for forecasting.
        """
        data_for_forecast = {
            "y_train": self.get_data("y", "train"),
            "y_test": self.get_data("y", "test"),
            "dates_train": self.get_data("dates", "train"),
            "dates_test": self.get_data("dates", "test"),
            "regressors_train": self.get_data("regressors", 'train'),
            "regressors_test": self.get_data("regressors", 'test'),
            "window_index": self.window.index
         }
        return data_for_forecast

    def next_window(self):
        """
        Move the backtest window to the next position in the time series.
        """
        self.window.next()


class Backtester:
    """
    A class representing a backtester for time series forecasting.

    :param backtest_dataset: An instance of TimeseriesBacktestDataset
     representing the dataset.
    :param preprocessor: An instance of YourPreprocessorClass representing the
    data preprocessor.
    :param models: A list of YourModelClass instances representing the
    forecasting models.
    :param results: An instance of YourResultsClass representing the results
    collector.
    :param mode: either "primary_model" or "model_selection"
    """
    def __init__(
            self,
            backtest_dataset: TimeseriesBacktestDataset,
            preprocessor: Preprocessor,
            models: List[TimeSeriesModel],
            results: ForecastResults,
            mode: str
    ) -> None:
        self.backtest_dataset = backtest_dataset
        self.preprocessor = preprocessor
        self.models = models
        self.results = results
        self.mode = mode

    def run(self):
        """
        Run the backtesting process using the specified forecasting models.

        The backtesting process involves iterating through the dataset's
        windows,
        forecasting target values using the specified models, and collecting
        the results.
        """
        while self.backtest_dataset.window.is_not_last():
            data_for_forecast = self.backtest_dataset.get_data_for_forecast()
            self.results.add_data_for_forecast(
                data_for_forecast=data_for_forecast
            )
            self.forecast_models(
                window_index=data_for_forecast['window_index'],
                y_train=data_for_forecast['y_train'],
                y_test=data_for_forecast['y_test'],
                regressors_train=data_for_forecast['regressors_train'],
                regressors_test=data_for_forecast['regressors_test']
            )
            self.backtest_dataset.window.next()

    def forecast_models(self, window_index: int, y_train: List[float], y_test: List[float],
                        regressors_train: Dict[str, List[Union[float, int]]]=None,
                        regressors_test: Dict[str, List[Union[float, int]]]=None):
        """
        Forecast target values using the specified models.

        :param window_index: The index of the current backtest window.
        :param y_train: The target values of the training window.
        :param y_test: The target values of the test window.
        :param regressors_train: Additional regressor data for the training
        window (optional).
        :param regressors_test: Additional regressor data for the test window
        (optional).
        """

        regressors_train, regressors_test = self.preprocess_regressors(
            regressors_train=regressors_train,
            regressors_test=regressors_test
        )
        self.preprocessor.fit(y_train)
        y_train = self.preprocessor.transform(y_train)
        y_test = self.preprocessor.transform(y_test)
        for model in self.models:
            model.fit(y_train=y_train, regressors_train=regressors_train)
            if self.mode == "model_selection":
                forecast = model.predict(regressors_test)
            elif self.mode == "primary_model":
                forecast = model.predict(horizon=len(y_test))
            forecast = self.preprocessor.inverse_transform(forecast)
            self.results.add_forecast(
                window_index=window_index,
                fit_model=model,
                forecast=forecast
            )

    def preprocess_regressors(self,
                              regressors_train: Dict[str, List[Union[float, int]]],
                              regressors_test:  Dict[str, List[Union[float, int]]]):
        if regressors_train:
            for regressor_name in regressors_train.keys():
                regressor_train = regressors_train[regressor_name]
                regressor_test = regressors_test[regressor_name]
                self.preprocessor.fit(regressor_train)
                regressors_train[regressor_name] = self.preprocessor.transform(
                    regressor_train
                )
                regressors_test[regressor_name] = self.preprocessor.transform(
                    regressor_test
                )
        return regressors_train, regressors_test