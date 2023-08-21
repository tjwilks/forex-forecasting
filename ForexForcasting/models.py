from abc import abstractmethod, ABC
from typing import List, Optional, Dict, Union
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
import numpy as np


class TimeSeriesModel(ABC):
    _accepts_regressors: bool = False

    @property
    def accepts_regressors(self) -> bool:
        """
        Check if the model accepts regressors.

        Returns:
            bool: True if the model accepts regressors, False otherwise.
        """
        return self._accepts_regressors

    @abstractmethod
    def fit(
            self,
            y_train: List[float],
            regressors_train: Optional[Dict[str, List[Union[float, int]]]]
    ) -> None:
        """
        Fit the model to the training data.

        Parameters:
            y_train (List[float]): The target values for training.
            regressors_train: Optional[Dict[str, List[Union[float, int]]]]:
                The regressor data
        """
        pass

    @abstractmethod
    def predict(self, horizon: int) -> List[float]:
        """
        Make predictions for the given horizon.

        Parameters:
            horizon (int): The number of time steps to predict.

        Returns:
            List[float]: The predicted values.
        """
        pass

    @abstractmethod
    def get_reference(self) -> str:
        """
        Get a reference string for the model.

        Returns:
            str: The reference string.
        """
        pass


class RandomWalk(TimeSeriesModel):

    def __init__(self):
        self.last_observation_seen: Optional[float] = None
        self._accepts_regressors: bool = False

    def fit(
        self,
        y_train:
        List[float],
        regressors_train: Optional[Dict[str, List[Union[float, int]]]] = None
    ) -> None:
        """
        Fit the Random Walk model.

        Parameters:
            y_train (List[float]): The target values for training.
            regressors_train: Optional[Dict[str, List[Union[float, int]]]]: 
                The regressor data
        """
        self.last_observation_seen = y_train[-1]

    def predict(self, horizon: int) -> List[float]:
        """
        Make predictions using the Random Walk model.

        Parameters:
            horizon (int): The number of time steps to predict.

        Returns:
            List[float]: The predicted values.
        """
        return [self.last_observation_seen for _ in range(horizon)]

    def get_reference(self) -> str:
        """
        Get a reference string for the Random Walk model.

        Returns:
            str: The reference string.
        """
        return "RandomWalk"


class ARIMA(TimeSeriesModel):

    def __init__(
            self,
            auto_regressive: int = 1,
            integrated: int = 0,
            moving_average: int = 0,
            trend_type: Optional[str] = None):

        self.order = (auto_regressive, integrated, moving_average)
        self.trend_type = trend_type
        self.model: Optional[SARIMAX] = None
        self._accepts_regressors: bool = False

    def fit(
        self,
        y_train: List[float],
        regressors_train: Optional[Dict[str, List[Union[float, int]]]] = None
    ) -> None:
        """
        Fit the ARIMA model.

        Parameters:
            y_train (List[float]): The target values.
            regressors_train: Optional[Dict[str, List[Union[float, int]]]]:
                The regressor data
        Returns:
            None
        """
        self.model = SARIMAX(
            endog=y_train,
            order=self.order,
            trend=self.trend_type,
            enforce_invertibility=False,
            enforce_stationarity=False
        ).fit(disp=0)

    def predict(self, horizon: int) -> List[float]:
        """
        Make predictions using the ARIMA model.

        Parameters:
            horizon (int): The number of time steps to predict.

        Returns:
            List[float]: The predicted values.
        """
        return self.model.forecast(horizon)

    def get_reference(self) -> str:
        """
        Get a reference string for the ARIMA model.

        Returns:
            str: The reference string.
        """
        reference = f"ARIMA (AR: {self.order[0]}, " \
                    f"I: {self.order[1]}, " \
                    f"MA: {self.order[2]})"
        return reference


class UIRPForecaster(TimeSeriesModel):
    _accepts_regressors = True

    def __init__(self):
        self.current_exchange_rate: Optional[float] = None
        self.last_ir_delta: Optional[float] = None

    def fit(self, y_train: List[float], regressors_train: Dict[str, List[Union[float, int]]]) -> None:
        """
        Fit the UIRPForecaster model to the training data.

        y_train (List[float]): Historical exchange rate data.
        regressors_train Dict[str, List[Union[float, int]]]: Dictionary
            containing historical interest rate delta between two countries
            with exchange rate
        """
        self.current_exchange_rate = y_train[-1]
        self.last_ir_delta = regressors_train['interest_rate_delta'][-1]/100

    def predict(self, horizon: int) -> List[float]:
        """
        Make UIRP-based exchange rate change prediction.

        :return: The predicted change in exchange rate.
        """
        last_exchange_rate = self.current_exchange_rate
        weekly_ir_delta = self.last_ir_delta / 52
        prediction = []
        for period in range(1, horizon+1):
            predicted_exchange_rate = last_exchange_rate * (1-weekly_ir_delta)
            prediction.append(predicted_exchange_rate)
            last_exchange_rate = predicted_exchange_rate
        return prediction

    def get_reference(self) -> str:
        """
        Get a reference string for the UIRPForecaster model.

        :return: The reference string.
        """
        return "UIRP Forecaster"


class TaylorRulesForecaster(TimeSeriesModel):
    _accepts_regressors = True

    def __init__(self, num_lags):
        self.num_lags = num_lags
        self.model = LinearRegression()
        self.last_observation = None

    def fit(self, y_train, regressors_train=None):
        X, y = self.create_datasets(y_train, regressors_train, self.num_lags)
        self.model.fit(X, y)
        self.last_observation = y_train[-1], X[-1]

    def predict(self, horizon):
        predictions = []
        for _ in range(horizon):
            last_y_val, last_x_val = self.last_observation
            predicted_change = self.model.predict(last_x_val.reshape(1, -1))
            prediction = last_y_val + predicted_change
            predictions.append(prediction)
            self.last_observation = prediction, last_x_val
        return predictions

    def get_reference(self):
        return "Taylor Rules Forecaster"

    @staticmethod
    def create_datasets(
            y_train,
            regressors_train=None,
            n_reg_lags=1
    ):
        y = np.diff(y_train)
        y = y[n_reg_lags-1:]
        X = []
        for i in range(n_reg_lags, len(y_train)):
            x = []
            for regressor_train in regressors_train.values():
                x += regressor_train[i - n_reg_lags:i]
            x += y_train[i-n_reg_lags:i]
            X.append(x)
        return np.array(X), np.array(y)
