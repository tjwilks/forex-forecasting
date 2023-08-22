from abc import abstractmethod, ABC
from typing import List, Optional, Dict, Union
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as data
torch.manual_seed(42)

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


class AdaptiveHedge(TimeSeriesModel):
    _accepts_regressors = True

    def __init__(self, alpha, multiplier):
        self.alpha = alpha
        self.multiplier = multiplier
        self.weights = None

    def fit(self, y_train, regressors_train):
        self.fit_y_train = y_train
        self.fit_regressors_train = regressors_train
        exp_losses = {}
        sum_losses = {}
        for model_name, model_forecast in regressors_train.items():
            model_error = [
                abs(y_i-forecast_i)
                for y_i, forecast_i in zip(y_train, model_forecast)
            ]
            exp_decay_error = [
                ((1 - self.alpha)**(i+1))*error_val
                for i, error_val in enumerate(model_error)
            ]
            sum_loss = np.abs(exp_decay_error).sum()
            exp_loss = math.e ** (-self.multiplier * sum_loss)
            sum_losses[model_name] = sum_loss
            exp_losses[model_name] = exp_loss
        sum_exp_losses = sum(exp_losses.values())
        self.weights = {model_name: exp_loss / sum_exp_losses for
                        model_name, exp_loss in exp_losses.items()}
        self.is_fit = True

    def predict(self, x):
        weighted_forecasts = []
        for model_name in self.weights.keys():
            weighted_forecast = np.array(
                [self.weights[model_name]*forecast
                 for forecast in x[model_name]]
            )
            weighted_forecasts.append(weighted_forecast)
        return sum(weighted_forecasts)

    def get_weights(self):
        if self.is_fit:
            return np.array(list(self.weights.values()))
        else:
            raise ValueError("Method has not been fit to data yet")

    def get_reference(self) -> str:
        return "AdaptiveHedge"


class LSTMBaseForecaster(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class LSTMForecaster(TimeSeriesModel):
    def __init__(self, input_size, hidden_size, num_layers, num_epochs):
        self.model = LSTMBaseForecaster(input_size, hidden_size, num_layers)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.num_epochs = num_epochs
        self.training_data_for_predict = None

    def fit(self, y_train, regressors_train=None):
        x, y = [], []
        for i in range(len(y_train) - 1):
            feature = y_train[i:i + 1]
            for regressor in regressors_train.values():
                feature.append(regressor[i:i + 1][0])
            target = y_train[i + 1:i + 1 + 1]
            x.append(feature)
            y.append(target)

        x = torch.tensor(x)
        y = torch.tensor(y)
        dataset = data.TensorDataset(x, y)
        loader = data.DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=8
        )
        for epoch in range(self.num_epochs):
            self.model.train()
            for X_batch, y_batch in loader:
                y_pred = self.model(X_batch)
                loss = self.loss_function(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.set_training_data_for_predict(y_train, regressors_train)

    def predict(self, horizon):
        predictions = []
        with torch.no_grad():
            y_pred = self.model(self.training_data_for_predict)
            y_pred = y_pred[-1].item()
            predictions.append(y_pred)
        return predictions

    def get_reference(self) -> str:
        return f"LSTMForecaster"

    def set_training_data_for_predict(self, y_train, regressors_train):
        training_data_for_predict = []
        for i in range(len(y_train)):
            observation = [y_train[i]]
            for regressor in regressors_train.values():
                observation.append(regressor[i])
            training_data_for_predict.append(observation)
        self.training_data_for_predict = torch.tensor(training_data_for_predict)