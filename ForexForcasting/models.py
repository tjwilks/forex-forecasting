from abc import abstractmethod, ABC
from typing import List, Optional, Dict, Union, Tuple
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
    """
    RandomWalk class representing the Random Walk model.

    Attributes:
        last_observation_seen (Optional[float]): The last observation seen.

    """

    def __init__(self):
        """
        Initialize the Random Walk model.
        """
        self.last_observation_seen: Optional[float] = None
        self._accepts_regressors: bool = False

    def fit(
        self,
        y_train: List[float],
        regressors_train: Optional[Dict[str, List[Union[float, int]]]] = None
    ) -> None:
        """
        Fit the Random Walk model.

        Parameters:
            y_train (List[float]): The target values for training.
            regressors_train (Optional[Dict[str, List[Union[float, int]]]]): The regressor data.
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
    """
    ARIMA class representing the AutoRegressive Integrated Moving Average model.

    Parameters:
        auto_regressive (int, optional): The number of auto-regressive terms.
        integrated (int, optional): The order of differencing.
        moving_average (int, optional): The number of moving average terms.
        trend_type (str, optional): Type of trend component.

    Attributes:
        order (tuple): Order of ARIMA model.
        trend_type (str): Type of trend component.
        model (SARIMAX): The SARIMAX model instance.

    """

    def __init__(
            self,
            auto_regressive: int = 1,
            integrated: int = 0,
            moving_average: int = 0,
            trend_type: Optional[str] = None):

        """
        Initialize the ARIMA model.

        Parameters:
            auto_regressive (int, optional): The number of auto-regressive terms.
            integrated (int, optional): The order of differencing.
            moving_average (int, optional): The number of moving average terms.
            trend_type (str, optional): Type of trend component.
        """
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
            regressors_train (Dict[str, List[Union[float, int]]], optional): The regressor data.

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
    """
    UIRPForecaster class representing the Uncovered Interest Rate Parity model.

    Attributes:
        current_exchange_rate (Optional[float]): The current exchange rate.
        last_ir_delta (Optional[float]): The last interest rate delta.

    """

    _accepts_regressors = True

    def __init__(self):
        """
        Initialize the UIRPForecaster model.
        """
        self.current_exchange_rate: Optional[float] = None
        self.last_ir_delta: Optional[float] = None

    def fit(self, y_train: List[float], regressors_train: Dict[str, List[Union[float, int]]]) -> None:
        """
        Fit the UIRPForecaster model to the training data.

        Parameters:
            y_train (List[float]): Historical exchange rate data.
            regressors_train (Dict[str, List[Union[float, int]]]):
            Dictionary containing historical interest rate delta between
            two countries with exchange rate.
        """
        self.current_exchange_rate = y_train[-1]
        self.last_ir_delta = regressors_train['interest_rate_delta'][-1] / 100

    def predict(self, horizon: int) -> List[float]:
        """
        Make UIRP-based exchange rate change prediction.

        Parameters:
            horizon (int): The number of time steps to predict.

        Returns:
            List[float]: The predicted change in exchange rate.
        """
        last_exchange_rate = self.current_exchange_rate
        weekly_ir_delta = self.last_ir_delta / 52
        prediction = []
        for period in range(1, horizon + 1):
            predicted_exchange_rate = last_exchange_rate * (1 - weekly_ir_delta)
            prediction.append(predicted_exchange_rate)
            last_exchange_rate = predicted_exchange_rate
        return prediction

    def get_reference(self) -> str:
        """
        Get a reference string for the UIRPForecaster model.

        Returns:
            str: The reference string.
        """
        return "UIRP Forecaster"

class TaylorRulesForecaster(TimeSeriesModel):
    """
    TaylorRulesForecaster class representing the Taylor Rules forecasting model.

    Parameters:
        num_lags (int): Number of lagged values to consider.

    Attributes:
        num_lags (int): Number of lagged values to consider.
        model (LinearRegression): Linear regression model for prediction.
        last_observation (tuple): Tuple containing last observed y value and X value.

    """

    _accepts_regressors = True

    def __init__(self, num_lags: int):
        """
        Initialize the TaylorRulesForecaster model.

        Parameters:
            num_lags (int): Number of lagged values to consider.
        """
        self.num_lags = num_lags
        self.model = LinearRegression()
        self.last_observation = None

    def fit(self, y_train: List[float], regressors_train: Dict[str, List[float]] = None) -> None:
        """
        Fit the TaylorRulesForecaster model to the training data.

        Parameters:
            y_train (List[float]): Target values for training.
            regressors_train (Dict[str, List[float]], optional): Regressor data.

        Returns:
            None
        """
        X, y = self.create_datasets(y_train, regressors_train, self.num_lags)
        self.model.fit(X, y)
        self.last_observation = y_train[-1], X[-1]

    def predict(self, horizon: int) -> List[float]:
        """
        Make predictions using the TaylorRulesForecaster model.

        Parameters:
            horizon (int): Number of time steps to predict.

        Returns:
            List[float]: List of predicted values.
        """
        predictions = []
        for _ in range(horizon):
            last_y_val, last_x_val = self.last_observation
            predicted_change = self.model.predict(last_x_val.reshape(1, -1))
            prediction = last_y_val + predicted_change
            predictions.append(prediction)
            self.last_observation = prediction, last_x_val
        return predictions

    def get_reference(self) -> str:
        """
        Get a reference string for the TaylorRulesForecaster model.

        Returns:
            str: The reference string.
        """
        return "Taylor Rules Forecaster"

    @staticmethod
    def create_datasets(
            y_train: List[float],
            regressors_train: Dict[str, List[float]] = None,
            n_reg_lags: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create datasets for training the model.

        Parameters:
            y_train (List[float]): Target values for training.
            regressors_train (Dict[str, List[float]], optional): Regressor data.
            n_reg_lags (int, optional): Number of lagged values for regressors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing X and y arrays.
        """
        y = np.diff(y_train)
        y = y[n_reg_lags - 1:]
        X = []
        for i in range(n_reg_lags, len(y_train)):
            x = []
            for regressor_train in regressors_train.values():
                x += regressor_train[i - n_reg_lags:i]
            x += y_train[i - n_reg_lags:i]
            X.append(x)
        return np.array(X), np.array(y)


class AdaptiveHedge(TimeSeriesModel):
    """
    AdaptiveHedge class representing the adaptive hedge forecasting model.

    Parameters:
        alpha (float): Alpha parameter for exponential weighting.
        multiplier (float): Multiplier to adjust the impact of the errors.

    Attributes:
        alpha (float): Alpha parameter for exponential weighting.
        multiplier (float): Multiplier to adjust the impact of the errors.
        weights (Dict[str, float]): Weight assigned to each model.
        is_fit (bool): Flag indicating if the model has been fit to data.

    """

    _accepts_regressors = True

    def __init__(self, alpha: float, multiplier: float):
        """
        Initialize the AdaptiveHedge model.

        Parameters:
            alpha (float): Alpha parameter for exponential weighting.
            multiplier (float): Multiplier to adjust the impact of the errors.
        """
        self.alpha = alpha
        self.multiplier = multiplier
        self.weights = None
        self.is_fit = False

    def fit(self, y_train: List[float],
            regressors_train: Dict[str, List[float]]) -> None:
        """
        Fit the AdaptiveHedge model to the training data.

        Parameters:
            y_train (List[float]): Target values for training.
            regressors_train (Dict[str, List[float]]): Regressor data.

        Returns:
            None
        """
        self.fit_y_train = y_train
        self.fit_regressors_train = regressors_train
        exp_losses = {}
        sum_losses = {}
        for model_name, model_forecast in regressors_train.items():
            model_error = [
                abs(y_i - forecast_i)
                for y_i, forecast_i in zip(y_train, model_forecast)
            ]
            exp_decay_error = [
                ((1 - self.alpha) ** (i + 1)) * error_val
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

    def predict(self, x: Dict[str, List[float]]) -> List[float]:
        """
        Make predictions using the AdaptiveHedge model.

        Parameters:
            x (Dict[str, List[float]]): Dictionary containing forecasts from different models.

        Returns:
            List[float]: The weighted sum of forecasts.
        """
        weighted_forecasts = []
        for model_name in self.weights.keys():
            weighted_forecast = np.array(
                [self.weights[model_name] * forecast
                 for forecast in x[model_name]]
            )
            weighted_forecasts.append(weighted_forecast)
        return list(np.sum(weighted_forecasts, axis=0))

    def get_weights(self) -> np.ndarray:
        """
        Get the weights assigned to each model.

        Returns:
            np.ndarray: Array of weights.

        Raises:
            ValueError: If the model has not been fit to data yet.
        """
        if self.is_fit:
            return np.array(list(self.weights.values()))
        else:
            raise ValueError("Method has not been fit to data yet")

    def get_reference(self) -> str:
        """
        Get a reference string for the AdaptiveHedge model.

        Returns:
            str: The reference string.
        """
        return "AdaptiveHedge"


class LSTMBaseForecaster(nn.Module):
    """
    LSTMBaseForecaster class representing the LSTM model for forecasting.

    Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden state of LSTM.
        num_layers (int): Number of LSTM layers.

    Attributes:
        lstm (nn.LSTM): LSTM layer.
        linear (nn.Linear): Linear layer for final output.

    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        """
        Initialize the LSTMBaseForecaster.

        Parameters:
            input_size (int): Number of input features.
            hidden_size (int): Number of units in the hidden state of LSTM.
            num_layers (int): Number of LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        """
        Forward pass through the LSTMBaseForecaster.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class LSTMForecaster(TimeSeriesModel):
    """
    LSTMForecaster class for time series forecasting using LSTM.

    Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden state of LSTM.
        num_layers (int): Number of LSTM layers.
        num_epochs (int): Number of training epochs.

    Attributes:
        model (LSTMBaseForecaster): LSTM model.
        loss_function (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of training epochs.
        training_data_for_predict (torch.Tensor): Training data for prediction.

    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_epochs: int):
        super().__init__()
        self.model = LSTMBaseForecaster(input_size, hidden_size, num_layers)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.num_epochs = num_epochs
        self.training_data_for_predict = None

    def fit(
            self,
            y_train: List[float],
            regressors_train: Optional[Dict[str, List[Union[float, int]]]] = None
    ) -> None:
        """
        Fit the LSTM model to the training data.

        Parameters:
            y_train (List[float]): The target values for training.
            regressors_train (Optional[Dict[str, List[Union[float, int]]]]):
                The regressor data.
        """
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

    def predict(self, horizon: int) -> List[float]:
        """
        Make predictions using the LSTM model.

        Parameters:
            horizon (int): The number of time steps to predict.

        Returns:
            List[float]: The predicted values.
        """
        predictions = []
        with torch.no_grad():
            y_pred = self.model(self.training_data_for_predict)
            y_pred = y_pred[-1].item()
            predictions.append(y_pred)
        return predictions

    def get_reference(self) -> str:
        """
        Get a reference string for the LSTMForecaster model.

        Returns:
            str: The reference string.
        """
        return f"LSTMForecaster"

    def set_training_data_for_predict(
            self, y_train: List[float],
            regressors_train: Dict[str, List[Union[float, int]]]) -> None:
        """
        Set training data for prediction.

        Parameters:
            y_train (List[float]): The target values.
            regressors_train (Dict[str, List[Union[float, int]]]): The regressor data.
        """
        training_data_for_predict = []
        for i in range(len(y_train)):
            observation = [y_train[i]]
            for regressor in regressors_train.values():
                observation.append(regressor[i])
            training_data_for_predict.append(observation)
        self.training_data_for_predict = torch.tensor(training_data_for_predict)