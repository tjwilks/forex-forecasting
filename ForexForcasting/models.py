from abc import abstractmethod, ABC
from typing import List, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
    def fit(self, training_data: List[float]) -> None:
        """
        Fit the model to the training data.

        Parameters:
            training_data (List[float]): The training data.
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

    def fit(self, y_train: List[float]) -> None:
        """
        Fit the Random Walk model.

        Parameters:
            y_train (List[float]): The target values for training.
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

    def fit(self, y: List[float]) -> None:
        """
        Fit the ARIMA model.

        Parameters:
            y (List[float]): The target values.

        Returns:
            None
        """
        self.model = SARIMAX(
            endog=y,
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
