from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

class Preprocessor:
    """
    A class for preprocessing data using various techniques based on a configuration.
    """

    def __init__(self, config: dict):
        """
        Initialize the Preprocessor with the provided configuration.

        :param config: A dictionary containing preprocessing configuration.
        """
        self.config = config
        self.steps = self.set_steps()

    def fit(self, data: list):
        """
        Fit the preprocessing steps to the data.

        :param data: The input data for fitting the preprocessing steps.
        """
        for step in self.steps.values():
            step.fit(np.array(data).reshape(-1, 1))

    def transform(self, data: list) -> list:
        """
        Apply the preprocessing steps to transform the data.

        :param data: The input data for applying the preprocessing steps.
        :return: Transformed data after preprocessing.
        """
        for step in self.steps.values():
            data = step.transform(np.array(data).reshape(-1, 1))
            data = data.flatten().tolist()
        return data

    def inverse_transform(self, data: list) -> list:
        """
        Apply inverse transformations to the data.

        :param data: The input data for applying inverse transformations.
        :return: Data after applying inverse transformations.
        """
        for step in self.config.keys():
            if self.config[step]['inverse_transform']:
                data = self.steps[step].inverse_transform(np.array(data).reshape(-1, 1))
                data = data.flatten().tolist()
        return data

    def set_steps(self) -> dict:
        """
        Set up the preprocessing steps based on the configuration.

        :return: A dictionary of preprocessing steps.
        """
        steps = dict()
        if "scaling" in self.config:
            steps['scaling'] = self.get_scaler()
        if "null_imputation" in self.config:
            steps['null_imputation'] = self.get_null_inputer()
        return steps

    def get_scaler(self):
        """
        Get a scaler based on the configuration.

        :return: A scaler instance.
        """
        if self.config['scaling']['strategy'] == "min_max":
            return MinMaxScaler()
        elif self.config['scaling']['strategy'] == "standard":
            return StandardScaler()
        else:
            raise ValueError(f"Invalid scaling strategy: "
                             f"{self.config['scaling']['strategy']}")

    def get_null_inputer(self):
        """
        Get a null imputer based on the configuration.

        :return: A null imputer instance.
        """
        if self.config['null_imputation']['strategy'] == "simple":
            return SimpleImputer()
        elif self.config['null_imputation']['strategy'] == "knn":
            return KNNImputer()
        else:
            raise ValueError(
                f"Invalid null imputation strategy: "
                f"{self.config.null_imputation['strategy']}")