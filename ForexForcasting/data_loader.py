import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import List, Dict, Union

class ConfigLoader:
    """
    ConfigLoader class to load configuration from JSON.

    Methods:
        load(path: str) -> dict:
            Load the configuration from a JSON file.

    """

    def load(self, path: str) -> dict:
        """
        Load the configuration from a JSON file.

        Parameters:
            path (str): Path to the JSON file.

        Returns:
            dict: Loaded configuration.
        """
        with open(path, 'r') as j:
            config = json.loads(j.read())
        return config

class DataLoader:
    """
    DataLoader class to load data from different sources.

    Methods:
        load(source_type: str, path: str) -> pd.DataFrame:
            Load data from a source based on source type.

    """

    def load(self, source_type: str, path: str) -> pd.DataFrame:
        """
        Load data from a source based on source type.

        Parameters:
            source_type (str): Type of data source ('dir' or 'csv').
            path (str): Path to the data source.

        Returns:
            pd.DataFrame: Loaded data.
        """
        if source_type == 'dir':
            data = self.load_from_dir(path)
        elif source_type == 'csv':
            data = self.load_from_csv(path)
        else:
            raise ValueError("DataLoader load method's source_type argument must be either 'dir' or 'csv'")
        return data

    def load_from_dir(self, path: str) -> pd.DataFrame:
        """
        Load data from multiple CSV files in a directory.

        Parameters:
            path (str): Path to the directory.

        Returns:
            pd.DataFrame: Concatenated dataset from CSV files.
        """
        filenames = [
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        datasets = []
        for file in filenames:
            data = self.load_from_csv(file)
            datasets.append(data)
        return pd.concat(datasets, axis=0)

class ForexLoader(DataLoader):
    """
    ForexLoader class to load forex data.

    Methods:
        load_from_csv(path: str) -> pd.DataFrame:
            Load forex data from a CSV file.

    """

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """
        Load forex data from a CSV file.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded forex data.
        """
        data = pd.read_csv(
            path,
            parse_dates=['Date'],
            date_format="%Y-%m-%d"
        )
        data = data.rename(columns={"Date": "date"})
        data = data.set_index("date").resample('W').mean().reset_index()
        data['currency_pair'] = f"USD/{os.path.basename(path)[:3]}"
        return data


class MacroLoader(DataLoader):
    """
    MacroLoader class to load macroeconomic data.

    Methods:
        get_us_delta(data: pd.DataFrame, macro_var: str) -> pd.DataFrame:
            Calculate the delta of a macroeconomic variable relative to the US.
        convert_dates(data: pd.DataFrame) -> pd.DataFrame:
            Convert date strings to pandas Timestamp objects.

    """

    def get_us_delta(self, data: pd.DataFrame, macro_var: str) -> pd.DataFrame:
        """
        Calculate the delta of a macroeconomic variable relative to the US.

        Parameters:
            data (pd.DataFrame): DataFrame containing macroeconomic data.
            macro_var (str): Macro variable name.

        Returns:
            pd.DataFrame: DataFrame with added delta column.
        """
        us_data = data[data['currency'] == "USD"]
        us_data = us_data.drop(columns="currency")
        us_data = us_data.rename(columns={macro_var: f"us_{macro_var}"})
        data = data[data['currency'] != "USD"]
        data = data.merge(us_data, on="date", how="inner")
        data[f'{macro_var}_delta'] = data[macro_var] - data[f"us_{macro_var}"]
        data['currency_pair'] = "USD/" + data['currency']
        data = data.drop(columns=[macro_var, f"us_{macro_var}", "currency"])
        return data

    @staticmethod
    def convert_dates(data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date strings to pandas Timestamp objects.

        Parameters:
            data (pd.DataFrame): DataFrame containing date column.

        Returns:
            pd.DataFrame: DataFrame with converted date column.
        """
        months = data['date'].str[-2:].astype(int)
        years = data['date'].str[:4].astype(int)
        data['date'] = [
            pd.Timestamp(day=1, month=m, year=y) for m, y in zip(months, years)
        ]
        return data

class InterestRateLoader(MacroLoader):
    """
    InterestRateLoader class to load interest rate data.

    Methods:
        load_from_dir(path: str) -> pd.DataFrame:
            Load interest rate data from multiple files in a directory.
        load_from_csv(path: str) -> pd.DataFrame:
            Load interest rate data from a CSV file.

    """

    def load_from_dir(self, path: str) -> pd.DataFrame:
        """
        Load interest rate data from multiple files in a directory.

        Parameters:
            path (str): Path to the directory containing CSV files.

        Returns:
            pd.DataFrame: Loaded interest rate data.
        """
        filenames = [
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        datasets = []
        for file in filenames:
            data = self.load_from_csv(file)
            datasets.append(data)
        data = pd.concat(datasets, axis=0)
        data = self.get_us_delta(data, "interest_rate")
        return data

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """
        Load interest rate data from a CSV file.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded interest rate data.
        """
        data = pd.read_csv(path)
        data = data.drop(
            columns=[col for col in data.columns if "Unnamed: " in col]
        )
        data = pd.melt(
            data,
            id_vars=['Indicator'],
            var_name='date',
            value_name='interest_rate'
        )
        data = data[data["Indicator"] == "Central Bank Policy Rate"]
        data = data.drop(columns="Indicator")
        data = self.convert_dates(data)
        data = self.get_currency_codes(data, path)
        data['interest_rate'] = data['interest_rate'].replace("...", np.NAN)
        data = data.astype({
            "currency": "string",
            "interest_rate": "float"
        })
        data = data.set_index("date").resample('W').ffill().reset_index()
        return data

    @staticmethod
    def get_currency_codes(data: pd.DataFrame, path: str) -> pd.DataFrame:
        """
        Get currency codes from the path and add them to the DataFrame.

        Parameters:
            data (pd.DataFrame): DataFrame containing interest rate data.
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame with added currency codes.
        """
        data['currency'] = os.path.basename(path)[:3]
        return data


class InflationRateLoader(MacroLoader):
    """
    InflationRateLoader class to load inflation rate data.

    Methods:
        load_from_csv(path: str) -> pd.DataFrame:
            Load inflation rate data from a CSV file.

    """

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """
        Load inflation rate data from a CSV file.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded inflation rate data.
        """
        data = pd.read_csv(path)
        data = pd.melt(
            data,
            id_vars=['Country Code'],
            var_name='date',
            value_name='inflation_rate'
        )
        data["inflation_rate"] = data["inflation_rate"].astype(str)
        data["inflation_rate"] = data["inflation_rate"].str.replace(
            ",", ".").str.replace(
            " ", "")
        data["inflation_rate"] = data["inflation_rate"].astype(float)
        data = self.convert_dates(data)
        data = self.get_currency_codes(data)
        data = data.drop(columns="Country Code")
        data = self.get_us_delta(data, "inflation_rate")
        data = data.set_index("date")
        data = data.groupby(
            "currency_pair", as_index=False).resample('W').ffill()
        data = data.reset_index()
        data = data.drop(columns="level_0")
        return data

    @staticmethod
    def get_currency_codes(data: pd.DataFrame) -> pd.DataFrame:
        """
        Get currency codes and add them to the DataFrame.

        Parameters:
            data (pd.DataFrame): DataFrame containing inflation rate data.

        Returns:
            pd.DataFrame: DataFrame with added currency codes.
        """
        currency_code_dict = {
            "ARG": "ARS", "CHL": "CLP", "COL": "COP", "USA": "USD"
        }
        data['currency'] = data['Country Code'].apply(
            lambda country_code: currency_code_dict[country_code]
        )
        return data


class GDPGrowthRateLoader(MacroLoader):
    """
    GDPGrowthRateLoader class to load GDP growth rate data.

    Methods:
        load_from_csv(path: str) -> pd.DataFrame:
            Load GDP growth rate data from a CSV file.

    """

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """
        Load GDP growth rate data from a CSV file.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded GDP growth rate data.
        """
        data = pd.read_csv(path)
        data = data[data['Subject'] == "Gross domestic product - expenditure "
                                       "approach"]
        data = data[
            data['Measure'] == "Growth rate based on seasonally adjusted "
                               "volume data, percentage change on the "
                               "previous quarter"
            ]
        data = data[["Country", "TIME", "Value"]]
        data = self.convert_dates(data)
        data = self.get_currency_codes(data)
        data = data.drop(columns=['Country', 'TIME'])
        data = data.rename(columns={"Value": "gdp_growth_rate"})
        data = self.get_us_delta(data, "gdp_growth_rate")
        data = data.set_index("date").groupby(
            "currency_pair", as_index=False).resample('W').ffill()
        data = data.reset_index()
        data = data.drop(columns="level_0")
        return data

    @staticmethod
    def convert_dates(data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date strings to pandas Timestamp objects.

        Parameters:
            data (pd.DataFrame): DataFrame containing date column.

        Returns:
            pd.DataFrame: DataFrame with converted date column.
        """
        month_mapping = {1: 1, 2: 4, 3: 7,  4: 10}
        months = data['TIME'].str[-1:].apply(lambda q: month_mapping[int(q)])
        years = data['TIME'].str[:4].astype(int)
        data['date'] = [
            pd.Timestamp(day=1, month=m, year=y) for m, y in zip(months, years)
        ]
        return data

    @staticmethod
    def get_currency_codes(data: pd.DataFrame) -> pd.DataFrame:
        """
        Get currency codes and add them to the DataFrame.

        Parameters:
            data (pd.DataFrame): DataFrame containing GDP growth rate data.

        Returns:
            pd.DataFrame: DataFrame with added currency codes.
        """
        currency_code_dict = {
            "Argentina": "ARS",
            "Chile": "CLP",
            "Colombia": "COP",
            "United States": "USD"
        }
        data['currency'] = data['Country'].apply(
            lambda country_code: currency_code_dict[country_code]
        )
        return data


class EconNewsDataLoader(DataLoader):
    """
    EconNewsDataLoader class to load economic news sentiment data.

    Methods:
        load_from_csv(path: str) -> pd.DataFrame:
            Load economic news sentiment data from a CSV file.

    """

    def __init__(self, date_downloaded: str):
        """
        Initialize the EconNewsDataLoader.

        Parameters:
            date_downloaded (str): Date of data download.
        """
        self.date_downloaded = date_downloaded

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """
        Load economic news sentiment data from a CSV file.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded economic news sentiment data.
        """
        data = pd.read_csv(path)
        data = data.drop(
            columns=[col for col in data.columns if "Unnamed: " in col]
        )
        data.columns = ['all_data']
        data = data[~data['all_data'].isna()]
        dates = self.calculate_dates(data)
        articles = self.extract_articles(data)
        sentiment_scores = self.calculate_sentiment(articles)
        data = pd.DataFrame(
            {"date": pd.to_datetime(dates), "sentiment_score": sentiment_scores}
        )
        data = data.set_index("date").groupby("date").mean().resample('D').ffill()
        data = data.resample('W').mean().reset_index()
        data["currency_pair"] = f"USD/{os.path.basename(path)[:3]}"
        return data

    @staticmethod
    def calculate_sentiment(articles: List[str]) -> List[float]:
        """
        Calculate sentiment scores for a list of articles.

        Parameters:
            articles (List[str]): List of article texts.

        Returns:
            List[float]: List of sentiment scores.
        """
        model_name = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )
        inputs = tokenizer(articles, padding=True, truncation=True,
                           return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        sentiment_scores = outputs.logits.mean(dim=1).tolist()
        return sentiment_scores

    @staticmethod
    def extract_articles(data: pd.DataFrame) -> List[str]:
        """
        Extract articles from the DataFrame.

        Parameters:
            data (pd.DataFrame): DataFrame containing article data.

        Returns:
            List[str]: List of article texts.
        """
        first_article = data.iloc[1, :]
        articles = data.loc[data['all_data'].str.startswith("\t"), "all_data"]
        articles = pd.concat([first_article, articles], axis=0).to_list()
        return articles

    def calculate_dates(self, data: pd.DataFrame) -> List[str]:
        """
        Calculate dates from the article data.

        Parameters:
            data (pd.DataFrame): DataFrame containing article data.

        Returns:
            List[str]: List of calculated dates.
        """
        days_ago = data[data['all_data'].str.contains("days ago")]
        days_ago = days_ago['all_data'].str[:2].apply(
            lambda day_ago:
            datetime.strptime(self.date_downloaded, "%Y-%m-%d") -
            timedelta(days=int(day_ago))
        )
        days_ago = days_ago.dt.date
        dates = data.loc[data['all_data'].str.match(
            r'^\d{4}-\d{2}-\d{2}'),
                         "all_data"
        ]
        dates = pd.to_datetime(dates.str[:10], format="%Y-%m-%d").dt.date
        dates = pd.concat([days_ago, dates], axis=0).to_list()
        return dates
