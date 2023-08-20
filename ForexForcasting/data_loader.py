import os
import numpy as np
import pandas as pd
import json


class ConfigLoader:

    def load(self, path):
        with open(path, 'r') as j:
            config = json.loads(j.read())
        return config


class DataLoader:

    def load(self, source_type, path):
        if source_type == 'dir':
            data = self.load_from_dir(path)
        elif source_type == 'csv':
            data = self.load_from_csv(path)
        else:
            raise ValueError("DataLoader load method's source_type argument"
                             'must be either "dir" or "csv"')
        return data


class ForexLoader(DataLoader):

    def load_from_dir(self, path):
        filenames = [
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        datasets = []
        for file in filenames:
            data = self.load_from_csv(file)
            datasets.append(data)
        return pd.concat(datasets, axis=0)

    def load_from_csv(self, path):
        data = pd.read_csv(
            path,
            parse_dates=['Date'],
            date_format="%Y-%m-%d"
        )
        data = data.rename(columns={"Date": "date"})
        data = data.set_index("date").resample('W').mean().reset_index()
        data['currency_pair'] = f"USD/{os.path.basename(path)[:3]}"
        return data


class InterestRateLoader(DataLoader):

    def load_from_dir(self, path):
        filenames = [
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        datasets = []
        for file in filenames:
            data = self.load_from_csv(file)
            datasets.append(data)
        data = pd.concat(datasets, axis=0)
        us_data = data[data['currency'] == "USD"]
        us_data = us_data.drop(columns="currency")
        us_data = us_data.rename(columns={"interest_rate": "us_interest_rate"})
        data = data[data['currency'] != "USD"]
        data = data.merge(us_data, on="date", how="inner")
        data['interest_rate_delta'] = data["interest_rate"] - data["us_interest_rate"]
        data['currency_pair'] = "USD/" + data['currency']
        data = data.drop(columns=["interest_rate", "us_interest_rate", "currency"])
        return data

    def load_from_csv(self, path):
        data = pd.read_csv(
            path
        )
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
        months = data['date'].str[-2:].astype(int)
        years = data['date'].str[:4].astype(int)
        data['date'] = [
            pd.Timestamp(day=1, month=m, year=y) for m, y in zip(months, years)
        ]
        data['interest_rate'] = data['interest_rate'].replace("...", np.NAN)
        data['currency'] = os.path.basename(path)[:3]
        data = data.astype({
            "currency": "string",
            "interest_rate": "float"
        })
        data = data.set_index("date").resample('W').ffill().reset_index()
        return data
