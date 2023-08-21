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


class MacroLoader(DataLoader):

    def get_us_delta(self, data, macro_var):
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
    def convert_dates(data):
        months = data['date'].str[-2:].astype(int)
        years = data['date'].str[:4].astype(int)
        data['date'] = [
            pd.Timestamp(day=1, month=m, year=y) for m, y in zip(months, years)
        ]
        return data


class InterestRateLoader(MacroLoader):

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
        data = self.get_us_delta(data, "interest_rate")
        return data

    def load_from_csv(self, path):
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
    def get_currency_codes(data, path):
        data['currency'] = os.path.basename(path)[:3]
        return data


class InflationRateLoader(MacroLoader):

    def load_from_csv(self, path):
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
    def get_currency_codes(data):
        currency_code_dict = {
            "ARG": "ARS", "CHL": "CLP", "COL": "COP", "USA": "USD"
        }
        data['currency'] = data['Country Code'].apply(
            lambda country_code: currency_code_dict[country_code]
        )
        return data


class GDPGrowthRateLoader(MacroLoader):

    def load_from_csv(self, path):
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
    def convert_dates(data):
        month_mapping = {1: 1, 2: 4, 3: 7,  4: 10}
        months = data['TIME'].str[-1:].apply(lambda q: month_mapping[int(q)])
        years = data['TIME'].str[:4].astype(int)
        data['date'] = [
            pd.Timestamp(day=1, month=m, year=y) for m, y in zip(months, years)
        ]
        return data

    @staticmethod
    def get_currency_codes(data):
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