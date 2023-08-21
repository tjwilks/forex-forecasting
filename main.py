import configparser
import os
import sys

import pandas as pd

from ForexForcasting.data_loader import ForexLoader, ConfigLoader, InterestRateLoader, InflationRateLoader, GDPGrowthRateLoader
from ForexForcasting.backtesting import TimeseriesBacktestDataset
from ForexForcasting.backtesting import Backtester, BacktestWindow
from ForexForcasting.models import RandomWalk, ARIMA, UIRPForecaster, TaylorRulesForecaster
from ForexForcasting.results import ForecastResults
from ForexForcasting.preprocessing import Preprocessor
pd.set_option('display.max_columns', None)  # Set to None to display all columns
pd.set_option('display.width', 1000000)  # Set to None to display all columns


def main(main_config):
    forex_loader = ForexLoader()
    interest_rate_loader = InterestRateLoader()
    inflation_rate_loader = InflationRateLoader()
    gdp_growth_rate_loader = GDPGrowthRateLoader()
    forex_data = forex_loader.load(
        source_type=main_config['general']['forex_source_type'],
        path=main_config['general']['forex_data_path']
    )
    interest_rate_data = interest_rate_loader.load(
        source_type=main_config['general']['interest_rate_source_type'],
        path=main_config['general']['interest_rate_data_path']
    )
    inflation_rate_data = inflation_rate_loader.load(
        source_type=main_config['general']['inflation_rate_source_type'],
        path=main_config['general']['inflation_rate_data_path']
    )
    gdp_growth_rate_data = gdp_growth_rate_loader.load(
        source_type=main_config['general']['gdp_growth_rate_source_type'],
        path=main_config['general']['gdp_growth_rate_data_path']
    )
    data = forex_data.merge(interest_rate_data, on=["currency_pair", 'date'])
    data = data.merge(inflation_rate_data, on=["currency_pair", 'date'])
    data = data.merge(gdp_growth_rate_data, on=["currency_pair", 'date'])
    data = data[data["currency_pair"] == "USD/COP"]
    data = data[data['date'].dt.year > 2015]
    backtest_dataset = TimeseriesBacktestDataset(
        dates=data['date'].to_list(),
        y_data=data['Close'].to_list(),
        window=BacktestWindow(
            n_obs=len(data),
            train_window_len=52,
            max_test_window_length=1
        ),
        regressor_data={
            "interest_rate_delta": data['interest_rate_delta'].to_list(),
            "inflation_rate_delta": data["inflation_rate_delta"].to_list(),
            "gdp_growth_rate_delta": data['gdp_growth_rate_delta'].to_list()
        }
    )
    config_loader = ConfigLoader()
    preprocessing_config = config_loader.load(
        main_config['general']['preprocessing_config_path']
    )
    preprocessor = Preprocessor(preprocessing_config)
    models = [
        RandomWalk(),
        ARIMA(1, 1, 0),
        UIRPForecaster(),
        TaylorRulesForecaster(num_lags=1),
    ]
    backtester = Backtester(
        backtest_dataset=backtest_dataset,
        models=models,
        results=ForecastResults(backtest_dataset),
        preprocessor=preprocessor
    )
    backtester.run()
    backtester.results.plot_error_over_time(
        list(range(len(backtester.results.forecast_datasets)))
    )
    backtester.results.calculate_mean_model_error(
        list(range(len(backtester.results.forecast_datasets)))
    )



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_file_path = sys.argv[1]
    config.read(config_file_path)
    main(config)

