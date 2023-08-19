import configparser
import os
import sys
from ForexForcasting.data_loader import ForexLoader, ConfigLoader
from ForexForcasting.backtesting import TimeseriesBacktestDataset
from ForexForcasting.backtesting import Backtester, BacktestWindow
from ForexForcasting.models import RandomWalk, ARIMA
from ForexForcasting.results import ForecastResults
from ForexForcasting.preprocessing import Preprocessor

def main(main_config):
    forex_loader = ForexLoader()
    data = forex_loader.load(
        source_type=main_config['general']['source_type'],
        path=main_config['general']['path']
    )
    data = data[data["currency_pair"] == "USD/COP"]
    data = data[data['Date'].dt.year > 2015]
    backtest_dataset = TimeseriesBacktestDataset(
        dates=data['Date'].to_list(),
        y_data=data['Close'].to_list(),
        window=BacktestWindow(
            n_obs=len(data),
            train_window_len=52,
            max_test_window_length=1
        )
    )
    config_loader = ConfigLoader()
    preprocessing_config = config_loader.load(
        main_config['general']['preprocessing_config_path']
    )
    preprocessor = Preprocessor(preprocessing_config)
    models = [
        RandomWalk(),
        ARIMA(1, 1, 0),
        ARIMA(2, 1, 0)
    ]
    backtester = Backtester(
        backtest_dataset=backtest_dataset,
        models=models,
        results=ForecastResults(backtest_dataset),
        preprocessor=preprocessor
    )
    backtester.run()
    backtester.results.plot_error_over_time(
        list(range(365))
    )
    backtester.results.calculate_mean_model_error(
        list(range(len(backtester.results.forecast_datasets)))
    )



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_file_path = sys.argv[1]
    config.read(config_file_path)
    main(config)

