import sys
from configparser import ConfigParser
from ForexForcasting.data_loader import (
    ForexLoader,
    ConfigLoader,
    InterestRateLoader,
    InflationRateLoader,
    GDPGrowthRateLoader,
    EconNewsDataLoader,
    DataLoaderComposite,
)
from ForexForcasting.backtesting import (
    TimeseriesBacktestDataset,
    Backtester,
    BacktestWindow,
)
from ForexForcasting.models import (
    RandomWalk,
    ARIMA,
    UIRPForecaster,
    TaylorRulesForecaster,
    AdaptiveHedge,
    LSTMForecaster,
)
from ForexForcasting.results import ForecastResults
from ForexForcasting.preprocessing import Preprocessor


def main(main_config):
    # load data for analysis
    data_loader = DataLoaderComposite(
        loaders={
            "forex": ForexLoader(),
            "interest_rate": InterestRateLoader(),
            "inflation_rate": InflationRateLoader(),
            "gdp_growth_rate": GDPGrowthRateLoader(),
            "econ_news": EconNewsDataLoader(date_downloaded="2023-08-01"),
        },
        config=config,
    )
    data = data_loader.load_data()

    # set up for backtesting primary models
    backtest_dataset = TimeseriesBacktestDataset(
        dates=data['date'].to_list(),
        y_data=data['Close'].to_list(),
        window=BacktestWindow(
            n_obs=len(data),
            train_window_len=52,
            max_test_window_length=1,
        ),
        regressor_data={
            "interest_rate_delta": data['interest_rate_delta'].to_list(),
            "inflation_rate_delta": data["inflation_rate_delta"].to_list(),
            "gdp_growth_rate_delta": data['gdp_growth_rate_delta'].to_list(),
            "sentiment_score": data['sentiment_score'].to_list(),
        },
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
        LSTMForecaster(5, 50, 1, 3),
    ]
    primary_model_results = ForecastResults(backtest_dataset)

    # backtest primary models
    primary_model_backtester = Backtester(
        backtest_dataset=backtest_dataset,
        models=models,
        results=primary_model_results,
        preprocessor=preprocessor,
        mode="primary_model",
    )
    primary_model_backtester.run()

    # set up for backtesting model selection
    preprocessor = Preprocessor({})
    models = [
        AdaptiveHedge(alpha=0.9, multiplier=1)
    ]
    forecasts_dataset = primary_model_backtester.results.get_forecasts_dataset()
    model_selection_backtest_dataset = TimeseriesBacktestDataset(
        dates=forecasts_dataset['date'],
        y_data=forecasts_dataset['y_test'],
        window=BacktestWindow(
            n_obs=len(forecasts_dataset['date']),
            train_window_len=10,
            max_test_window_length=1,
        ),
        regressor_data={
            key: val
            for key, val in forecasts_dataset.items()
            if key not in ['date', 'y_test']
        },
    )

    # backtest model selection models
    model_selection_results = ForecastResults(model_selection_backtest_dataset)
    model_selection_backtester = Backtester(
        backtest_dataset=model_selection_backtest_dataset,
        models=models,
        results=model_selection_results,
        preprocessor=preprocessor,
        mode="model_selection",
    )
    model_selection_backtester.run()

    # analyse model performance
    model_selection_date_range = [
        forecast_dataset.dates_test[0] for window_index, forecast_dataset in
        model_selection_backtester.results.forecast_datasets.items()
    ]
    model_selection_window_range = [
        window_index for window_index, forecast_dataset in
        model_selection_backtester.results.forecast_datasets.items() if
        min(model_selection_date_range) <= forecast_dataset.dates_test[0] >=
        max(model_selection_date_range)
    ]
    primary_model_backtester.results.calculate_mean_model_error(
        model_selection_window_range
    )
    model_selection_backtester.results.calculate_mean_model_error()


if __name__ == '__main__':
    config = ConfigParser()
    config_file_path = sys.argv[1]
    config.read(config_file_path)
    main(config)

