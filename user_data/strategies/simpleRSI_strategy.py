# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


# This class is a sample. Feel free to customize it.
class SimpleRSIStrategy(IStrategy):
    """
    This strategy enters a long position when RSI is below 20 and
    enters a short position when RSI is above 80.
    """

    INTERFACE_VERSION = 3

    can_short: bool = True

    minimal_roi = {
        "0": 0.20,
    }

    stoploss = -0.05

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return 5.0

    trailing_stop = False

    timeframe = "1m"

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters (optional, adjust as needed)
    entry_rsi_long = IntParameter(low=1, high=40, default=10, space="entry", optimize=True, load=True)
    entry_rsi_short = IntParameter(low=60, high=100, default=90, space="entry", optimize=True, load=True)
    exit_rsi_long = IntParameter(low=60, high=100, default=70, space="exit", optimize=True, load=True)
    exit_rsi_short = IntParameter(low=1, high=40, default=30, space="exit", optimize=True, load=True)


    startup_candle_count: int = 200

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14) # Default RSI period is 14
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Check if RSI < entry_rsi_long at least 10 times in the last 20 candles
        long_rsi_check = (dataframe["rsi"] < self.entry_rsi_long.value).rolling(window=20).sum() >= 10
        dataframe.loc[
            (
                long_rsi_check
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # Check if RSI > entry_rsi_short at least 10 times in the last 20 candles
        short_rsi_check = (dataframe["rsi"] > self.entry_rsi_short.value).rolling(window=20).sum() >= 10
        dataframe.loc[
            (
                short_rsi_check
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] > self.exit_rsi_long.value)
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                (dataframe["rsi"] < self.exit_rsi_short.value)
                & (dataframe["volume"] > 0)
            ),
            "exit_short",
        ] = 1

        return dataframe
