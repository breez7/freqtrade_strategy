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


class VolumeStrategy(IStrategy):
    """
    Volume Strategy based on Volume Profile concepts.
    Reference: https://www.tradingview.com/chart/XBTUSD.P/b8j5QPEi/
    
    Concepts mapped to indicators:
    - POC (Point of Control/HVP): Approximated by VWAP (Volume Weighted Average Price).
      High Volume Nodes are areas where price spends a lot of time with high volume.
    - VA (Value Area): Approximated by VWAP +/- 2 Standard Deviations.
    - LVP (Low Volume Nodes): Areas outside or at the edges of the Value Area.
    
    Strategies:
    1. LVP Entry (Reversal): Bounce off Value Area Low (VAL) or High (VAH).
    2. HVP Entry (Breakout): Crossover of POC with momentum.
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "120": 0.02, # 120분 경과 후: 2% 수익이면 청산
        "60": 0.05,  # 60분 경과 후: 5% 수익이면 청산
        "0": 0.10,   # 0분 경과(진입 직후): 10% 수익이면 청산
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.01

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    # Lookback for VWAP/Profile calculation
    profile_period = IntParameter(20, 200, default=50, space="buy")
    
    # RSI for confirmation
    buy_rsi = IntParameter(1, 50, default=30, space="buy")
    sell_rsi = IntParameter(50, 100, default=70, space="sell")

    # Startup candle count
    startup_candle_count: int = 200

    plot_config = {
        "main_plot": {
            "vp_poc": {"color": "orange"},
            "vp_val": {"color": "green"},
            "vp_vah": {"color": "red"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "blue"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        
        # 1. Calculate Typical Price
        # dataframe['typical_price'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        # Use pandas-ta or manual calculation if not available.
        # Freqtrade uses qtpylib which has typical_price.
        typical_price = qtpylib.typical_price(dataframe)

        # 2. Volume Weighted Average Price (VWAP) - Rolling
        # Since 'rolling' weighted average is complex in pandas, we approximate or use a loop.
        # Actually, for "Fixed Range" volume profile behavior, a rolling VWAP is a good proxy.
        # Formula: sum(price * vol) / sum(vol) over window.
        
        period = self.profile_period.value
        
        pv = typical_price * dataframe['volume']
        dataframe['vp_poc'] = pv.rolling(window=period).sum() / dataframe['volume'].rolling(window=period).sum()
        
        # 3. Standard Deviation for Value Area (VA)
        # We use standard deviation of price to estimate the width of the volume profile.
        # VA covers 70% of volume typically. +/- 1 Stdev covers ~68%.
        # We'll use 2 Stdev to cover ~95% (Full profile width) or 1 for VA.
        # Let's use 2 to identify significant LVP edges.
        
        std = typical_price.rolling(window=period).std()
        dataframe['vp_val'] = dataframe['vp_poc'] - (2 * std) # Value Area Low
        dataframe['vp_vah'] = dataframe['vp_poc'] + (2 * std) # Value Area High
        
        # 4. RSI for confirmation
        dataframe['rsi'] = ta.RSI(dataframe)

        # 5. ADX to detect sideways market (ADX < 25 usually means weak trend / sideways)
        dataframe['adx'] = ta.ADX(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        
        # --- LVP Strategy (Reversal) ---
        # Long Entry: Price touches VAL (Support) and RSI is low (Oversold).
        # We look for price being near VAL (Low Volume Node / Support).
        # Added condition: ADX < 25 to ensure market is sideways/ranging.
        
        dataframe.loc[
            (
                # Market is sideways (ADX < 25)
                (dataframe['adx'] < 25) &
                # Price is near Value Area Low (e.g. within 1% or crossed it)
                (dataframe['close'] <= dataframe['vp_val']) &
                # Reversal confirmation: Price is starting to go up?
                # Or simply "Buy the dip" at LVP support
                (dataframe['rsi'] < self.buy_rsi.value) &
                (dataframe['volume'] > 0)
            ),
            "enter_long",
        ] = 1
        
        # Short Entry (LVP Reversal from Top): Price touches VAH (Resistance).
        dataframe.loc[
            (
                # Market is sideways (ADX < 25)
                (dataframe['adx'] < 25) &
                (dataframe['close'] >= dataframe['vp_vah']) &
                (dataframe['rsi'] > self.sell_rsi.value) &
                (dataframe['volume'] > 0)
            ),
            "enter_short",
        ] = 1

        # --- HVP Strategy (Breakout) ---
        # Breakout of POC often leads to strong trends.
        # If Price Crosses POC from below -> Long
        # If Price Crosses POC from above -> Short
        # This can be added as a secondary condition or separate signal.
        # For this sample, we combine them.
        
        # Example HVP Long Breakout:
        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe['close'], dataframe['vp_poc'])) &
        #         (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) # High Volume Breakout
        #     ),
        #     "enter_long"
        # ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        # Exit Long if RSI is high or price hits POC (Mean Reversion)
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi.value) |
                (dataframe['close'] >= dataframe['vp_poc'])
            ),
            "exit_long",
        ] = 1

        # Exit Short if RSI is low or price hits POC (Mean Reversion)
        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value) |
                (dataframe['close'] <= dataframe['vp_poc'])
            ),
            "exit_short",
        ] = 1

        return dataframe
