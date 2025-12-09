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
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib
from scipy.signal import argrelextrema

class DivergencyStrategy(IStrategy):
    """
    Divergency Strategy
    - Detects Bullish Divergence (Price Lower Low, RSI Higher Low) -> Long
    - Detects Bearish Divergence (Price Higher High, RSI Lower High) -> Short
    - Timeframe: 15m
    - Conservative settings for high win rate
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # Conservative quick profit taking
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    # Optimal stoploss designed for the strategy.
    # Tight stoploss for conservation
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = "15m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    rsi_period = IntParameter(10, 25, default=14, space="buy")
    
    # Divergence Lookback (How far back to check for peaks/troughs)
    lookback_range = IntParameter(5, 50, default=20, space="buy")

    # Order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        
        # 1. RSI Calculation
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        
        # 2. Peak/Trough Detection for Divergence
        # We need to identify local minima (troughs) and maxima (peaks)
        # Using a rolling window approach or argrelextrema (more expensive but accurate)
        # For efficiency in backtesting/live, we can use a simpler rolling min/max check
        
        # Determine peaks and troughs in Price and RSI
        # Order 3 means we check 3 candles before and after for local extrema
        # Note: This looks into the future for historical data if not handled carefully.
        # However, for live signals, we only know a peak occurred AFTER it happened (lag).
        # We will use a custom simple detection:
        # A low is a low if it's lower than N previous candles and N next candles? 
        # In real-time, we can only know it's a low if current candle is higher than previous low candle.
        
        # Simplified Divergence Logic for Real-time/Backtest Safety:
        # 1. Identify if candle[i-1] was a local Low/High (Pivot)
        # 2. If it was, store it.
        # 3. Compare current Price/RSI with stored Pivot.
        
        # We will iterate to find divergences (Vectorized approach is hard for complex divergence)
        # But we can approximate.
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        
        # We will implement the divergence check logic here or in a helper method
        # Since vectorizing divergence is tricky, we can use a custom apply or loop for signal generation
        # BUT, for performance, we should try vectorization.
        
        # Let's try a robust approach:
        # 1. Find Pivot Lows/Highs in Price and RSI (e.g., lowest in last 5 candles)
        # 2. Compare current Low/High with Previous Low/High
        
        n = 5 # Pivot window
        
        # Calculate Rolling Min/Max to find pivots
        # shift(1) to avoid lookahead bias - we compare current close with past window
        
        # --- Bullish Divergence (Long) ---
        # Condition:
        # Current Price Low < Previous Price Low Pivot
        # Current RSI Low > Previous RSI Low Pivot
        
        # Step 1: Define what is a "Pivot Low"
        # A candle is a pivot low if its low is the minimum of the window centered on it? No, that's lookahead.
        # Real-time pivot: low[i-2] < low[i-3] ... and low[i-2] < low[i-1] ...
        # Let's use a simpler heuristic:
        # Price made a new low in the last X candles, but RSI did NOT make a new low.
        
        lookback = self.lookback_range.value
        
        # Price lowest in lookback
        dataframe['price_min'] = dataframe['low'].rolling(window=lookback).min()
        # RSI lowest in lookback
        dataframe['rsi_min'] = dataframe['rsi'].rolling(window=lookback).min()
        
        # Bullish Divergence Signal:
        # 1. Current Price is near the lookback Low (New Low formed)
        # 2. BUT Current RSI is significantly HIGHER than the RSI lookback Low
        
        # Refined Logic:
        # Find index of Price Low in window
        # Find index of RSI Low in window
        # If Price Low Index is Recent (e.g. current candle), AND RSI Low Index was Old (earlier in window)
        # That means Price kept dropping (new low now), but RSI bottomed out earlier (higher low now).
        
        # This logic is hard to fully vectorise perfectly without custom functions.
        # Let's use a standard "New Low in Price, Higher Low in RSI" logic.
        
        dataframe.loc[
            (
                # 1. Price is making a local low (e.g. lowest in last 10 candles)
                (dataframe['low'] < dataframe['low'].shift(1)) &
                (dataframe['low'] <= dataframe['low'].rolling(window=10).min()) &
                
                # 2. RSI is NOT making a new low (RSI > RSI Lowest in last 20 candles)
                (dataframe['rsi'] > dataframe['rsi'].rolling(window=lookback).min() + 5) & # Buffer
                
                # 3. RSI is Oversold (Confirmation)
                (dataframe['rsi'] < 40) & 
                
                # 4. Volume check
                (dataframe['volume'] > 0)
            ),
            "enter_long",
        ] = 1
        
        # --- Bearish Divergence (Short) ---
        # Condition:
        # Price made a new High, RSI did not.
        
        dataframe.loc[
            (
                # 1. Price is making a local high
                (dataframe['high'] > dataframe['high'].shift(1)) &
                (dataframe['high'] >= dataframe['high'].rolling(window=10).max()) &
                
                # 2. RSI is NOT making a new high (RSI < RSI Highest in last 20 candles)
                (dataframe['rsi'] < dataframe['rsi'].rolling(window=lookback).max() - 5) &
                
                # 3. RSI is Overbought
                (dataframe['rsi'] > 60) &
                
                (dataframe['volume'] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        # Exit Long: RSI Overbought or Trend Reversal
        dataframe.loc[
            (
                (dataframe['rsi'] > 70)
            ),
            "exit_long",
        ] = 1

        # Exit Short: RSI Oversold
        dataframe.loc[
            (
                (dataframe['rsi'] < 30)
            ),
            "exit_short",
        ] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Default leverage to 5x for conservative play.
        """
        return 5.0
