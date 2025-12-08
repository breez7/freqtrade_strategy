# pragma pylint: disable=missing-docstring, invalid-name, pointless-string_statement
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

class DCAStrategy(IStrategy):
    """
    DCA Strategy
    - Buy $300 every day at 9:00 AM.
    - If current price < 200 SMA, buy $600.
    - No selling.
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # Set to very high values to effectively disable ROI selling
    minimal_roi = {
        "0": 100.0  # Require 10000% profit to sell
    }

    # Optimal stoploss designed for the strategy.
    # Set to -1 to effectively disable stoploss (100% drop required)
    stoploss = -1.0

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    # Using 1d to buy at daily close (start of next day)
    timeframe = "1d"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Enable position adjustment for DCA
    position_adjustment_enable = True
    max_entry_position_adjustment = -1

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        # 200-period Simple Moving Average (SMA)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        # Buy daily. Since timeframe is 1d, this will signal on every new candle.
        dataframe.loc[
            (dataframe['volume'] > 0),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        # No exit signal logic as per requirement
        return dataframe

    def _get_dca_amount(self, pair: str, current_time: datetime, current_rate: float) -> float:
        """
        Calculate DCA amount based on 200 SMA.
        """
        dca_amount = 300.0
        
        if self.dp:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                # Get the last closed candle before current_time for backtesting safety
                # In live mode, this also ensures we use the confirmed previous close/indicators
                past_candles = dataframe.loc[dataframe['date'] < current_time]
                
                if not past_candles.empty:
                    last_candle = past_candles.iloc[-1]
                    
                    # Check if SMA 200 exists and compare with current price
                    if 'sma_200' in last_candle and not pd.isna(last_candle['sma_200']):
                        if current_rate < last_candle['sma_200']:
                            dca_amount = 600.0
        return dca_amount

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              **kwargs) -> Optional[float]:
        """
        Adjust trade position (DCA)
        """
        # Check if we have already bought today
        if trade.orders:
            # Get the last filled buy order
            buy_orders = [o for o in trade.orders if o.ft_order_side == 'buy' and o.status == 'closed']
            if buy_orders:
                last_buy_order = buy_orders[-1]
                # Ensure we don't buy again if we already bought on this date
                # current_time is the simulation time (candle open time in backtest)
                if last_buy_order.order_date.date() == current_time.date():
                    return None

        return self._get_dca_amount(trade.pair, current_time, current_rate)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        
        return self._get_dca_amount(pair, current_time, current_rate)
