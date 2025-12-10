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


class MixedStrategy(IStrategy):
    """
    Mixed Strategy
    
    1. 진입 시점을 판단할 수 있는 판단 지표를 7가지 이상 가지고 있으며 4개의 지표가 만족하면 진입을 한다.
    2. exit 시점을 판단할 수 있는 판단 지표를 7가지 이상 가지고 있으며 4개의 지표가 만족하면 exit를 한다.
    3. 1과 2에서 4개의 지표가 만족해야한다고 했지만 이는 조절 가능하도록 해야한다. 
    4. 여기서 사용하는 판단 지표는 충분히 좋은 지표들을 사용해야한다. 간단한면서 강력한것도 있어야하고 복잡하면서 강력한것도 있어야 한다.
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 1.0,  # Let Trailing Stop handle exits
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = False
    
    # Custom stoploss
    use_custom_stoploss = True

    # Optimal timeframe for the strategy.
    timeframe = "15m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Hyperoptable parameters for voting thresholds
    # Entry Thresholds (Lower to enter more easily)
    buy_voting_threshold = IntParameter(low=1, high=8, default=1, space="buy", optimize=True, load=True)
    sell_voting_threshold = IntParameter(low=1, high=8, default=1, space="sell", optimize=True, load=True)

    # Exit Thresholds (Higher to exit later/let profit run)
    exit_long_voting_threshold = IntParameter(low=1, high=8, default=7, space="sell", optimize=True, load=True)
    exit_short_voting_threshold = IntParameter(low=1, high=8, default=7, space="buy", optimize=True, load=True)

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """

        # 1. RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # 2. MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # 3. Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # 4. EMA
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # 5. Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # 6. ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # 7. MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # 8. CCI
        dataframe['cci'] = ta.CCI(dataframe)

        # 9. Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # 10. Volume Moving Average (for conservative entry)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        # 11. Double Divergence Logic
        # Define pivot lookback window (2 means check 2 candles before and after)
        # Note: We use a lagged confirmation. A peak at t-2 is confirmed at t.
        
        # --- 10a. Identify Peaks (Highs) and Troughs (Lows) ---
        # Confirmed Peak at t-1: High[t-1] is higher than t-2 and t
        # We perform the check at current index `i`, looking back.
        
        # Pivot High (Bearish Div)
        # Check if candle at shift(1) is a local maximum
        dataframe['is_peak'] = (
            (dataframe['high'].shift(1) > dataframe['high'].shift(2)) &
            (dataframe['high'].shift(1) > dataframe['high'])
        )
        
        # Pivot Low (Bullish Div)
        # Check if candle at shift(1) is a local minimum
        dataframe['is_trough'] = (
            (dataframe['low'].shift(1) < dataframe['low'].shift(2)) &
            (dataframe['low'].shift(1) < dataframe['low'])
        )
        
        # --- 10b. Vectorized Double Divergence Check ---
        # We extract the peaks/troughs into a separate view to shift by "Events" rather than "Candles"
        
        # ---------------- Bearish Double Divergence ----------------
        # Filter only peaks
        peaks = dataframe[dataframe['is_peak']].copy()
        
        if not peaks.empty:
            # Shift to compare P3 (Current), P2 (Prev), P1 (PrevPrev)
            # We want Price P3 > P2 > P1
            # We want RSI  R3 < R2 < R1
            
            # Price at Peak (High of t-1)
            peaks['peak_price'] = dataframe['high'].shift(1)
            peaks['peak_rsi'] = dataframe['rsi'].shift(1)
            
            # Now shift by event (row in peaks dataframe)
            peaks['p2_price'] = peaks['peak_price'].shift(1)
            peaks['p1_price'] = peaks['peak_price'].shift(2)
            
            peaks['r2_rsi'] = peaks['peak_rsi'].shift(1)
            peaks['r1_rsi'] = peaks['peak_rsi'].shift(2)
            
            # Condition
            peaks['double_div_bearish'] = (
                (peaks['peak_price'] >= peaks['p2_price']) &
                (peaks['p2_price'] >= peaks['p1_price']) &
                (peaks['peak_rsi'] <= peaks['r2_rsi']) &
                (peaks['r2_rsi'] <= peaks['r1_rsi'])
            )
            
            # --- Regular Bearish Divergence (New) ---
            peaks['single_div_bearish'] = (
                (peaks['peak_price'] > peaks['p2_price']) &
                (peaks['peak_rsi'] < peaks['r2_rsi'])
            )
            
            # Map back to main dataframe
            # Initialize False
            dataframe['double_div_bearish'] = False
            dataframe['single_div_bearish'] = False
            # Update True values
            dataframe.loc[peaks.index, 'double_div_bearish'] = peaks['double_div_bearish']
            dataframe.loc[peaks.index, 'single_div_bearish'] = peaks['single_div_bearish']
            
        else:
            dataframe['double_div_bearish'] = False
            dataframe['single_div_bearish'] = False

        # ---------------- Bullish Double Divergence ----------------
        # Filter only troughs
        troughs = dataframe[dataframe['is_trough']].copy()
        
        if not troughs.empty:
            # Price at Trough (Low of t-1)
            troughs['trough_price'] = dataframe['low'].shift(1)
            troughs['trough_rsi'] = dataframe['rsi'].shift(1)
            
            # Shift by event
            troughs['p2_price'] = troughs['trough_price'].shift(1)
            troughs['p1_price'] = troughs['trough_price'].shift(2)
            
            troughs['r2_rsi'] = troughs['trough_rsi'].shift(1)
            troughs['r1_rsi'] = troughs['trough_rsi'].shift(2)
            
            # Condition: Price Lower Lows, RSI Higher Lows
            troughs['double_div_bullish'] = (
                (troughs['trough_price'] <= troughs['p2_price']) &
                (troughs['p2_price'] <= troughs['p1_price']) &
                (troughs['trough_rsi'] >= troughs['r2_rsi']) &
                (troughs['r2_rsi'] >= troughs['r1_rsi'])
            )
            
            # --- Regular Bullish Divergence (New) ---
            troughs['single_div_bullish'] = (
                (troughs['trough_price'] < troughs['p2_price']) &
                (troughs['trough_rsi'] > troughs['r2_rsi'])
            )
            
            # Map back
            dataframe['double_div_bullish'] = False
            dataframe['single_div_bullish'] = False
            dataframe.loc[troughs.index, 'double_div_bullish'] = troughs['double_div_bullish']
            dataframe.loc[troughs.index, 'single_div_bullish'] = troughs['single_div_bullish']
            
        else:
            dataframe['double_div_bullish'] = False
            dataframe['single_div_bullish'] = False

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        
        # --- Buy Votes (Bullish) ---
        dataframe['buy_vote'] = 0

        # Indicator 1: RSI < 30 (Oversold)
        dataframe.loc[dataframe['rsi'] < 30, 'buy_vote'] += 1

        # Indicator 2: MACD > Signal (Bullish Crossover/Trend)
        dataframe.loc[dataframe['macd'] > dataframe['macdsignal'], 'buy_vote'] += 1

        # Indicator 3: Close < BB Lower Band (Mean Reversion)
        dataframe.loc[dataframe['close'] < dataframe['bb_lowerband'], 'buy_vote'] += 1

        # Indicator 4: EMA9 > EMA21 (Golden Cross-ish)
        dataframe.loc[dataframe['ema9'] > dataframe['ema21'], 'buy_vote'] += 1

        # Indicator 5: Stochastic Fast K < 20 (Oversold)
        dataframe.loc[dataframe['fastk'] < 20, 'buy_vote'] += 1

        # Indicator 6: ADX > 25 (Strong Trend)
        dataframe.loc[dataframe['adx'] > 25, 'buy_vote'] += 1

        # Indicator 7: MFI < 20 (Oversold Volume)
        dataframe.loc[dataframe['mfi'] < 20, 'buy_vote'] += 1
        
        # Indicator 8: CCI < -100 (Oversold)
        dataframe.loc[dataframe['cci'] < -100, 'buy_vote'] += 1

        # --- Sell Votes (Bearish) - Moved from exit_trend to support enter_short ---
        dataframe['sell_vote'] = 0

        # Indicator 1: RSI > 70 (Overbought)
        dataframe.loc[dataframe['rsi'] > 70, 'sell_vote'] += 1

        # Indicator 2: MACD < Signal (Bearish Crossover/Trend)
        dataframe.loc[dataframe['macd'] < dataframe['macdsignal'], 'sell_vote'] += 1

        # Indicator 3: Close > BB Upper Band (Mean Reversion)
        dataframe.loc[dataframe['close'] > dataframe['bb_upperband'], 'sell_vote'] += 1

        # Indicator 4: EMA9 < EMA21 (Death Cross-ish)
        dataframe.loc[dataframe['ema9'] < dataframe['ema21'], 'sell_vote'] += 1

        # Indicator 5: Stochastic Fast K > 80 (Overbought)
        dataframe.loc[dataframe['fastk'] > 80, 'sell_vote'] += 1

        # Indicator 6: MFI > 80 (Overbought Volume)
        dataframe.loc[dataframe['mfi'] > 80, 'sell_vote'] += 1

        # Indicator 7: CCI > 100 (Overbought)
        dataframe.loc[dataframe['cci'] > 100, 'sell_vote'] += 1
        
        # Indicator 8: Parabolic SAR above price (Bearish trend)
        dataframe.loc[dataframe['sar'] > dataframe['close'], 'sell_vote'] += 1

        # --- Entry Logic ---
        # Requirement: Double Divergence is MANDATORY.
        # Requirement: Voting Threshold is MANDATORY.

        # Enter Long (Double Divergence - Strong)
        dataframe.loc[
            (
                (dataframe['double_div_bullish'] == True) &
                (dataframe['buy_vote'] >= self.buy_voting_threshold.value) &
                (dataframe['close'] > dataframe['ema200']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
            
        # Enter Short (Double Divergence - Strong)
        dataframe.loc[
            (
                (dataframe['double_div_bearish'] == True) &
                (dataframe['sell_vote'] >= self.sell_voting_threshold.value) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        
        # Note: 'sell_vote' and 'buy_vote' are calculated in populate_entry_trend.
        # We reuse them here to ensure consistency and avoid code duplication.
        
        # Exit Long if Sell Votes are high (Bearish) - Stronger signal needed to exit
        dataframe.loc[
            (
                (dataframe['sell_vote'] >= self.exit_long_voting_threshold.value) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        # Exit Short if Buy Votes are high (Bullish) - Stronger signal needed to exit
        dataframe.loc[
            (
                (dataframe['buy_vote'] >= self.exit_short_voting_threshold.value) &
                (dataframe['volume'] > 0)
            ),
            'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Default leverage to 10x.
        """
        return 10.0

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as a negative number).
        s.t. current_rate - (current_rate * stoploss_distance) = stoploss_price
        
        Freqtrade expects the return value to be the stoploss percentage (e.g., -0.05 for 5% below).
        However, when using custom_stoploss, we can return 1 to keep the previous stoploss, 
        or a specific new relative stoploss.
        """
        
        # Step-wise Trailing Stop
        if current_profit > 0.08:
            return -0.01  # Lock in profit (allow 1% drop from current price)
        if current_profit > 0.04:
            return -0.02  # Tighten stoploss (allow 2% drop from current price)
            
        return 1  # Return 1 to keep current stoploss
