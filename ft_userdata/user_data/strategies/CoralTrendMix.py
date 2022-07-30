# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.persistence import Trade
import datetime
from coral_trend import get_coral_trend

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class CoralTrendMix(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.001,
        "45": 0.002,
        "30": 0.003,
        "15": 0.004,
        "0": 0.01
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.20

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.10
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)

    # dataframe['fast_sm'] = fast_sm
    # dataframe['fast_cd'] = fast_cd
    # dataframe['slow_sm'] = slow_sm
    # dataframe['slow_cd'] = slow_cd
    # dataframe['bfr_fast'] = coral_trend(dataframe, fast_sm, fast_cd)
    # dataframe['bfr_slow'] = coral_trend(dataframe, slow_sm, slow_cd)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    use_custom_stoploss = False

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        if current_profit < 0.01:
            return -1 # return a value bigger than the inital stoploss to keep using the inital stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit / 2 

        # Use a minimum of 1.5% and a maximum of 3%
        return max(min(desired_stoploss, 0.30), 0.15)

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        # MACD
        # macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']

        # # EMA - Exponential Moving Average
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4.0
        dataframe['ema3'] = ta.EMA(dataframe['ohlc4'], timeperiod=3)
        dataframe['ema21'] = ta.EMA(dataframe['ohlc4'], timeperiod=21)

        # Parabolic SAR
        acceleration = 0.08
        maximum = 0.2
        afstep=0.03
        aflimit = 0.03
        epstep = 0.03
        eplimit = 0.3
        dataframe['acceleration'] = acceleration
        dataframe['afstep'] = maximum
        dataframe['aflimit'] = afstep
        dataframe['epstep'] = epstep
        dataframe['eplimit'] = eplimit
        dataframe['sar'] = ta.SAR(dataframe, acceleration=acceleration, maximum=maximum, afstep=afstep, aflimit=aflimit, epstep=epstep, eplimit=eplimit)

        # #############################################################################
        # ###################### Coral Trend Indicator ################################
        fast_sm = 14
        fast_cd = 0.8
        medium_sm = 30
        medium_cd = 0.8
        slow_sm = 100
        slow_cd = 0.8
        dataframe['fast_sm'] = fast_sm
        dataframe['fast_cd'] = fast_cd
        dataframe['medium_sm'] = medium_sm
        dataframe['medium_cd'] = medium_cd
        dataframe['slow_sm'] = slow_sm
        dataframe['slow_cd'] = slow_cd
        dataframe['bfr_fast'] = get_coral_trend(dataframe, fast_sm, fast_cd)
        dataframe['bfr_medium'] = get_coral_trend(dataframe, medium_sm, medium_cd)
        dataframe['bfr_slow'] = get_coral_trend(dataframe, slow_sm, slow_cd)
        dataframe['fast_medium'] = dataframe['bfr_fast'] - dataframe['bfr_medium']
        dataframe['medium_slow'] = dataframe['bfr_medium'] - dataframe['bfr_slow']

        # dataframe['bfr_fast_direction'] = np.greater(dataframe['bfr_fast'], dataframe['bfr_fast'].shift(1))

        # dataframe['bfr_medium_direction'] = np.greater(dataframe['bfr_medium'], dataframe['bfr_medium'].shift(1))

        # dataframe['bfr_slow_direction'] = np.greater(dataframe['bfr_slow'], dataframe['bfr_slow'].shift(1))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                should_buy(dataframe)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                should_sell(dataframe)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe['bfr_medium'], dataframe['ema3'])) 
        #     ),
        #     'exit_long'] = 1

        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe['ema3'], dataframe['bfr_medium'])) 
        #     ),
        #     'exit_short'] = 1

        return dataframe
    
def coral_trend(dataframe: DataFrame, sm: int, cd: int) -> DataFrame:
    di = (sm - 1.0) / 2.0 + 1.0
    c1 = 2.0 / (di + 1.0)
    c2 = 1.0 - c1
    c3 = 3.0 * (cd * cd + cd * cd * cd)
    c4 = -3.0 * (2.0 * cd * cd + cd + cd * cd * cd)
    c5 = 3.0 * cd + 1.0 + cd * cd * cd + 3.0 * cd * cd

    dataframe['bfr'] = 0.0

    for index in range(1,7):
        dataframe['i'+ str(index)] = 0.0

    for index, row in dataframe.iterrows():
        if index == 0:
            row ['i1'] = c1*row['close']
            row['i2'] = c1*row ['i1']
            row['i3'] = c1*row['i2']
            row['i4'] = c1*row['i3']
            row['i5'] = c1*row['i4']
            row['i6'] = c1*row['i5']
        else:
            prevRow = dataframe.loc[index-1]
            row['i1'] = c1*row['close'] + c2*prevRow['i1']
            row['i2'] = c1*row['i1'] + c2*prevRow['i2']
            row['i3'] = c1*row['i2'] + c2*prevRow['i3']
            row['i4'] = c1*row['i3'] + c2*prevRow['i4']
            row['i5'] = c1*row['i4'] + c2*prevRow['i5']
            row['i6'] = c1*row['i5'] + c2*prevRow['i6']

        dataframe.loc[index] = row
        dataframe.loc[index, 'bfr'] = -cd*cd*cd*dataframe.loc[index,'i6'] + c3*(dataframe.loc[index,'i5']) + c4*(dataframe.loc[index,'i4']) + c5*(dataframe.loc[index,'i3'])
        
    # print (dataframe[['bfr', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6']])
    return dataframe['bfr']

def is_bullish_trend(dataframe) -> bool:
    return is_green(dataframe['bfr_fast'])

def should_buy(dataframe) -> bool:
    return is_bullish_trend(dataframe) & qtpylib.crossed_above(dataframe['ema3'], dataframe['bfr_medium'])

def is_bearish_trend(dataframe) -> bool:
    return is_red(dataframe['bfr_fast'])

def should_sell(dataframe) -> bool:
    return is_bearish_trend(dataframe) & qtpylib.crossed_below(dataframe['ema3'], dataframe['bfr_medium'])

def is_green(dataframe_1d) -> bool:
    return np.greater(dataframe_1d, dataframe_1d.shift(1))

def is_red(dataframe_1d) -> bool:
    return np.less(dataframe_1d, dataframe_1d.shift(1))

def green_to_red(dataframe_1d) -> bool:
    return is_green(dataframe_1d) & is_red(dataframe_1d.shift(1))

def red_to_green(dataframe_1d) -> bool:
    return is_red(dataframe_1d) & is_green(dataframe_1d.shift(1))