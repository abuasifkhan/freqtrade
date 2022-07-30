# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.persistence import Trade
import datetime
from coral_trend import CoralTrend

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class CoralTrendMixHyperOpt(IStrategy):
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
        "120": 0.0030,
        "60": 0.001,
        "30": 0.002,
        "0": 0.004
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.5

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Hyperoptable parameters
    # buy_adx_threshold = DecimalParameter(10, 40, decimals=2, default=20)
    # buy_adx_enabled = BooleanParameter(default=False)

    # use_macd = CategoricalParameter([True, False], default=False)
    # buy_macd_threshold = DecimalParameter(-5, 2, decimals=2, default=0)
    # sell_macd_threshold = DecimalParameter(-2, 5, decimals=2, default=0)

    fast_sm_value = 14 # CategoricalParameter([7, 10, 14, 21], default=14)
    fast_cd_value = 0.8 # DecimalParameter(0.2, 1.0, decimals=1, default=0.8)
    medium_sm_value = 50 # CategoricalParameter([25, 30, 35, 40, 50], default=50)
    medium_cd_value = 0.4 # DecimalParameter(0.2, 1.0, decimals=1, default=0.4)
    # slow_sm_value = CategoricalParameter([200, 300, 400, 500, 600], default=300)
    # slow_cd_value = DecimalParameter(0.2, 1.0, decimals=1, default=0.8)

    fast_period_value = 6 #CategoricalParameter([6, 12, 18, 24, 30, 36, 42, 48], default=6)
    # slow_period_value = CategoricalParameter([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500], default=50)
    signal_value = 9 #CategoricalParameter([9, 18, 27, 36, 45], default=9)
    
    buy_trigger = CategoricalParameter(["medium_bfr_ema_cross", "medium_bfr_color_change"], default="medium_bfr_ema_cross", space="buy")

    # dataframe['fast_sm'] = fast_sm
    # dataframe['fast_cd'] = fast_cd
    # dataframe['slow_sm'] = slow_sm
    # dataframe['slow_cd'] = slow_cd
    # dataframe['bfr_fast'] = coral_trend(dataframe, fast_sm, fast_cd)
    # dataframe['bfr_slow'] = coral_trend(dataframe, slow_sm, slow_cd)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 800

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

    # use_custom_stoploss = False

    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
        # if current_profit < 0.01:
        #     return -1 # return a value bigger than the inital stoploss to keep using the inital stoploss

        # # After reaching the desired offset, allow the stoploss to trail by half the profit
        # desired_stoploss = current_profit / 2 

        # # Use a minimum of 1.5% and a maximum of 3%
        # return max(min(desired_stoploss, 0.3), 0.15)

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
        
        # macd = ta.MACD(dataframe, fastperiod=self.fast_period_value, slowperiod=self.slow_period_value, signalperiod=self.signal_value)
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
        afstep = 0.03
        aflimit = 0.03
        epstep = 0.03
        eplimit = 0.3
        dataframe['acceleration'] = acceleration
        dataframe['afstep'] = maximum
        dataframe['aflimit'] = afstep
        dataframe['epstep'] = epstep
        dataframe['eplimit'] = eplimit
        dataframe['sar'] = ta.SAR(dataframe, acceleration=acceleration, maximum=maximum,
                                  afstep=afstep, aflimit=aflimit, epstep=epstep, eplimit=eplimit)

        # #############################################################################
        # ###################### Coral Trend Indicator ################################
        dataframe['fast_sm'] = self.fast_sm_value
        dataframe['fast_cd'] = self.fast_cd_value
        dataframe['medium_sm'] = self.medium_sm_value
        dataframe['medium_cd'] = self.medium_cd_value
        # dataframe['slow_sm'] = self.slow_sm_value
        # dataframe['slow_cd'] = self.slow_cd_value
        dataframe['bfr_fast'] = coral_trend(dataframe, self.fast_sm_value, self.fast_cd_value)
        dataframe['bfr_medium'] = coral_trend(dataframe, self.medium_sm_value, self.medium_cd_value)
        # dataframe['bfr_slow'] = coral_trend(dataframe, self.slow_sm_value, self.slow_cd_value)
        # #############################################################################
        # ###################### End Coral Trend Indicator ################################

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        condition = []

        # GUARDS AND TRENDS
        condition.append(is_bullish_trend(dataframe))
        # TRIGGERS
        if self.buy_trigger == "medium_bfr_ema_cross":
            condition.append(
                qtpylib.crossed_above(dataframe['ema3'], dataframe['bfr_medium'])
            )
        elif self.buy_trigger == "medium_bfr_color_change":
            condition.append(
                red_to_green(dataframe['bfr_medium'])
            )

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # GUARDS AND TRENDS
        condition = []
        # condition.append(is_bearish_trend(dataframe))
        # if self.buy_trigger == "medium_bfr_ema_cross":
        #     condition.append(
        #         qtpylib.crossed_below(dataframe['ema3'], dataframe['bfr_medium'])
        #     )
        # elif self.buy_trigger == "medium_bfr_color_change":
        #     condition.append(
        #         green_to_red(dataframe['bfr_medium'])
            # )
            
        return dataframe





def is_bullish_trend(dataframe) -> bool:
    return is_green(dataframe['bfr_fast'])

def should_buy(dataframe) -> bool:
    return is_bullish_trend(dataframe) & qtpylib.crossed_above(dataframe['ema3'], dataframe['bfr_fast'])

def is_bearish_trend(dataframe) -> bool:
    return is_red(dataframe['bfr_fast'])

def should_sell(dataframe) -> bool:
    return is_bearish_trend(dataframe) & qtpylib.crossed_below(dataframe['ema3'], dataframe['bfr_fast'])

def is_green(dataframe_1d) -> bool:
    return np.greater(dataframe_1d, dataframe_1d.shift(1))

def is_red(dataframe_1d) -> bool:
    return np.less(dataframe_1d, dataframe_1d.shift(1))

def green_to_red(dataframe_1d) -> bool:
    return is_green(dataframe_1d) & is_red(dataframe_1d.shift(1))

def red_to_green(dataframe_1d) -> bool:
    return is_red(dataframe_1d) & is_green(dataframe_1d.shift(1))