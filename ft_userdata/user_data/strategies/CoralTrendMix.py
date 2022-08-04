# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.persistence import Trade
from coral_trend import *
import datetime
from technical.indicators.indicators import *
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
    # ROI table:
    minimal_roi = {
        "0": 0.294,
        "37": 0.058,
        "133": 0.028,
        "493": 0
    }

    atr_parameters = {
        "length": 16,
        "threshold": 0.989
    }

    # MY INDICATORS
    # SAR parameters
    sar_parameters = {
        "acceleration": 0.04,
        "maximum": 0.2,
        "afstep": 0.03,
        "aflimit": 0.03,
        "epstep": 0.03,
        "eplimit": 0.3,
    }

    # Coral Parameters
    fast_coral_trend_parameters = {
        "fast_sm": 14,
        "fast_cd": 0.4
    }

    medium_coral_trend_parameters = {
        "medium_sm": 100,
        "medium_cd": 0.4
    }

    slow_coral_trend_parameters = {
        "slow_sm": 300,
        "slow_cd": 0.444
    }

    pmax_parameters = {
        "period": 15, 
        "multiplier": 4, 
        "length": 15,
        "MAtype": 1
    }

    macd_parameters = {
        "fast_period": 6,
        "slow_period": 50,
        "signal_period": 9,
    }

    # END OF MY INDICATORS

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # Stoploss:
    stoploss = -0.171

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
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

    use_custom_stoploss = True

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


    # def is_uptrend(dataframe) -> bool:
    #     return is_green(dataframe['coral_fast']) # High winrate

    def is_uptrend(self, dataframe) -> bool:
        # return (dataframe['coral_medium'] < dataframe['ema3']) #& (dataframe['coral_fast'] < dataframe['ema3']) #&  (dataframe['PMAX'] == 'up')
        return (dataframe['atrP'] > self.atr_parameters['threshold']) & (dataframe['coral_medium'] < dataframe['ema3']) &  (dataframe['pmax'] == 'up')

    # def should_long(dataframe) -> bool:
    #     return is_uptrend(dataframe) & qtpylib.crossed_above(dataframe['ema3'], dataframe['coral_medium'])

    def should_long(self, dataframe) -> bool:
        return (self.is_uptrend(dataframe)) & (qtpylib.crossed_above(dataframe['ema3'], dataframe['coral_medium']))

    # def is_downtrend(dataframe) -> bool:
    #     return is_red(dataframe['coral_fast'])

    def is_downtrend(self, dataframe) -> bool:
        # return (dataframe['coral_medium'] > dataframe['low']) #(dataframe['PMAX'] == 'down') ##& (dataframe['coral_fast'] > dataframe['ema3']) #& 
        return (dataframe['atrP'] < self.atr_parameters['threshold']) & (dataframe['coral_medium'] > dataframe['ema3']) & (dataframe['pmax'] == 'down')

    def should_short(self, dataframe) -> bool:
        # return (self.is_downtrend(dataframe)) & (qtpylib.crossed_below(dataframe['ema3'], dataframe['coral_medium']))
        return (self.is_downtrend(dataframe)) & (qtpylib.crossed_below(dataframe['ema3'], dataframe['coral_medium']))

    # def should_short(dataframe) -> bool:
    #     return is_downtrend(dataframe) & qtpylib.crossed_below(dataframe['ema3'], dataframe['coral_medium'])

    def is_green(self, dataframe_1d) -> bool:
        return np.greater(dataframe_1d, dataframe_1d.shift(1))

    def is_red(self, dataframe_1d) -> bool:
        return np.less(dataframe_1d, dataframe_1d.shift(1))

    def green_from_red(self, dataframe_1d) -> bool:
        return self.is_red(dataframe_1d.shift(1)) & self.is_green(dataframe_1d)

    def red_from_green(self, dataframe_1d) -> bool:
        return self.is_green(dataframe_1d.shift(1)) & self.is_red(dataframe_1d)
    
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

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # ATR
        dataframe[f'atr'] = ta.ATR(dataframe, period=self.atr_parameters['length'])
        dataframe[f'atrP'] = dataframe[f'atr'] / dataframe['close'].fillna(1)

        # # EMA - Exponential Moving Average
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4.0
        dataframe['ema3'] = ta.EMA(dataframe['ohlc4'], timeperiod=3)
        dataframe['ema21'] = ta.EMA(dataframe['ohlc4'], timeperiod=21)

        # Parabolic SAR
        acceleration = self.sar_parameters["acceleration"]
        maximum = self.sar_parameters["maximum"]
        afstep = self.sar_parameters["afstep"]
        aflimit = self.sar_parameters["aflimit"]
        epstep = self.sar_parameters["epstep"]
        eplimit = self.sar_parameters["eplimit"]
        dataframe['acceleration'] = acceleration
        dataframe['afstep'] = maximum
        dataframe['aflimit'] = afstep
        dataframe['epstep'] = epstep
        dataframe['eplimit'] = eplimit
        dataframe['sar'] = ta.SAR(dataframe, acceleration=acceleration, maximum=maximum, afstep=afstep, aflimit=aflimit, epstep=epstep, eplimit=eplimit)

        #PMAX
        pmax_period = self.pmax_parameters["period"]
        pmax_multiplier = self.pmax_parameters["multiplier"]
        pmax_length = self.pmax_parameters["length"]
        pmax_MAtype = self.pmax_parameters["MAtype"]
        dataframe = PMAX(dataframe, period=pmax_period, multiplier=pmax_multiplier, length=pmax_length, MAtype=pmax_MAtype)
        pm = "pm_" + str(pmax_period) + "_" + str(pmax_multiplier) + "_" + str(pmax_length) + "_" + str(pmax_MAtype)
        pmx = "pmax_" + str(pmax_period) + "_" + str(pmax_multiplier) + "_" + str(pmax_length) + "_" + str(pmax_MAtype)
        dataframe['pmax'] = dataframe[pmx]
        print(dataframe[pmx])
        print(dataframe[pm])

        # #############################################################################
        # ###################### Coral Trend Indicator ################################
        fast_sm = self.fast_coral_trend_parameters["fast_sm"]
        fast_cd = self.fast_coral_trend_parameters["fast_cd"]
        medium_sm = self.medium_coral_trend_parameters["medium_sm"]
        medium_cd = self.medium_coral_trend_parameters["medium_cd"]
        slow_sm = self.slow_coral_trend_parameters["slow_sm"]
        slow_cd = self.slow_coral_trend_parameters["slow_cd"]
        dataframe['coral_fast'] = coral_trend(dataframe, fast_sm, fast_cd)
        dataframe['coral_medium'] = coral_trend(dataframe, medium_sm, medium_cd)
        dataframe['coral_slow'] = coral_trend(dataframe, slow_sm, slow_cd)

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
                self.should_long(dataframe)
            ),
            'enter_long'] = 1
        
        dataframe.loc[(
            self.should_short(dataframe)
        ), 'enter_short'] = 1

        # dataframe.loc[
        #     (
        #         self.should_short(dataframe)
        #     ),
        #     'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """

        return dataframe
