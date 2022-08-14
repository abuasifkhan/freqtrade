# -----------------------------------------
# Using new strategy
# 1. Have a should_use booleanparameter in the strategy class
# 2. Get a parameter for config parameters
# 3. Define the index name inside init_index_names()
# 4. Define in Populate indicator method
# 5. Define in entry/exit guards and triggers
# -----------------------------------------
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.persistence import Trade
from functools import reduce
import datetime
# from coral_trend import *
from technical.indicators.indicators import *

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class SSLChannelStrategy(IStrategy):
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

    # Buy hyperspace params:
    buy_params = {
        "buy_large_ssl_length": 40,
        "buy_small_ssl_length": 30,
        "current_profit": 0.03,
        "maximum_stoploss": 0.05,
        "minimum_stoploss": 0.05,
        "shouldIgnoreRoi": True,
        "shouldUseStopLoss": True,
        "should_exit_profit_only": True,
        "should_use_exit_signal": True,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_ssl_length": 25,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.328,
        "100": 0.08,
        "215": 0.044,
        "435": 0
    }

    # Stoploss:
    stoploss = -0.16

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # MY INDICATORS
    # END OF MY INDICATORS

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

    shouldIgnoreRoi = buy_params['shouldIgnoreRoi']
    shouldUseStopLoss = buy_params['shouldUseStopLoss']

    buy_small_ssl_length = buy_params['buy_small_ssl_length']
    buy_large_ssl_length = buy_params['buy_large_ssl_length']
    sell_ssl_length = sell_params['sell_ssl_length']
    ssl_channel_down_index_pattern = 'ssl_channel_down_{0}'
    ssl_channel_up_index_pattern = 'ssl_channel_up_{0}'
    buy_small_ssl_channel_down_index_name = ''
    buy_small_ssl_channel_up_index_name = ''
    buy_large_ssl_channel_down_index_name = ''
    buy_large_ssl_channel_up_index_name = ''
    sell_ssl_channel_down_index_name = ''
    sell_ssl_channel_up_index_name = ''

    buy_trigger = "ssl_channel_buy"
    sell_trigger = "ssl_channel_sell"

    # These values can be overridden in the config.
    should_use_exit_signal = buy_params['should_use_exit_signal']
    should_exit_profit_only = buy_params['should_exit_profit_only']
    use_exit_signal = should_use_exit_signal
    exit_profit_only = should_exit_profit_only
    ignore_roi_if_entry_signal = shouldIgnoreRoi

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    use_custom_stoploss = shouldUseStopLoss
    current_profit = buy_params['current_profit']
    minimum_stoploss = buy_params['minimum_stoploss']
    maximum_stoploss = buy_params['maximum_stoploss']

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        if current_profit < self.current_profit:
            return -1 # return a value bigger than the inital stoploss to keep using the inital stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit / 2 

        # Use a minimum of 1.5% and a maximum of 3%
        return max(min(desired_stoploss, self.minimum_stoploss), self.maximum_stoploss)

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

        # populate SSL Channel
        sslDown, sslUp = SSLChannels(dataframe, self.buy_small_ssl_length)
        dataframe[self.ssl_channel_down_index_pattern.format(self.buy_small_ssl_length)] = sslDown
        dataframe[self.ssl_channel_up_index_pattern.format(self.buy_small_ssl_length)] = sslUp
        
        # if self.ssl_channel_down_index_pattern.format(ssl) not in dataframe.columns:
        sslDown, sslUp = SSLChannels(dataframe, self.buy_large_ssl_length)
        dataframe[self.ssl_channel_down_index_pattern.format(self.buy_large_ssl_length)] = sslDown    
        dataframe[self.ssl_channel_up_index_pattern.format(self.buy_large_ssl_length)] = sslUp
        
        # if self.ssl_channel_down_index_pattern.format(ssl) not in dataframe.columns:
        sslDown, sslUp = SSLChannels(dataframe, self.sell_ssl_length)
        dataframe[self.ssl_channel_down_index_pattern.format(self.sell_ssl_length)] = sslDown
        dataframe[self.ssl_channel_up_index_pattern.format(self.sell_ssl_length)] = sslUp

        dataframe.fillna(0, inplace=True)

        return dataframe

    def init_index_names(self):
        self.buy_small_ssl_channel_down_index_name = self.ssl_channel_down_index_pattern.format(self.buy_small_ssl_length)
        self.buy_small_ssl_channel_up_index_name = self.ssl_channel_up_index_pattern.format(self.buy_small_ssl_length)
        self.buy_large_ssl_channel_down_index_name = self.ssl_channel_down_index_pattern.format(self.buy_large_ssl_length)
        self.buy_large_ssl_channel_up_index_name = self.ssl_channel_up_index_pattern.format(self.buy_large_ssl_length)
        self.sell_ssl_channel_down_index_name = self.ssl_channel_down_index_pattern.format(self.sell_ssl_length)
        self.sell_ssl_channel_up_index_name = self.ssl_channel_up_index_pattern.format(self.sell_ssl_length)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.init_index_names()

        long_condition = self.populate_long_entry_guards(dataframe)
        long_condition = self.populate_long_trigger(dataframe, long_condition=long_condition)

        if long_condition:
                dataframe.loc[
                    reduce(lambda x, y: x & y, long_condition),
                    'enter_long'] = 1
        
        short_condition = self.populate_short_entry_guards(dataframe)
        short_condition = self.populate_short_trigger(dataframe, short_condition=short_condition)
        
        if short_condition:
                dataframe.loc[
                    reduce(lambda x, y: x & y, short_condition),
                    'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.init_index_names()

        long_exit = self.populate_long_exit_guards(dataframe)
        long_exit = self.populate_long_exit_trigger(dataframe, long_exit = long_exit)

        if long_exit:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_exit),
                'exit_long'] = 1
        
        short_exit = self.populate_short_exit_guards(dataframe)
        short_exit = self.populate_short_exit_trigger(dataframe, exit_short = short_exit)

        if short_exit:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_exit),
                'exit_short'] = 1
            
        return dataframe

    def populate_long_entry_guards(self, dataframe: DataFrame) -> DataFrame:
        long_condition = []

        long_condition.append(dataframe[self.buy_small_ssl_channel_down_index_name] < dataframe[self.buy_small_ssl_channel_up_index_name])
        
        return long_condition
    
    def populate_long_exit_guards(self, dataframe: DataFrame) -> DataFrame:
        short_exit = []
        
        return short_exit
    
    def populate_short_entry_guards(self, dataframe: DataFrame) -> DataFrame:
        short_condition = []

        # GUARDS AND TRENDS
        short_condition.append(dataframe[self.buy_small_ssl_channel_down_index_name] > dataframe[self.buy_small_ssl_channel_up_index_name])
        
        return short_condition
    
    def populate_short_exit_guards(self, dataframe: DataFrame) -> DataFrame:
        short_exit = []

        # GUARDS AND TRENDS
        
        return short_exit

    def populate_long_trigger(self, dataframe: DataFrame, long_condition):
        if self.buy_trigger == 'ssl_channel_buy':
            long_condition.append(self.ssl_cross_above(dataframe, self.buy_small_ssl_channel_up_index_name, self.buy_small_ssl_channel_down_index_name) |
                                   self.ssl_cross_above(dataframe, self.buy_large_ssl_channel_down_index_name, self.buy_large_ssl_channel_up_index_name))

        return long_condition
    
    def populate_long_exit_trigger(self, dataframe: DataFrame, long_exit):
        long_exit.append(self.ssl_cross_below(dataframe, self.buy_small_ssl_channel_up_index_name, self.buy_small_ssl_channel_down_index_name) |
                                   self.ssl_cross_below(dataframe, self.buy_large_ssl_channel_down_index_name, self.buy_large_ssl_channel_up_index_name))
        return long_exit
    
    def populate_short_trigger(self, dataframe: DataFrame, short_condition):
        if self.buy_trigger == 'ssl_channel_buy':
            short_condition.append(self.ssl_cross_below(dataframe, self.buy_small_ssl_channel_up_index_name, self.buy_small_ssl_channel_down_index_name) |
                                   self.ssl_cross_below(dataframe, self.buy_large_ssl_channel_down_index_name, self.buy_large_ssl_channel_up_index_name))
        return short_condition
    
    def populate_short_exit_trigger(self, dataframe: DataFrame, exit_short):
        exit_short.append(self.ssl_cross_above(dataframe, self.buy_small_ssl_channel_up_index_name, self.buy_small_ssl_channel_down_index_name) |
                                   self.ssl_cross_above(dataframe, self.buy_large_ssl_channel_down_index_name, self.buy_large_ssl_channel_up_index_name))
        
        return exit_short
    
    def ssl_cross_above(self, dataframe: DataFrame, up_index_name: str, down_index_name: str) -> bool:
        return qtpylib.crossed_above(dataframe[up_index_name], dataframe[down_index_name])
    
    def ssl_cross_below(self, dataframe: DataFrame, up_index_name: str, down_index_name: str) -> bool:
        return qtpylib.crossed_below(dataframe[up_index_name], dataframe[down_index_name])