# -----------------------------------------
# Using new strategy
# 1. Have a should_use booleanparameter in the strategy class
# 2. Get a parameter for config parameters
# 3. Define the index name inside init_index_names()
# 4. Define in Populate indicator method
# 5. Define in entry/exit guards and triggers
# -----------------------------------------
from typing import Optional
from freqtrade.optimize.space.decimalspace import SKDecimal
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.persistence import Trade
from functools import reduce
import datetime
# from coral_trend import *
from technical.indicators.indicators import *
import custom_indicators as cta

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_open, stoploss_from_absolute)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class SSLChannelHyperOpt(IStrategy):
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
        "buy_coral_sm": 21,
        "buy_leverage": 10,
        "buy_small_ssl_length": 5,
        "shouldIgnoreRoi": False,
        "shouldUseStopLoss": True,
        "should_exit_profit_only": True,
        "should_use_exit_signal": True,
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_endtrend_respect_roi": True,
        "cexit_pullback": False,
        "cexit_pullback_amount": 0.021,
        "cexit_pullback_respect_roi": True,
        "cexit_roi_end": 0.003,
        "cexit_roi_start": 0.046,
        "cexit_roi_time": 794,
        "cexit_roi_type": "step",
        "cexit_trend_type": "rmi",
        "cstop_bail_how": "any",
        "cstop_bail_roc": -2.241,
        "cstop_bail_time": 899,
        "cstop_bail_time_trend": False,
        "cstop_loss_threshold": -0.021,
        "cstop_max_stoploss": -0.061,
        "maximum_stoploss": 0.005,
        "minimum_take_profit": 0.8,
        "profit_trigger": 0.01,
        "sell_ssl_length": 30,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.347,
        "69": 0.128,
        "184": 0.06,
        "270": 0
    }

    # Stoploss:
    stoploss = -0.334

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    # Optimal timeframe for the strategy.
    timeframe = '5m'

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

    shouldIgnoreRoi = BooleanParameter(default=buy_params['shouldIgnoreRoi'], space='buy')
    shouldUseStopLoss = BooleanParameter(default=buy_params['shouldUseStopLoss'], space='buy')

    # --------------------------------
    buy_small_ssl_length = CategoricalParameter([5, 5], default=buy_params['buy_small_ssl_length'], space='buy', optimize=True)
    sell_ssl_length = CategoricalParameter([30, 30], default=sell_params['sell_ssl_length'], space='sell')

    # Custom Sell Profit (formerly Dynamic ROI)
    cexit_roi_type = CategoricalParameter(['static', 'decay', 'step'], default=sell_params['cexit_roi_type'], space='sell', load=True,
                                          optimize=True)
    cexit_roi_time = IntParameter(720, 1440, default=sell_params['cexit_roi_time'], space='sell', load=True, optimize=True)
    cexit_roi_start = DecimalParameter(0.01, 0.05, default=sell_params['cexit_roi_start'], space='sell', load=True, optimize=True)
    cexit_roi_end = DecimalParameter(0.0, 0.01, default=sell_params['cexit_roi_end'], space='sell', load=True, optimize=True)
    cexit_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default=sell_params['cexit_trend_type'], space='sell',
                                            load=True, optimize=True)
    cexit_pullback = CategoricalParameter([True, False], default=sell_params['cexit_pullback'], space='sell', load=True, optimize=True)
    cexit_pullback_amount = DecimalParameter(0.005, 0.03, default=sell_params['cexit_pullback_amount'], space='sell', load=True, optimize=True)
    cexit_pullback_respect_roi = CategoricalParameter([True, False], default=sell_params['cexit_pullback_respect_roi'], space='sell', load=True,
                                                      optimize=True)
    cexit_endtrend_respect_roi = CategoricalParameter([True, False], default=sell_params['cexit_endtrend_respect_roi'], space='sell', load=True,
                                                      optimize=True)

    # Custom Stoploss
    cstop_loss_threshold = DecimalParameter(-0.05, -0.01, default=sell_params['cstop_loss_threshold'], space='sell', load=True, optimize=True)
    cstop_bail_how = CategoricalParameter(['roc', 'time', 'any', 'none'], default=sell_params['cstop_bail_how'], space='sell', load=True,
                                          optimize=True)
    cstop_bail_roc = DecimalParameter(-5.0, -1.0, default=-sell_params['cstop_bail_roc'], space='sell', load=True, optimize=True)
    cstop_bail_time = IntParameter(60, 1440, default=sell_params['cstop_bail_time'], space='sell', load=True, optimize=True)
    cstop_bail_time_trend = CategoricalParameter([True, False], default=sell_params['cstop_bail_time_trend'], space='sell', load=True, optimize=True)
    cstop_max_stoploss =  DecimalParameter(-0.30, -0.01, default=sell_params['cstop_max_stoploss'], space='sell', load=True, optimize=True)
    custom_trade_info = {}


    ssl_channel_down_index_pattern = 'ssl_channel_down_{0}'
    ssl_channel_up_index_pattern = 'ssl_channel_up_{0}'
    buy_small_ssl_channel_down_index_name = ''
    buy_small_ssl_channel_up_index_name = ''
    sell_ssl_channel_down_index_name = ''
    sell_ssl_channel_up_index_name = ''
    # --------------------------------

    # --------------------------------
    buy_coral_sm =  CategoricalParameter([21, 21], default=buy_params['buy_coral_sm'], space='buy') # 21
    buy_coral_cd = 0.9
    buy_coral_index_name = ''
    # --------------------------------

    buy_leverage = IntParameter(1, 20, default=1, space='buy')

    buy_trigger = "ssl_channel_buy"
    sell_trigger = "ssl_channel_sell"

    # These values can be overridden in the config.
    should_use_exit_signal = BooleanParameter(default=buy_params['should_use_exit_signal'], space='buy')
    should_exit_profit_only = BooleanParameter(default=buy_params['should_exit_profit_only'], space='buy')
    use_exit_signal = should_use_exit_signal.value
    exit_profit_only = should_exit_profit_only.value
    ignore_roi_if_entry_signal = shouldIgnoreRoi.value

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    use_custom_stoploss = shouldUseStopLoss.value
    profit_trigger = CategoricalParameter([0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], default=sell_params['profit_trigger'], space='sell')
    maximum_stoploss = CategoricalParameter([0.002, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], default=sell_params['maximum_stoploss'], space='sell')
    minimum_take_profit = CategoricalParameter([0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], default=sell_params['minimum_take_profit'], space='sell')

    # Define a custom stoploss space.
    def stoploss_space():
        return [SKDecimal(-0.02, -0.01, decimals=3, name='stoploss')]

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had-trend']

        if current_profit > self.profit_trigger.value:
            return self.minimum_take_profit.value
        elif current_profit < 0:
            return self.maximum_stoploss.value

        # limit stoploss
        if current_profit <  self.cstop_max_stoploss.value:
            return 0.01

        # Determine how we sell when we are in a loss
        if current_profit < self.cstop_loss_threshold.value:
            if self.cstop_bail_how.value == 'roc' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on rate of change
                if last_candle['sroc'] <= self.cstop_bail_roc.value:
                    return 0.01
            if self.cstop_bail_how.value == 'time' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on time, unless time_trend is true and there is a potential reversal
                if trade_dur > self.cstop_bail_time.value:
                    if self.cstop_bail_time_trend.value == True and in_trend == True:
                        return 1
                    else:
                        return 0.01
        return 1
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.cexit_pullback_amount.value))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.cexit_roi_type.value == 'static':
            min_roi = self.cexit_roi_start.value
        elif self.cexit_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.cexit_roi_start.value, self.cexit_roi_end.value, 0,
                                       self.cexit_roi_time.value, trade_dur)
        elif self.cexit_roi_type.value == 'step':
            if trade_dur < self.cexit_roi_time.value:
                min_roi = self.cexit_roi_start.value
            else:
                min_roi = self.cexit_roi_end.value

        # Determine if there is a trend
        if self.cexit_trend_type.value == 'rmi' or self.cexit_trend_type.value == 'any':
            if last_candle['rmi-up-trend'] == 1:
                in_trend = True
        if self.cexit_trend_type.value == 'ssl' or self.cexit_trend_type.value == 'any':
            if last_candle['ssl-dir'] == 'up':
                in_trend = True
        if self.cexit_trend_type.value == 'candle' or self.cexit_trend_type.value == 'any':
            if last_candle['candle-up-trend'] == 1:
                in_trend = True

        # Don't sell if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful sell message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a sell, maybe
            if self.cexit_pullback.value == True and (current_profit <= pullback_value):
                if self.cexit_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif self.cexit_pullback_respect_roi.value == False:
                    if current_profit > min_roi:
                        return 'intrend_pullback_roi'
                    else:
                        return 'intrend_pullback_noroi'
            # We are in a trend and pullback is disabled or has not happened or various criteria were not met, hold
            return None
        # If we are not in a trend, just use the roi value
        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had-trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_roi'
                elif self.cexit_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None

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
        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)
        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)
        dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()
        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)

        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}
            if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
                self.custom_trade_info[metadata['pair']]['had-trend'] = False

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown, 'up', 'down')
        dataframe['atr'] = qtpylib.atr(dataframe)
        # dataframe['rsi'] = cta.rsi(dataframe, length=7)

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        dataframe['mastreak'] = cta.mastreak(dataframe, period=4)
        
        # Use Coral + SAR + SSL + VWAP + Rolling VWAP to eliminate sideways moves
        # dataframe['vwap'] = qtpylib.vwap(dataframe)
        dataframe['rolling_vwap'] = qtpylib.rolling_vwap(dataframe)
        ###################### Coral Trend Indicator ################################
        dataframe = self.populate_coral_trend(dataframe)

        # ###################### End Coral Trend Indicator ################################

        # populate SSL Channel
        for ssl in self.buy_small_ssl_length.range:
            # if self.ssl_channel_down_index_pattern.format(ssl) not in dataframe.columns:
            sslDown, sslUp = SSLChannels(dataframe, ssl)
            dataframe[self.ssl_channel_down_index_pattern.format(ssl)] = sslDown
            dataframe[self.ssl_channel_up_index_pattern.format(ssl)] = sslUp
        
        for ssl in self.sell_ssl_length.range:
            sslDown, sslUp = SSLChannels(dataframe, ssl)
            dataframe[self.ssl_channel_down_index_pattern.format(ssl)] = sslDown
            dataframe[self.ssl_channel_up_index_pattern.format(ssl)] = sslUp

        dataframe.fillna(0, inplace=True)

        return dataframe

    def init_index_names(self):
        self.buy_small_ssl_channel_down_index_name = self.ssl_channel_down_index_pattern.format(self.buy_small_ssl_length.value)
        self.buy_small_ssl_channel_up_index_name = self.ssl_channel_up_index_pattern.format(self.buy_small_ssl_length.value)
        self.sell_ssl_channel_down_index_name = self.ssl_channel_down_index_pattern.format(self.sell_ssl_length.value)
        self.sell_ssl_channel_up_index_name = self.ssl_channel_up_index_pattern.format(self.sell_ssl_length.value)
        self.buy_coral_index_name = f'coral_{self.buy_coral_sm.value}_{self.buy_coral_cd}'

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.init_index_names()

        long_condition = []

        long_condition.append(
            (dataframe[self.buy_coral_index_name] < dataframe['close'])
        )
        long_condition.append(
            self.ssl_cross_above(dataframe, self.buy_small_ssl_channel_up_index_name, self.buy_small_ssl_channel_down_index_name)
            )

        if long_condition:
                dataframe.loc[
                    reduce(lambda x, y: x & y, long_condition),
                    'enter_long'] = 1
        
        short_condition = []

        # GUARDS AND TRENDS
        short_condition.append(
            # (dataframe[self.buy_pmax_index_name] == 'down') &
            (dataframe[self.buy_coral_index_name] > dataframe['close'])
        )
        short_condition.append(
            self.ssl_cross_below(dataframe, self.buy_small_ssl_channel_up_index_name, self.buy_small_ssl_channel_down_index_name)
            )
        
        if short_condition:
                dataframe.loc[
                    reduce(lambda x, y: x & y, short_condition),
                    'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.init_index_names()
            
        return dataframe
    
    def ssl_cross_above(self, dataframe: DataFrame, up_index_name: str, down_index_name: str) -> bool:
        return qtpylib.crossed_above(dataframe[up_index_name], dataframe[down_index_name])
    
    def ssl_cross_below(self, dataframe: DataFrame, up_index_name: str, down_index_name: str) -> bool:
        return qtpylib.crossed_below(dataframe[up_index_name], dataframe[down_index_name])
    
    def populate_coral_trend(self, dataframe: DataFrame) -> DataFrame:
        print (self.buy_coral_sm.value, self.buy_coral_cd)
        for sm in self.buy_coral_sm.range:
            dataframe[f'coral_{sm}_{self.buy_coral_cd}'] = coral_trend(dataframe, sm, self.buy_coral_cd)
        print ('Coral Trend Indicator Loaded')

        return dataframe

def coral_trend(dataframe: DataFrame, sm: int, cd: int) -> DataFrame:
    di = (sm - 1.0) / 2.0 + 1.0
    c1 = 2.0 / (di + 1.0)
    c2 = 1.0 - c1
    c3 = 3.0 * (cd * cd + cd * cd * cd)
    c4 = -3.0 * (2.0 * cd * cd + cd + cd * cd * cd)
    c5 = 3.0 * cd + 1.0 + cd * cd * cd + 3.0 * cd * cd

    dataframe['coral'] = 0.0

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
        dataframe.loc[index, 'coral'] = -cd*cd*cd*dataframe.loc[index,'i6'] + c3*(dataframe.loc[index,'i5']) + c4*(dataframe.loc[index,'i4']) + c5*(dataframe.loc[index,'i3'])
        
    return dataframe['coral']