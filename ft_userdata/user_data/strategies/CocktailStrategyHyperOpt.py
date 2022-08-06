# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
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

class CocktailStrategyHyperOpt(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.435,
        "46": 0.147,
        "196": 0.052,
        "525": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.202

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # MY INDICATORS
    # SAR parameters
    sar_parameters = {
        "acceleration": 0.08,
        "maximum": 0.2,
        "afstep": 0.03,
        "aflimit": 0.03,
        "epstep": 0.03,
        "eplimit": 0.3,
    }

    # Coral Parameters
    fast_coral_trend_parameters = {
        "fast_sm": 21,
        "fast_cd": 0.4
    }

    medium_coral_trend_parameters = {
        "medium_sm": 50,
        "medium_cd": 0.9
    }

    pmax_parameters = {
        "period": 18, 
        "multiplier": 4, 
        "length": 21,
        "MAtype": 1
    }
    
    atr_parameters = {
        "length": 14,
        "threshold": 0.5
    }

    macd_parameters = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    }
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

    dataframe = pd.DataFrame()

    # Hyperoptable parameters
    ema_length = CategoricalParameter([1, 3, 5], default=3, space='buy')
    ema_index_name = ''
    # buy_adx_threshold = DecimalParameter(10, 40, decimals=2, default=20)
    # buy_adx_enabled = BooleanParameter(default=False)

    use_fast_bfr_as_guard = BooleanParameter(default=True, space='buy')
    fast_sm_value = CategoricalParameter([7, 14, 21], default=fast_coral_trend_parameters['fast_sm'], space='buy')
    # fast_cd_value = DecimalParameter(0.4, 0.5, decimals=1, default=fast_coral_trend_parameters['fast_cd'], space='buy')
    fast_cd_value = fast_coral_trend_parameters['fast_cd']
    fast_coral_index_name = ''

    use_medium_bfr_as_guard = BooleanParameter(default=True, space='buy')
    medium_sm_value =  CategoricalParameter([50, 100], default=medium_coral_trend_parameters['medium_sm'], space='buy')
    medium_cd_value =  CategoricalParameter([0.4, 0.9], default=medium_coral_trend_parameters['medium_cd'], space='buy')
    medium_coral_index_name = ''

    shouldIgnoreRoi = BooleanParameter(default=False, space='buy')
    shouldUseStopLoss = BooleanParameter(default=False, space='buy')

    use_pmax_as_guard = BooleanParameter(default=True, space='buy')
    pmax_period = CategoricalParameter([5, 10, 15], default=pmax_parameters['period'], space='buy')
    pmax_multiplier = CategoricalParameter([4, 7, 10], default=pmax_parameters['multiplier'], space='buy')
    pmax_length = CategoricalParameter([15, 20, 30], default=pmax_parameters['length'], space='buy')
    pmax_index_name = ''

    use_sar_as_guard = BooleanParameter(default=True, space='buy')
    sar_accelaretion = CategoricalParameter([0.02, 0.04, 0.06, 0.08], default=sar_parameters['acceleration'], space='buy')
    sar_maximum = CategoricalParameter([0.1, 0.2, 0.3], default=sar_parameters['maximum'], space='buy')
    sar_index_name = ''

    use_atrP_as_guard = BooleanParameter(default=True, space="buy")
    atr_length = IntParameter(5, 20, default=atr_parameters['length'], space="buy")
    atr_threshold = DecimalParameter(0.1, 1.0, decimals=3, default=atr_parameters['threshold'], space='buy')
    atr_index_name = ''
    atrP_index_name = ''

    use_macd_as_guard = BooleanParameter(default=True, space="buy")
    macd_fast_period = CategoricalParameter([6, 10, 15, 20], default=macd_parameters['fast_period'], space="buy")
    macd_slow_period = CategoricalParameter([50, 60, 70], default=macd_parameters['slow_period'], space="buy")
    macd_signal = CategoricalParameter([10, 18, 22], default=macd_parameters['signal_period'], space="buy")
    macd_index_name = ''
    macd_signal_index_name = ''
    macd_histogram_index_name = ''

    buy_trigger = CategoricalParameter(["fast_bfr_color_change", "medium_bfr_ema_cross", "medium_bfr_color_change", "macd_crossover", "sar_ema_cross"], default="medium_bfr_ema_cross", space="buy")
    sell_trigger = CategoricalParameter(["fast_bfr_color_change", "medium_bfr_ema_cross", "medium_bfr_color_change", "macd_crossover", "sar_ema_cross"], default="medium_bfr_ema_cross", space="sell")

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = shouldIgnoreRoi.value

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    use_custom_stoploss = shouldUseStopLoss.value

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        if current_profit < 0.01:
            return -1 # return a value bigger than the inital stoploss to keep using the inital stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit / 2 

        # Use a minimum of 1.5% and a maximum of 3%
        return max(min(desired_stoploss, 0.3), 0.15)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Init index names
        self.init_index_names()

        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4.0

        self.dataframe = dataframe

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.init_index_names()
        dataframe = self.populate_indicator_for_current_epoch(dataframe)
        long_condition = self.populate_long_entry_guards(dataframe, metadata)
        long_condition = self.populate_long_trigger(dataframe, long_condition=long_condition)

        if long_condition:
                dataframe.loc[
                    reduce(lambda x, y: x & y, long_condition),
                    'enter_long'] = 1
        
        short_condition = self.populate_short_entry_guards(dataframe, metadata)
        short_condition = self.populate_short_trigger(dataframe, short_condition=short_condition)
        
        if short_condition:
                dataframe.loc[
                    reduce(lambda x, y: x & y, short_condition),
                    'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.init_index_names()
        conditions = []
        return dataframe
    
    def populate_long_entry_guards(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_condition = []

        # GUARDS AND TRENDS
        if self.use_sar_as_guard.value == True:
            sar_index_name = self.sar_index_name.format(self.sar_accelaretion.value, self.sar_maximum.value)
            long_condition.append(dataframe[sar_index_name] < dataframe['close'])

        if self.use_pmax_as_guard.value == True:
            pmax_index_name = self.pmax_index_name.format(self.pmax_period.value, self.pmax_multiplier.value, self.pmax_length.value)
            long_condition.append(dataframe[pmax_index_name] == 'up')

        if self.use_atrP_as_guard.value == True:
            atrP_index_name = self.atrP_index_name.format(self.atr_length.value)
            long_condition.append(dataframe[atrP_index_name] > self.atr_threshold.value)

        if self.use_fast_bfr_as_guard.value == True:
            fast_coral_index_name = self.fast_coral_index_name.format(self.fast_coral_length.value)
            ema = self.ema_index_name.format(self.ema_length.value)
            long_condition.append(dataframe[fast_coral_index_name] < dataframe[ema])

        if self.use_medium_bfr_as_guard.value == True:
            medium_coral_index_name = self.medium_coral_index_name.format(self.medium_sm_value.value, self.medium_cd_value.value)
            ema = self.ema_index_name.format(self.ema_length.value)
            long_condition.append(dataframe[medium_coral_index_name] < dataframe[ema])

        if self.use_macd_as_guard.value == True:
            macd_index_name = self.macd_index_name.format(self.macd_fast_period.value, self.macd_slow_period.value, self.macd_signal.value)
            long_condition.append(dataframe[macd_index_name] > 0)
        
        return long_condition
    
    def populate_short_entry_guards(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        short_condition = []

        # GUARDS AND TRENDS
        # short_condition.append(self.is_downtrend(dataframe))
        if self.use_sar_as_guard.value == True:
            sar_index_name = self.sar_index_name.format(self.sar_accelaretion.value, self.sar_maximum.value)    
            short_condition.append(dataframe[sar_index_name] > dataframe['close'])

        if self.use_pmax_as_guard.value == True:
            pmax_index_name = self.pmax_index_name.format(self.pmax_period.value, self.pmax_multiplier.value, self.pmax_length.value)
            short_condition.append(dataframe[pmax_index_name] == 'down')

        if self.use_atrP_as_guard.value == True:
            atrP_index_name = self.atrP_index_name.format(self.atr_length.value)
            short_condition.append(dataframe[atrP_index_name] > self.atr_threshold.value)

        if self.use_fast_bfr_as_guard.value == True:
            fast_coral_index_name = self.fast_coral_index_name.format(self.fast_coral_length.value)
            ema = self.ema_index_name.format(self.ema_length.value)
            short_condition.append(dataframe[fast_coral_index_name] > dataframe[ema])

        if self.use_medium_bfr_as_guard.value == True:
            medium_coral_index_name = self.medium_coral_index_name.format(self.medium_sm_value.value, self.medium_cd_value.value)
            ema = self.ema_index_name.format(self.ema_length.value)
            short_condition.append(dataframe[medium_coral_index_name] > dataframe[ema])

        if self.use_macd_as_guard.value == True:
            macd_index_name = self.macd_index_name.format(self.macd_fast_period.value, self.macd_slow_period.value, self.macd_signal.value)
            short_condition.append(dataframe[macd_index_name] < 0)
        
        return short_condition
    
    def populate_long_trigger(self, dataframe: DataFrame, long_condition):
        if self.buy_trigger.value == 'fast_bfr_color_change':
            long_condition.append(
                self.green_from_red(dataframe[self.fast_coral_index_name])
            )
        elif self.buy_trigger.value == "sar_ema_cross":
            long_condition.append(
                qtpylib.crossed_above(dataframe[self.ema_index_name], dataframe[self.sar_index_name])
            )
        elif self.buy_trigger.value == "medium_bfr_ema_cross":
            long_condition.append(
                qtpylib.crossed_above(dataframe[self.ema_index_name], dataframe[self.medium_coral_index_name])
            )
        elif self.buy_trigger.value == "medium_bfr_color_change":
            long_condition.append(
                self.green_from_red(dataframe[self.medium_coral_index_name])
            )
        elif self.buy_trigger.value == "macd_crossover":
            long_condition.append(
                qtpylib.crossed_above(dataframe[self.macd_histogram_index_name], 0)
            )
        return long_condition
    
    def populate_short_trigger(self, dataframe: DataFrame, short_condition):
        if self.buy_trigger.value == 'fast_bfr_color_change':
            short_condition.append(
                self.red_from_green(dataframe[self.fast_coral_index_name])
            )
        elif self.buy_trigger.value == "sar_ema_cross":
            short_condition.append(
                qtpylib.crossed_below(dataframe['ema3'], dataframe[self.sar_index_name])
            )
        elif self.buy_trigger.value == "medium_bfr_ema_cross":
            short_condition.append(
                qtpylib.crossed_below(dataframe['ema3'], dataframe[self.medium_coral_index_name])
            )
        elif self.buy_trigger.value == "medium_bfr_color_change":
            short_condition.append(
                self.red_from_green(dataframe[self.medium_coral_index_name])
            )
        elif self.buy_trigger.value == "macd_crossover":
            short_condition.append(
                qtpylib.crossed_below(dataframe[self.macd_histogram_index_name], 0)
            )
        return short_condition

    def init_index_names(self):
        self.atr_index_name = 'atr_{0}'
        self.atrP_index_name = 'atrP_{0}'
        self.macd_index_name = 'macd_{0}_{1}_{2}'
        self.macd_signal_index_name = 'macd_signal_{0}_{1}'
        self.macd_histogram_index_name = 'macdhist_{0}_{1}_{2}'
        self.sar_index_name = 'sar_{0}_{1}'
        self.pmax_index_name = 'pmax_{0}_{1}_{2}_{3}'
        self.fast_coral_index_name = 'coral_fast_{0}_{1}'
        self.medium_coral_index_name = 'coral_medium_{0}_{1}'
    
    def populate_atr(self, dataframe: DataFrame, period: int) -> DataFrame:
        atr_i = self.atr_index_name.format(period)
        atrP_i = self.atrP_index_name.format(period)
        if period not in dataframe.columns:
                dataframe[atr_i] = ta.ATR(dataframe, period=period)
                dataframe[atrP_i] = dataframe[atr_i] / dataframe['close'].fillna(1)
        
        return dataframe

    def populate_macd(self, dataframe: DataFrame, macd_fast: int, macd_slow: int, macd_signal: int) -> DataFrame:
        macd_i = self.macd_index_name.format(macd_fast, macd_slow, macd_signal)
        macd_signal_i = self.macd_signal_index_name.format(macd_fast, macd_slow, macd_signal)
        macd_histogram_i = self.macd_histogram_index_name.format(macd_fast, macd_slow, macd_signal)

        # MACD
        if macd_histogram_i not in dataframe.columns:
            macd = ta.MACD(dataframe, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
            macd.rename(columns = {'macd' : macd_i}, inplace = True)
            macd.rename(columns = {'macdsignal' : macd_signal_i}, inplace = True)
            macd.rename(columns = {'macdhist' : macd_histogram_i}, inplace = True)
            dataframe = pd.concat([dataframe, macd], axis=1, join="inner")
        
        return dataframe

    def populate_ema(self, dataframe: DataFrame, length: int) -> DataFrame:
        # EMA
        ema_i = self.ema_index_name.format(length)
        if ema_i not in dataframe.columns:
            dataframe[ema_i] = ta.EMA(dataframe, period=length)
        
        return dataframe

    def populate_parabolic_sar(self, dataframe: DataFrame, accelaretion: float, maximum: float) -> DataFrame:
        # Parabolic SAR
        afstep = 0.03
        aflimit = 0.03
        epstep = 0.03
        eplimit = 0.3
        sar_i = self.sar_index_name.format(accelaretion, maximum)
        if sar_i not in dataframe.columns:
            dataframe[sar_i] = ta.SAR(dataframe['high'], dataframe['low'], acceleration=accelaretion, maximum=maximum,
                                    afstep=afstep, aflimit=aflimit, epstep=epstep, eplimit=eplimit)
        
        return dataframe
    
    def populate_pmax(self, dataframe: DataFrame, period: int, multiplier: float, length: int, MAtype: int) -> DataFrame:
        dataframe = PMAX(dataframe, period=period, multiplier=multiplier, length=length, MAtype=MAtype)
        
        return dataframe

    def populate_fast_coral(self, dataframe: DataFrame, sm_value: int, cd_value: int) -> DataFrame:
        coral_i = self.fast_coral_index_name.format(sm_value, cd_value)
        dataframe[coral_i] = coral_trend(dataframe, sm_value, cd_value)
        return dataframe
    
    def populate_medium_coral(self, dataframe: DataFrame, sm_value: int, cd_value: int) -> DataFrame:
        coral_i = self.medium_coral_index_name.format(sm_value, cd_value)
        dataframe[coral_i] = coral_trend(dataframe, sm_value, cd_value)
        return dataframe
    
    def is_uptrend(self, dataframe) -> bool:
        return (dataframe[self.pmax_index_name] == 'up')  #& (dataframe['bfr_medium'] < dataframe['ema3']) # & (dataframe['bfr_medium'] < dataframe['ema3'])

    def should_long(self, dataframe) -> bool:
        return self.is_uptrend(dataframe) & qtpylib.crossed_above(dataframe[self.ema_index_name], dataframe[self.sar_index_name])

    def is_downtrend(self, dataframe) -> bool:
        return (dataframe[self.pmax_index_name] == 'down') #& (dataframe['bfr_medium'] > dataframe['ema3']) #& (dataframe['bfr_medium'] > dataframe['low'])

    def should_short(self, dataframe) -> bool:
        return self.is_downtrend(dataframe) & qtpylib.crossed_below(dataframe[self.ema_index_name], dataframe[self.sar_index_name])

    def is_green(self, dataframe_1d) -> bool:
        return np.greater(dataframe_1d, dataframe_1d.shift(1))

    def is_red(self, dataframe_1d) -> bool:
        return np.less(dataframe_1d, dataframe_1d.shift(1))

    def green_from_red(self, dataframe_1d) -> bool:
        return self.is_red(dataframe_1d.shift(1)) & self.is_green(dataframe_1d)

    def red_from_green(self, dataframe_1d) -> bool:
        return self.is_green(dataframe_1d.shift(1)) & self.is_red(dataframe_1d)
    
    def populate_indicator_for_current_epoch(self, dataframe) -> DataFrame:
        dataframe = self.populate_atr(dataframe, period=self.atr_length.value)
        dataframe = self.populate_macd(dataframe, macd_fast=self.macd_fast_period.value, macd_slow=self.macd_slow_period.value, macd_signal=self.macd_signal.value)
        dataframe = self.populate_ema(dataframe, length=self.ema_length.value)
        dataframe = self.populate_parabolic_sar(dataframe, accelaretion=self.sar_accelaretion.value, maximum=self.sar_maximum.value)
        dataframe = self.populate_pmax(dataframe, period=self.pmax_period.value, multiplier=self.pmax_multiplier.value, length=self.pmax_length.value, MAtype=self.pmax_parameters['MAtype'])
        dataframe = self.populate_fast_coral(dataframe, sm_value=self.fast_sm_value.value, cd_value=self.fast_cd_value)
        dataframe = self.populate_medium_coral(dataframe, sm_value=self.medium_sm_value.value, cd_value=self.medium_cd_value.value)

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
        
    return dataframe['bfr']