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
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

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

    # Hyperoptable parameters
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
    #     return is_green(dataframe['bfr_fast']) # High winrate

    def is_uptrend(self, dataframe) -> bool:
        return (dataframe[self.pmax_index_name] == 'up')  #& (dataframe['bfr_medium'] < dataframe['ema3']) # & (dataframe['bfr_medium'] < dataframe['ema3'])

    # def should_long(dataframe) -> bool:
    #     return is_uptrend(dataframe) & qtpylib.crossed_above(dataframe['ema3'], dataframe['bfr_medium'])

    def should_long(self, dataframe) -> bool:
        return self.is_uptrend(dataframe) & qtpylib.crossed_above(dataframe['ema3'], dataframe[self.sar_index_name])

    # def is_downtrend(dataframe) -> bool:
    #     return is_red(dataframe['bfr_fast'])

    def is_downtrend(self, dataframe) -> bool:
        return (dataframe[self.pmax_index_name] == 'down') #& (dataframe['bfr_medium'] > dataframe['ema3']) #& (dataframe['bfr_medium'] > dataframe['low'])

    def should_short(self, dataframe) -> bool:
        return self.is_downtrend(dataframe) & qtpylib.crossed_below(dataframe['ema3'], dataframe[self.sar_index_name])

    # def should_short(dataframe) -> bool:
    #     return is_downtrend(dataframe) & qtpylib.crossed_below(dataframe['ema3'], dataframe['bfr_medium'])

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

        # Init index names
        self.init_index_names()

        # ATR
        for val in self.atr_length.range:
            dataframe[f'atr_{val}'] = ta.ATR(dataframe, period=val)
            dataframe[f'atrP_{val}'] = dataframe[f'atr_{val}'] / dataframe['close'].fillna(1)

        print ('ATR Loaded')

        print ('Loading MACD')
        # MACD        
        for fast in self.macd_fast_period.range:
            for slow in self.macd_slow_period.range:
                for signal in self.macd_signal.range:
                    macd = ta.MACD(dataframe, fastperiod=fast, slowperiod=slow, signalperiod=signal)
                    macd.rename(columns = {'macd' : f'macd_{fast}_{slow}_{signal}'}, inplace = True)
                    macd.rename(columns = {'macdsignal' : f'macdsignal_{fast}_{slow}_{signal}'}, inplace = True)
                    macd.rename(columns = {'macdhist' : f'macdhist_{fast}_{slow}_{signal}'}, inplace = True)
                    dataframe = pd.concat([dataframe, macd], axis=1, join="inner")
                    print (f'MACD: {fast} {slow} {signal}')
        print ('MACD Loaded')

        print ('EMA Loading')
        # # EMA - Exponential Moving Average
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4.0
        dataframe['ema3'] = ta.EMA(dataframe['ohlc4'], timeperiod=3)
        dataframe['ema21'] = ta.EMA(dataframe['ohlc4'], timeperiod=21)

        print ('EMA Loaded')

        # Parabolic SAR
        print ('Parabolic SAR Loading')
        for accelaretion in self.sar_accelaretion.range:
            for maximum in self.sar_maximum.range:
                afstep = 0.03
                aflimit = 0.03
                epstep = 0.03
                eplimit = 0.3
                name = f'sar_{accelaretion}_{maximum}'
                print("Printing: " + name)
                temp = ta.SAR(dataframe['high'], dataframe['low'], acceleration=accelaretion, maximum=maximum,
                                        afstep=afstep, aflimit=aflimit, epstep=epstep, eplimit=eplimit)
                dataframe[name] = temp
                print('Done')
        
        print ('Parabolic SAR Loaded')

        # # PMAX#PMAX
        print ('Profit Maximizer Loading')
        for period in self.pmax_period.range:
            for multiplier in self.pmax_multiplier.range:
                for length in self.pmax_length.range:
                    pmax_MAtype = self.pmax_parameters["MAtype"]
                    dataframe = PMAX(dataframe, period=period, multiplier=multiplier, length=length, MAtype=pmax_MAtype)
        print('Profit Maximizer Loaded')

        # #############################################################################
        ###################### Coral Trend Indicator ################################
        for fast_sm in self.fast_sm_value.range:
            # for fast_cd in self.fast_cd_value.range:
            print('Loading fast Coral Trend Indicator')
            dataframe[f'bfr_fast_{fast_sm}_{self.fast_cd_value}'] = coral_trend(dataframe, fast_sm, self.fast_cd_value)
        print ('Fast Coral Trend Indicator Loaded')

        for medium_sm in self.medium_sm_value.range:
            for medium_cd in self.medium_cd_value.range:
                print('Loading medium Coral Trend Indicator')
                dataframe[f'bfr_medium_{medium_sm}_{medium_cd}'] = coral_trend(dataframe, medium_sm, medium_cd)
        print ('Medium Coral Trend Indicator Loaded')
        # #############################################################################
        # ###################### End Coral Trend Indicator ################################

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.init_index_names()

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
    
    def init_index_names(self):
        self.atr_index_name = f'atr_{self.atr_length.value}'
        self.atrP_index_name = f'atrP_{self.atr_length.value}'
        self.macd_histogram_index_name = f'macdhist_{self.macd_fast_period.value}_{self.macd_slow_period.value}_{self.macd_signal.value}'
        self.sar_index_name = f'sar_{self.sar_accelaretion.value}_{self.sar_maximum.value}'
        self.pmax_index_name = f'pmax_{self.pmax_period.value}_{self.pmax_multiplier.value}_{self.pmax_length.value}_{self.pmax_parameters["MAtype"]}'
        self.fast_coral_index_name = f'bfr_fast_{self.fast_sm_value.value}_{self.fast_cd_value}'
        self.medium_coral_index_name = f'bfr_medium_{self.medium_sm_value.value}_{self.medium_cd_value.value}'

        print (self.atr_index_name, self.atrP_index_name, self.macd_histogram_index_name, self.sar_index_name, self.pmax_index_name, self.fast_coral_index_name, self.medium_coral_index_name)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.init_index_names()
        # GUARDS AND TRENDS
        condition = []
        # condition.append(is_bearish_trend(dataframe))
        # if self.buy_trigger == "medium_bfr_ema_cross":
        #     condition.append(
        #         qtpylib.crossed_below(dataframe['ema3'], dataframe['bfr_medium'])
        #     )
        # elif self.buy_trigger == "medium_bfr_color_change":
        #     condition.append(
        #         red_from_green(dataframe['bfr_medium'])
        #     )
            
        return dataframe

    def populate_long_entry_guards(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_condition = []

        # GUARDS AND TRENDS
        if self.use_sar_as_guard.value == True:
            long_condition.append(dataframe[f'sar_{self.sar_accelaretion.value}_{self.sar_maximum.value}'] < dataframe['close'])

        if self.use_pmax_as_guard.value == True:
            long_condition.append(dataframe[self.pmax_index_name] == 'up')

        if self.use_atrP_as_guard.value == True:
            long_condition.append(dataframe[self.atrP_index_name] > self.atr_threshold.value)

        if self.use_fast_bfr_as_guard.value == True:
            long_condition.append(dataframe[self.fast_coral_index_name] < dataframe['ema3'])

        if self.use_medium_bfr_as_guard.value == True:
            long_condition.append(dataframe[self.medium_coral_index_name] < dataframe['ema3'])

        if self.use_macd_as_guard.value == True:
            long_condition.append(dataframe[self.macd_histogram_index_name] > 0)
        
        return long_condition
    
    def populate_short_entry_guards(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        short_condition = []

        # GUARDS AND TRENDS
        # short_condition.append(self.is_downtrend(dataframe))
        if self.use_sar_as_guard.value == True:
            short_condition.append(dataframe[self.sar_index_name] > dataframe['close'])

        if self.use_pmax_as_guard.value == True:
            short_condition.append(dataframe[self.pmax_index_name] == 'down')

        if self.use_atrP_as_guard.value == True:
            short_condition.append(dataframe[self.atrP_index_name] > self.atr_threshold.value)

        if self.use_fast_bfr_as_guard.value == True:
            short_condition.append(dataframe[self.fast_coral_index_name] > dataframe['ema3'])

        if self.use_medium_bfr_as_guard.value == True:
            short_condition.append(dataframe[self.medium_coral_index_name] > dataframe['ema3'])

        if self.use_macd_as_guard.value == True:
            short_condition.append(dataframe[self.macd_histogram_index_name] < 0)
        
        return short_condition

    def populate_long_trigger(self, dataframe: DataFrame, long_condition):
        if self.buy_trigger.value == 'fast_bfr_color_change':
            long_condition.append(
                self.green_from_red(dataframe[self.fast_coral_index_name])
            )
        elif self.buy_trigger.value == "sar_ema_cross":
            long_condition.append(
                qtpylib.crossed_above(dataframe['ema3'], dataframe[self.sar_index_name])
            )
        elif self.buy_trigger.value == "medium_bfr_ema_cross":
            long_condition.append(
                qtpylib.crossed_above(dataframe['ema3'], dataframe[self.medium_coral_index_name])
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