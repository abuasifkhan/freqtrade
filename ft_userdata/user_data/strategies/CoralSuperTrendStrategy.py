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
class CoralSuperTrendStrategy(IStrategy):
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
        "buy_m1": 18,
        "buy_p1": 30,
        "buy_pmax_length": 5,
        "buy_pmax_multiplier": 15,
        "buy_pmax_period": 10,
        "buy_sar_accelaretion": 0.002,
        "buy_sar_maximum": 0.0001,
        "buy_sm": 21,
        "buy_trigger": "coral_color_change",
        "buy_use_coral_as_guard": False,
        "buy_use_pmax_as_guard": False,
        "buy_use_sar_as_guard": False,
        "buy_use_super_trend_as_guard": True,
        "current_profit": 0.09,
        "maximum_stoploss": 0.15,
        "minimum_stoploss": 0.2,
        "shouldIgnoreRoi": True,
        "shouldUseStopLoss": False,
        "should_exit_profit_only": True,
        "should_use_exit_signal": False,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 21,
        "sell_p1": 30,
        "sell_pmax_length": 30,
        "sell_pmax_multiplier": 10,
        "sell_pmax_period": 50,
        "sell_sar_accelaretion": 0.0002,
        "sell_sar_maximum": 0.1,
        "sell_sm": 7,
        "sell_trigger": "supertrend",
        "sell_use_coral_as_guard": True,
        "sell_use_pmax_as_guard": False,
        "sell_use_sar_as_guard": True,
        "sell_use_super_trend_as_guard": True,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.085,
        "40": 0.074,
        "93": 0.027,
        "174": 0
    }

    # Stoploss:
    stoploss = -0.032

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

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

    # --------------------------------
    buy_use_coral_as_guard = True
    buy_sm =  buy_params['buy_sm']
    buy_cd = 0.4
    buy_coral_index_name = ''

    sell_use_coral_as_guard = True
    sell_sm =  sell_params['sell_sm']
    sell_cd = 0.4
    sell_coral_index_name = ''
    # --------------------------------

    # --------------------------------
    buy_use_pmax_as_guard = True
    buy_pmax_period = buy_params['buy_pmax_period']
    buy_pmax_multiplier = buy_params['buy_pmax_multiplier']
    buy_pmax_length = buy_params['buy_pmax_length']
    buy_pmax_index_name = ''

    sell_use_pmax_as_guard = True
    sell_pmax_period = sell_params['sell_pmax_period']
    sell_pmax_multiplier = sell_params['sell_pmax_multiplier']
    sell_pmax_length = sell_params['sell_pmax_length']
    sell_pmax_index_name = ''
    # --------------------------------

    # --------------------------------
    buy_use_sar_as_guard = False
    buy_sar_accelaretion = buy_params['buy_sar_accelaretion']
    buy_sar_maximum = buy_params['buy_sar_maximum']
    buy_sar_index_name = ''

    sell_use_sar_as_guard = True
    sell_sar_accelaretion = sell_params['sell_sar_accelaretion']
    sell_sar_maximum = sell_params['sell_sar_maximum']
    sell_sar_index_name = ''
    # --------------------------------

    # --------------------------------
    buy_use_super_trend_as_guard = True
    buy_m1 = buy_params['buy_m1']
    buy_p1 = buy_params['buy_p1']
    buy_super_trend_index_name = ''

    sell_use_super_trend_as_guard = False
    sell_m1 = sell_params['sell_m1']
    sell_p1 = sell_params['sell_p1']
    sell_super_trend_index_name = ''
    # --------------------------------

    # buy_trigger = CategoricalParameter(["coral_ema_cross", "coral_color_change", "sar_ema_cross", "supertrend", "pmax_cross"], default="coral_ema_cross", space="buy")
    # sell_trigger = CategoricalParameter(["coral_ema_cross", "coral_color_change", "sar_ema_cross", "supertrend", "pmax_cross"], default="coral_ema_cross", space="sell")
    buy_trigger = buy_params['buy_trigger']
    sell_trigger = sell_params['sell_trigger']

    # These values can be overridden in the config.
    use_exit_signal = buy_params['should_use_exit_signal']
    exit_profit_only = buy_params['should_exit_profit_only']
    ignore_roi_if_entry_signal = buy_params['shouldIgnoreRoi']

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    use_custom_stoploss = buy_params['shouldUseStopLoss']

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        if current_profit < self.buy_params['current_profit']:
            return -1 # return a value bigger than the inital stoploss to keep using the inital stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit / 2 

        # Use a minimum of 1.5% and a maximum of 3%
        return max(min(desired_stoploss, self.buy_params['minimum_stoploss']), self.buy_params['maximum_stoploss'])


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


    def is_uptrend(self, dataframe) -> bool:
        return (dataframe[self.buy_pmax_index_name] == 'up')  #& (dataframe['coral_medium'] < dataframe['ema3']) # & (dataframe['coral_medium'] < dataframe['ema3'])

    def should_long(self, dataframe) -> bool:
        return self.is_uptrend(dataframe) & qtpylib.crossed_above(dataframe['ema3'], dataframe[self.buy_sar_index_name])

    def is_downtrend(self, dataframe) -> bool:
        return (dataframe[self.buy_pmax_index_name] == 'down') #& (dataframe['coral_medium'] > dataframe['ema3']) #& (dataframe['coral_medium'] > dataframe['low'])

    def should_short(self, dataframe) -> bool:
        return self.is_downtrend(dataframe) & qtpylib.crossed_below(dataframe['ema3'], dataframe[self.buy_sar_index_name])

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

        # Init index names
        self.init_index_names()

        print ('EMA Loading')
        # EMA - Exponential Moving Average
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4.0
        dataframe['ema3'] = ta.EMA(dataframe['ohlc4'], timeperiod=3)
        dataframe['ema21'] = ta.EMA(dataframe['ohlc4'], timeperiod=21)

        print ('EMA Loaded')

        # SuperTrend
        dataframe = self.populate_supertrend(dataframe)

        # Parabolic SAR
        dataframe = self.populate_parabolic_sar(dataframe)

        # # PMAX#PMAX
        dataframe = self.populate_profix_maximizer(dataframe)

        ###################### Coral Trend Indicator ################################
        dataframe = self.populate_coral_trend(dataframe)

        # ###################### End Coral Trend Indicator ################################
        dataframe.fillna(0, inplace=True)
        return dataframe

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
    
    def init_index_names(self):
        self.buy_sar_index_name = f'sar_{self.buy_sar_accelaretion}_{self.buy_sar_maximum}'
        self.sell_sar_index_name = f'sar_{self.sell_sar_accelaretion}_{self.sell_sar_maximum}'
        self.buy_pmax_index_name = f'pmax_{self.buy_pmax_period}_{self.buy_pmax_multiplier}_{self.buy_pmax_length}_{1}'
        self.sell_pmax_index_name = f'pmax_{self.sell_pmax_period}_{self.sell_pmax_multiplier}_{self.sell_pmax_length}_{1}'
        self.buy_coral_index_name = f'coral_{self.buy_sm}_{self.buy_cd}'
        self.sell_coral_index_name = f'coral_{self.sell_sm}_{self.sell_cd}'
        self.buy_super_trend_index_name = f'supertrend_1_buy_{self.buy_m1}_{self.buy_p1}'
        self.sell_super_trend_index_name = f'supertrend_1_sell_{self.sell_m1}_{self.sell_p1}'


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

        if self.buy_use_sar_as_guard == True:
            long_condition.append(dataframe[f'sar_{self.buy_sar_accelaretion}_{self.buy_sar_maximum}'] < dataframe['close'])

        if self.buy_use_pmax_as_guard == True:
            long_condition.append(dataframe[self.buy_pmax_index_name] == 'up')

        if self.buy_use_coral_as_guard == True:
            long_condition.append(dataframe[self.buy_coral_index_name] < dataframe['ema3'])
        
        if self.buy_use_super_trend_as_guard == True:
            long_condition.append(dataframe[self.buy_super_trend_index_name] == 'up')
        
        return long_condition
    
    def populate_long_exit_guards(self, dataframe: DataFrame) -> DataFrame:
        short_exit = []
        if self.sell_use_sar_as_guard == True:
            short_exit.append(dataframe[f'sar_{self.sell_sar_accelaretion}_{self.sell_sar_maximum}'] > dataframe['close'])
        
        if self.sell_use_pmax_as_guard == True:
            short_exit.append(dataframe[self.sell_pmax_index_name] == 'down')
        
        if self.sell_use_coral_as_guard == True:
            short_exit.append(dataframe[self.sell_coral_index_name] > dataframe['ema3'])
        
        if self.sell_use_super_trend_as_guard == True:
            short_exit.append(dataframe[self.sell_super_trend_index_name] == 'down')
        
        return short_exit
    
    def populate_short_entry_guards(self, dataframe: DataFrame) -> DataFrame:
        short_condition = []

        # GUARDS AND TRENDS
        if self.buy_use_sar_as_guard == True:
            short_condition.append(dataframe[self.buy_sar_index_name] > dataframe['close'])

        if self.buy_use_pmax_as_guard == True:
            short_condition.append(dataframe[self.buy_pmax_index_name] == 'down')

        if self.buy_use_coral_as_guard == True:
            short_condition.append(dataframe[self.buy_coral_index_name] > dataframe['ema3'])
        
        if self.sell_use_super_trend_as_guard == True:
            short_condition.append(dataframe[self.sell_super_trend_index_name] == 'down')
        
        return short_condition
    
    def populate_short_exit_guards(self, dataframe: DataFrame) -> DataFrame:
        short_exit = []

        # GUARDS AND TRENDS
        if self.sell_use_sar_as_guard == True:
            short_exit.append(dataframe[self.sell_sar_index_name] < dataframe['close'])
        
        if self.sell_use_pmax_as_guard == True:
            short_exit.append(dataframe[self.sell_pmax_index_name] == 'up')
        
        if self.sell_use_coral_as_guard == True:
            short_exit.append(dataframe[self.sell_coral_index_name] < dataframe['ema3'])
        
        if self.sell_use_super_trend_as_guard == True:
            short_exit.append(dataframe[self.sell_super_trend_index_name] == 'up')
        
        return short_exit

    def populate_long_trigger(self, dataframe: DataFrame, long_condition):
        if self.buy_trigger == "sar_ema_cross":
            long_condition.append(
                qtpylib.crossed_above(dataframe['ema3'], dataframe[self.buy_sar_index_name])
            )
        elif self.buy_trigger == "coral_ema_cross":
            long_condition.append(
                qtpylib.crossed_above(dataframe['ema3'], dataframe[self.buy_coral_index_name])
            )
        elif self.buy_trigger == "coral_color_change":
            long_condition.append(
                self.green_from_red(dataframe[self.buy_coral_index_name])
            )
        elif self.buy_trigger == "supertrend":
            long_condition.append(
                (dataframe[self.buy_super_trend_index_name] == 'up') &
                (dataframe[self.buy_super_trend_index_name].shift(1) == 'down')
            )

        return long_condition
    
    def populate_long_exit_trigger(self, dataframe: DataFrame, long_exit):
        if self.sell_trigger == "sar_ema_cross":
            long_exit.append(
                qtpylib.crossed_below(dataframe['ema3'], dataframe[self.sell_sar_index_name])
            )
        elif self.sell_trigger == "coral_ema_cross":
            long_exit.append(
                qtpylib.crossed_below(dataframe['ema3'], dataframe[self.sell_coral_index_name])
            )
        elif self.sell_trigger == "coral_color_change":
            long_exit.append(
                self.red_from_green(dataframe[self.sell_coral_index_name])
            )
        elif self.sell_trigger == "supertrend":
            long_exit.append(
                (dataframe[self.sell_super_trend_index_name] == 'down') &
                (dataframe[self.sell_super_trend_index_name].shift(1) == 'up')
            )

        return long_exit
    
    def populate_short_trigger(self, dataframe: DataFrame, short_condition):
        if self.buy_trigger == "sar_ema_cross":
            short_condition.append(
                qtpylib.crossed_below(dataframe['ema3'], dataframe[self.buy_sar_index_name])
            )
        elif self.buy_trigger == "coral_ema_cross":
            short_condition.append(
                qtpylib.crossed_below(dataframe['ema3'], dataframe[self.buy_coral_index_name])
            )
        elif self.buy_trigger == "coral_color_change":
            short_condition.append(
                self.red_from_green(dataframe[self.buy_coral_index_name])
            )
        elif self.buy_trigger == "supertrend":
            short_condition.append(
                (dataframe[self.buy_super_trend_index_name] == 'down') &
                (dataframe[self.buy_super_trend_index_name].shift(1) == 'up')
            )
        
        return short_condition
    
    def populate_short_exit_trigger(self, dataframe: DataFrame, exit_short):
        if self.sell_trigger == "sar_ema_cross":
            exit_short.append(
                qtpylib.crossed_above(dataframe['ema3'], dataframe[self.sell_sar_index_name])
            )
        elif self.sell_trigger == "coral_ema_cross":
            exit_short.append(
                qtpylib.crossed_above(dataframe['ema3'], dataframe[self.sell_coral_index_name])
            )
        elif self.sell_trigger == "coral_color_change":
            exit_short.append(
                self.green_from_red(dataframe[self.sell_coral_index_name])
            )
        elif self.sell_trigger == "supertrend":
            exit_short.append(
                (dataframe[self.sell_super_trend_index_name] == 'up') &
                (dataframe[self.sell_super_trend_index_name].shift(1) == 'down')
            )
        
        return exit_short
    
    def populate_supertrend(self, dataframe: DataFrame) -> DataFrame:
        dataframe[f"supertrend_1_buy_{self.buy_m1}_{self.buy_p1}"] = self.supertrend(
            dataframe, self.buy_m1, self.buy_p1
        )["STX"]

        dataframe[f"supertrend_1_sell_{self.sell_m1}_{self.sell_p1}"] = self.supertrend(
            dataframe, self.sell_m1, self.sell_p1
        )["STX"]

        return dataframe
    
    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df["TR"] = ta.TRANGE(df)
        df["ATR"] = ta.SMA(df["TR"], period)

        st = "ST_" + str(period) + "_" + str(multiplier)
        stx = "STX_" + str(period) + "_" + str(multiplier)

        # Compute basic upper and lower bands
        df["basic_ub"] = (df["high"] + df["low"]) / 2 + multiplier * df["ATR"]
        df["basic_lb"] = (df["high"] + df["low"]) / 2 - multiplier * df["ATR"]

        # Compute final upper and lower bands
        df["final_ub"] = 0.00
        df["final_lb"] = 0.00
        for i in range(period, len(df)):
            df["final_ub"].iat[i] = (
                df["basic_ub"].iat[i]
                if df["basic_ub"].iat[i] < df["final_ub"].iat[i - 1]
                or df["close"].iat[i - 1] > df["final_ub"].iat[i - 1]
                else df["final_ub"].iat[i - 1]
            )
            df["final_lb"].iat[i] = (
                df["basic_lb"].iat[i]
                if df["basic_lb"].iat[i] > df["final_lb"].iat[i - 1]
                or df["close"].iat[i - 1] < df["final_lb"].iat[i - 1]
                else df["final_lb"].iat[i - 1]
            )

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = (
                df["final_ub"].iat[i]
                if df[st].iat[i - 1] == df["final_ub"].iat[i - 1]
                and df["close"].iat[i] <= df["final_ub"].iat[i]
                else df["final_lb"].iat[i]
                if df[st].iat[i - 1] == df["final_ub"].iat[i - 1]
                and df["close"].iat[i] > df["final_ub"].iat[i]
                else df["final_lb"].iat[i]
                if df[st].iat[i - 1] == df["final_lb"].iat[i - 1]
                and df["close"].iat[i] >= df["final_lb"].iat[i]
                else df["final_ub"].iat[i]
                if df[st].iat[i - 1] == df["final_lb"].iat[i - 1]
                and df["close"].iat[i] < df["final_lb"].iat[i]
                else 0.00
            )
        # Mark the trend direction up/down
        df[stx] = np.where(
            (df[st] > 0.00), np.where((df["close"] < df[st]), "down", "up"), np.NaN
        )

        # Remove basic and final bands from the columns
        df.drop(["basic_ub", "basic_lb", "final_ub", "final_lb"], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={"ST": df[st], "STX": df[stx]})
    
    def populate_parabolic_sar(self, dataframe: DataFrame) -> DataFrame:
        print ('Parabolic SAR Loading')
        afstep = 0.03
        aflimit = 0.03
        epstep = 0.03
        eplimit = 0.3
        name = f'sar_{self.buy_sar_accelaretion}_{self.buy_sar_maximum}'
        print("Printing: " + name)
        temp = ta.SAR(dataframe['high'], dataframe['low'], acceleration=self.buy_sar_accelaretion, maximum=self.buy_sar_maximum,
                                afstep=afstep, aflimit=aflimit, epstep=epstep, eplimit=eplimit)
        dataframe[name] = temp
        print('Done')
        
        print ('Parabolic SAR Loaded')

        print ('Parabolic SAR Loading')

        afstep = 0.03
        aflimit = 0.03
        epstep = 0.03
        eplimit = 0.3
        name = f'sar_{self.sell_sar_accelaretion}_{self.sell_sar_maximum}'
        print("Printing: " + name)
        temp = ta.SAR(dataframe['high'], dataframe['low'], acceleration=self.sell_sar_accelaretion, maximum=self.sell_sar_maximum,
                                afstep=afstep, aflimit=aflimit, epstep=epstep, eplimit=eplimit)
        dataframe[name] = temp
        print('Done')
        
        print ('Parabolic SAR Loaded')

        return dataframe
    
    def populate_profix_maximizer(self, dataframe: DataFrame) -> DataFrame:
        print ('Profit Maximizer Loading')
        pmax_MAtype = 1
        dataframe = PMAX(dataframe, period=self.buy_pmax_period, multiplier=self.buy_pmax_multiplier, length=self.buy_pmax_length, MAtype=pmax_MAtype)
        print('Profit Maximizer Loaded')

        print ('Maximizer Loading')
        pmax_MAtype = 1
        dataframe = PMAX(dataframe, period=self.sell_pmax_period, multiplier=self.sell_pmax_multiplier, length=self.sell_pmax_length, MAtype=pmax_MAtype)
        print('Short Maximizer Loaded')

        return dataframe
    
    def populate_coral_trend(self, dataframe: DataFrame) -> DataFrame:
        print('Loading medium Coral Trend Indicator')
        dataframe[f'coral_{self.buy_sm}_{self.buy_cd}'] = coral_trend(dataframe, self.buy_sm, self.buy_cd)
        print ('Medium Coral Trend Indicator Loaded')

        print('Loading medium Coral Trend Indicator')
        dataframe[f'coral_{self.sell_sm}_{self.sell_cd}'] = coral_trend(dataframe, self.sell_sm, self.sell_cd)
        print ('Medium Coral Trend Indicator Loaded')

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
