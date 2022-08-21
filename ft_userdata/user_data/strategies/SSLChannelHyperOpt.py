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
                                IStrategy, IntParameter, stoploss_from_open)

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
        "buy_leverage": 4,
        "buy_small_ssl_length": 5,
        "maximum_stoploss": 0.005,
        "profit_threshold": 0.005,
        "shouldIgnoreRoi": True,
        "shouldUseStopLoss": False,
        "should_exit_profit_only": True,
        "should_use_exit_signal": False,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_ssl_length": 30,
        "use_coral_as_exit_trigger": True,
        "use_ssl_as_exit_trigger": True,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.089,
        "34": 0.057,
        "89": 0.036,
        "202": 0
    }

    # Stoploss:
    stoploss = -0.078

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

    shouldIgnoreRoi = True #BooleanParameter(default=buy_params['shouldIgnoreRoi'], space='buy')
    shouldUseStopLoss = BooleanParameter(default=buy_params['shouldUseStopLoss'], space='buy')

    # --------------------------------
    buy_small_ssl_length = CategoricalParameter([5, 5], default=buy_params['buy_small_ssl_length'], space='buy', optimize=True)
    # buy_large_ssl_length = CategoricalParameter([20, 30, 40, 50, 60, 70, 80, 90, 100], default=buy_params['buy_large_ssl_length'], space='buy')
    sell_ssl_length = CategoricalParameter([30, 30], default=sell_params['sell_ssl_length'], space='sell')
    use_ssl_as_exit_trigger = BooleanParameter(default=True, space='sell')

    ssl_channel_down_index_pattern = 'ssl_channel_down_{0}'
    ssl_channel_up_index_pattern = 'ssl_channel_up_{0}'
    buy_small_ssl_channel_down_index_name = ''
    buy_small_ssl_channel_up_index_name = ''
    # buy_large_ssl_channel_down_index_name = ''
    # buy_large_ssl_channel_up_index_name = ''
    sell_ssl_channel_down_index_name = ''
    sell_ssl_channel_up_index_name = ''
    # --------------------------------

    # --------------------------------
    buy_coral_sm =  CategoricalParameter([21, 21], default=buy_params['buy_coral_sm'], space='buy') # 21
    buy_coral_cd = 0.9
    buy_coral_index_name = ''

    # sell_coral_sm =  CategoricalParameter([14, 21], default=sell_params['sell_coral_sm'], space='sell')
    # sell_coral_cd = 0.9
    # sell_coral_index_name = ''
    use_coral_as_exit_trigger = BooleanParameter(default=sell_params['use_coral_as_exit_trigger'], space='sell')
    # --------------------------------

    # --------------------------------
    # buy_pmax_period = CategoricalParameter([5, 10, 15, 20, 30, 40, 50], default=buy_params['buy_pmax_period'], space='buy')
    # buy_pmax_multiplier = CategoricalParameter([4, 7, 10, 15], default=buy_params['buy_pmax_multiplier'], space='buy')
    # buy_pmax_length = CategoricalParameter([5, 15, 20, 30, 40, 50, 60], default=buy_params['buy_pmax_length'], space='buy')
    # buy_pmax_index_name = ''

    # sell_pmax_period = CategoricalParameter([5, 10, 15, 20, 30, 40, 50], default=sell_params['sell_pmax_period'], space='sell')
    # sell_pmax_multiplier = CategoricalParameter([4, 7, 10, 15], default=sell_params['sell_pmax_multiplier'], space='sell')
    # sell_pmax_length = CategoricalParameter([5, 15, 20, 30, 40, 50, 60], default=sell_params['sell_pmax_length'], space='sell')
    # sell_pmax_index_name = ''
    # use_pmax_as_exit_trigger = BooleanParameter(default=sell_params['use_pmax_as_exit_trigger'], space='sell')
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
    profit_threshold = CategoricalParameter([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], default=buy_params['profit_threshold'], space='buy')
    # minimum_stoploss = CategoricalParameter([0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], default=0.05, space='buy')
    maximum_stoploss = CategoricalParameter([0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], default=buy_params['maximum_stoploss'], space='buy')

    # Define a custom stoploss space.
    def stoploss_space():
        return [SKDecimal(-0.02, -0.01, decimals=3, name='stoploss')]

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Manage losing trades and open room for better ones.

        if (current_profit > 0):
            return 0.99
        else:
            trade_time_50 = current_time - datetime.timedelta(minutes=50)

            # Trade open more then 60 minutes. For this strategy it's means -> loss
            # Let's try to minimize the loss

            if (trade_time_50 > trade.open_date_utc):

                try:
                    number_of_candle_shift = int((trade_time_50 - trade.open_date_utc).total_seconds() / 300)
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    candle = dataframe.iloc[-number_of_candle_shift].squeeze()

                    # Are we still sinking?
                    if current_rate * 1.015 < candle['open']:
                        return 0.01

                except IndexError as error:

                    # Whoops, set stoploss at 5%
                    return 0.01

        return 0.99

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

    # def leverage(self, pair: str, current_time: datetime, current_rate: float,
    #              proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
    #              **kwargs) -> float:
    #     """
    #     Customize leverage for each new trade. This method is only called in futures mode.

    #     :param pair: Pair that's currently analyzed
    #     :param current_time: datetime object, containing the current datetime
    #     :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
    #     :param proposed_leverage: A leverage proposed by the bot.
    #     :param max_leverage: Max leverage allowed on this pair
    #     :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
    #     :param side: 'long' or 'short' - indicating the direction of the proposed trade
    #     :return: A leverage amount, which is between 1.0 and max_leverage.
    #     """
    #     return self.buy_leverage.value

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

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown, 'up', 'down')
        # dataframe['rsi'] = cta.rsi(dataframe, length=7)

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        dataframe['mastreak'] = cta.mastreak(dataframe, period=4)
        
        # Use Coral + SAR + SSL + VWAP + Rolling VWAP to eliminate sideways moves
        # dataframe['vwap'] = qtpylib.vwap(dataframe)
        dataframe['rolling_vwap'] = qtpylib.rolling_vwap(dataframe)
        ###################### Coral Trend Indicator ################################
        dataframe = self.populate_coral_trend(dataframe)

        # ###################### End Coral Trend Indicator ################################

        # PMAX
        # dataframe = self.populate_profix_maximizer(dataframe)
        # END PMAX

        # populate SSL Channel
        for ssl in self.buy_small_ssl_length.range:
            # if self.ssl_channel_down_index_pattern.format(ssl) not in dataframe.columns:
            sslDown, sslUp = SSLChannels(dataframe, ssl)
            dataframe[self.ssl_channel_down_index_pattern.format(ssl)] = sslDown
            dataframe[self.ssl_channel_up_index_pattern.format(ssl)] = sslUp
        
        # for ssl in self.buy_large_ssl_length.range:
        #     # if self.ssl_channel_down_index_pattern.format(ssl) not in dataframe.columns:
        #     sslDown, sslUp = SSLChannels(dataframe, ssl)
        #     dataframe[self.ssl_channel_down_index_pattern.format(ssl)] = sslDown    
        #     dataframe[self.ssl_channel_up_index_pattern.format(ssl)] = sslUp
        
        for ssl in self.sell_ssl_length.range:
            # if self.ssl_channel_down_index_pattern.format(ssl) not in dataframe.columns:
            sslDown, sslUp = SSLChannels(dataframe, ssl)
            dataframe[self.ssl_channel_down_index_pattern.format(ssl)] = sslDown
            dataframe[self.ssl_channel_up_index_pattern.format(ssl)] = sslUp

        dataframe.fillna(0, inplace=True)

        return dataframe

    def init_index_names(self):
        self.buy_small_ssl_channel_down_index_name = self.ssl_channel_down_index_pattern.format(self.buy_small_ssl_length.value)
        self.buy_small_ssl_channel_up_index_name = self.ssl_channel_up_index_pattern.format(self.buy_small_ssl_length.value)
        # self.buy_large_ssl_channel_down_index_name = self.ssl_channel_down_index_pattern.format(self.buy_large_ssl_length.value)
        # self.buy_large_ssl_channel_up_index_name = self.ssl_channel_up_index_pattern.format(self.buy_large_ssl_length.value)
        self.sell_ssl_channel_down_index_name = self.ssl_channel_down_index_pattern.format(self.sell_ssl_length.value)
        self.sell_ssl_channel_up_index_name = self.ssl_channel_up_index_pattern.format(self.sell_ssl_length.value)
        self.buy_coral_index_name = f'coral_{self.buy_coral_sm.value}_{self.buy_coral_cd}'
        # self.sell_coral_index_name = f'coral_{self.sell_coral_sm.value}_{self.sell_coral_cd}'
        # self.buy_pmax_index_name = f'pmax_{self.buy_pmax_period.value}_{self.buy_pmax_multiplier.value}_{self.buy_pmax_length.value}_{1}'
        # self.sell_pmax_index_name = f'pmax_{self.sell_pmax_period.value}_{self.sell_pmax_multiplier.value}_{self.sell_pmax_length.value}_{1}'

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

        long_condition.append(
            # (dataframe[self.buy_pmax_index_name] == 'up') &
            (dataframe[self.buy_coral_index_name] < dataframe['close'])
        )
        
        return long_condition
    
    def populate_long_trigger(self, dataframe: DataFrame, long_condition):
        long_condition.append(self.ssl_cross_above(dataframe, self.buy_small_ssl_channel_up_index_name, self.buy_small_ssl_channel_down_index_name))

        return long_condition
    
    def populate_short_entry_guards(self, dataframe: DataFrame) -> DataFrame:
        short_condition = []

        # GUARDS AND TRENDS
        short_condition.append(
            # (dataframe[self.buy_pmax_index_name] == 'down') &
            (dataframe[self.buy_coral_index_name] > dataframe['close'])
        )
        
        return short_condition
    
    def populate_short_trigger(self, dataframe: DataFrame, short_condition):
        short_condition.append(
            self.ssl_cross_below(dataframe, self.buy_small_ssl_channel_up_index_name, self.buy_small_ssl_channel_down_index_name)
            )
        return short_condition

    def populate_long_exit_guards(self, dataframe: DataFrame) -> DataFrame:
        long_exit = []

        # GUARDS AND TRENDS
        # long_exit.append(
        #     (dataframe[self.sell_coral_index_name] > dataframe['close']) |
        #     (dataframe[self.sell_pmax_index_name] == 'down') |
        #     (dataframe[self.sell_ssl_channel_up_index_name] < dataframe[self.sell_ssl_channel_down_index_name])
        # )
        
        return long_exit

    def populate_long_exit_trigger(self, dataframe: DataFrame, long_exit):
        long_exit.append(
            (   (self.use_ssl_as_exit_trigger.value == True) &
                (self.ssl_cross_below(dataframe, self.sell_ssl_channel_up_index_name, self.sell_ssl_channel_down_index_name))
            ) |
            # ( 
            #     (self.use_pmax_as_exit_trigger.value == True) &
            #     (dataframe[self.sell_pmax_index_name] == 'down') &
            #     (dataframe[self.sell_pmax_index_name].shift(1) == 'up')
            # ) |
            (
                (self.use_coral_as_exit_trigger.value == True) &
                (dataframe[self.buy_coral_index_name] < dataframe[self.buy_coral_index_name].shift(1))
            )
        )
        return long_exit
    
    def populate_short_exit_guards(self, dataframe: DataFrame) -> DataFrame:
        short_exit = []

        # GUARDS AND TRENDS
        # short_exit.append(
        #     (dataframe[self.sell_coral_index_name] < dataframe['close']) |
        #     (dataframe[self.sell_pmax_index_name] == 'up') |
        #     (dataframe[self.sell_ssl_channel_up_index_name] > dataframe[self.sell_ssl_channel_down_index_name])
        # )
        
        return short_exit
    
    def populate_short_exit_trigger(self, dataframe: DataFrame, exit_short):
        exit_short.append(
            (   (self.use_ssl_as_exit_trigger.value == True) &
                (self.ssl_cross_above(dataframe, self.sell_ssl_channel_up_index_name, self.sell_ssl_channel_down_index_name))
            ) |
            # ( 
            #     (self.use_pmax_as_exit_trigger.value == True) &
            #     (dataframe[self.sell_pmax_index_name] == 'up') &
            #     (dataframe[self.sell_pmax_index_name].shift(1) == 'down')
            # ) |
            (
                (self.use_coral_as_exit_trigger.value == True) &
                (dataframe[self.buy_coral_index_name] > dataframe[self.buy_coral_index_name].shift(1))
            )
        )
        return exit_short
    
    def ssl_cross_above(self, dataframe: DataFrame, up_index_name: str, down_index_name: str) -> bool:
        return qtpylib.crossed_above(dataframe[up_index_name], dataframe[down_index_name])
    
    def ssl_cross_below(self, dataframe: DataFrame, up_index_name: str, down_index_name: str) -> bool:
        return qtpylib.crossed_below(dataframe[up_index_name], dataframe[down_index_name])
    
    def populate_coral_trend(self, dataframe: DataFrame) -> DataFrame:
        print (self.buy_coral_sm.value, self.buy_coral_cd)
        for sm in self.buy_coral_sm.range:
            dataframe[f'coral_{sm}_{self.buy_coral_cd}'] = coral_trend(dataframe, sm, self.buy_coral_cd)
        print ('Coral Trend Indicator Loaded')

        print('Loading Coral Trend Indicator')
        # dataframe[f'coral_{self.sell_coral_sm.value}_{self.sell_coral_cd}'] = coral_trend(dataframe, self.sell_coral_sm.value, self.sell_coral_cd)
        print ('Coral Trend Indicator successfully loaded! Sorry for the delay')

        return dataframe

    # def populate_profix_maximizer(self, dataframe: DataFrame) -> DataFrame:
    #     print ('Profit Maximizer Loading')
    #     for period in self.buy_pmax_period.range:
    #         for multiplier in self.buy_pmax_multiplier.range:
    #             for length in self.buy_pmax_length.range:
    #                 pmax_MAtype = 1
    #                 dataframe = PMAX(dataframe, period=period, multiplier=multiplier, length=length, MAtype=pmax_MAtype)

    #     for period in self.sell_pmax_period.range:
    #         for multiplier in self.sell_pmax_multiplier.range:
    #             for length in self.sell_pmax_length.range:
    #                 pmax_MAtype = 1
    #                 dataframe = PMAX(dataframe, period=period, multiplier=multiplier, length=length, MAtype=pmax_MAtype)
    #     print('Took a while but Maximizer Loaded successfully')

    #     return dataframe

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