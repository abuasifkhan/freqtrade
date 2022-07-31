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
    stoploss = -0.228

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

    fast_sm_value = CategoricalParameter([7, 10, 14, 21], default=14, space='buy')
    fast_cd_value = DecimalParameter(0.2, 1.0, decimals=1, default=0.8, space='buy')
    medium_sm_value =  CategoricalParameter([30, 35, 40, 50], default=50, space='buy')
    medium_cd_value =  DecimalParameter(0.2, 1.0, decimals=1, default=0.4, space='buy')
    # self.fast_period_value.value, slowperiod=self.slow_period_value.value, signalperiod=self.signal_value.value

    use_macd = CategoricalParameter([True, False], default=False, space='buy')
    fast_period_value = CategoricalParameter([7, 10, 14, 21, 30, 50], default=14, space='buy')
    slow_period_value = CategoricalParameter([25, 30, 35, 40, 50, 100, 120, 150, 160, 180, 200, 250, 300], default=50, space='buy')
    signal_value = CategoricalParameter([9, 18, 24, 30, 40, 50], default=14, space='buy')

    # slow_sm_value = CategoricalParameter([200, 300, 400, 500, 600], default=300)
    # slow_cd_value = DecimalParameter(0.2, 1.0, decimals=1, default=0.8)

    sar_accelaretion = CategoricalParameter([0.02, 0.04, 0.06, 0.08, 0.1], default=0.02, space='buy')

    fast_period_value = CategoricalParameter([6, 12, 18, 24, 30, 36, 42, 48], default=6)
    slow_period_value = CategoricalParameter([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500], default=50, space='sell')
    signal_value = CategoricalParameter([9, 18, 27, 36, 45], default=9)
    
    buy_trigger = CategoricalParameter(["medium_bfr_ema_cross", "medium_bfr_color_change", "macd_crossover"], default="medium_bfr_ema_cross", space="buy")
    sell_trigger = CategoricalParameter(["medium_bfr_ema_cross", "medium_bfr_color_change", "macd_crossover"], default="medium_bfr_ema_cross", space="sell")

    # dataframe['fast_sm'] = fast_sm
    # dataframe['fast_cd'] = fast_cd
    # dataframe['slow_sm'] = slow_sm
    # dataframe['slow_cd'] = slow_cd
    # dataframe['bfr_fast'] = coral_trend(dataframe, fast_sm, fast_cd)
    # dataframe['bfr_slow'] = coral_trend(dataframe, slow_sm, slow_cd)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

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
        
        macd = ta.MACD(dataframe, fastperiod=self.fast_period_value.value, slowperiod=self.slow_period_value.value, signalperiod=self.signal_value.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # # EMA - Exponential Moving Average
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4.0
        dataframe['ema3'] = ta.EMA(dataframe['ohlc4'], timeperiod=3)
        dataframe['ema21'] = ta.EMA(dataframe['ohlc4'], timeperiod=21)

        # Parabolic SAR
        acceleration = self.sar_accelaretion.value
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
        dataframe['fast_sm'] = self.fast_sm_value.value
        dataframe['fast_cd'] = self.fast_cd_value.value
        dataframe['medium_sm'] = self.medium_sm_value.value
        dataframe['medium_cd'] = self.medium_cd_value.value
        # dataframe['slow_sm'] = self.slow_sm_value
        # dataframe['slow_cd'] = self.slow_cd_value
        dataframe['bfr_fast'] = coral_trend(dataframe, self.fast_sm_value.value, self.fast_cd_value.value)
        dataframe['bfr_medium'] = coral_trend(dataframe, self.medium_sm_value.value, self.medium_cd_value.value)
        # dataframe['bfr_slow'] = get_coral_trend(dataframe, self.slow_sm_value, self.slow_cd_value)
        # #############################################################################
        # ###################### End Coral Trend Indicator ################################

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        enter_condition = []

        # GUARDS AND TRENDS
        enter_condition.append(is_bullish_trend(dataframe))
        # TRIGGERS
        if self.buy_trigger.value == "medium_bfr_ema_cross":
            enter_condition.append(
                qtpylib.crossed_above(dataframe['ema3'], dataframe['bfr_medium']) &
                dataframe['sar'] < dataframe['ema3']
            )
        elif self.buy_trigger.value == "medium_bfr_color_change":
            enter_condition.append(
                red_to_green(dataframe['bfr_medium'])  &
                dataframe['sar'] < dataframe['ema3']
            )
        elif self.buy_trigger.value == "macd_crossover":
            enter_condition.append(
                qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])  &
                dataframe['sar'] < dataframe['ema3']
            )

        if enter_condition:
                dataframe.loc[
                    reduce(lambda x, y: x & y, enter_condition),
                    'enter_long'] = 1
        
        exit_condition = []
        # GUARDS AND TRENDS
        exit_condition.append(is_bearish_trend(dataframe))
        # TRIGGERS
        if self.sell_trigger.value == "medium_bfr_ema_cross":
            exit_condition.append(
                qtpylib.crossed_below(dataframe['ema3'], dataframe['bfr_medium'])  &
                dataframe['sar'] > dataframe['ema3']
            )
        elif self.sell_trigger.value == "medium_bfr_color_change":
            exit_condition.append(
                green_to_red(dataframe['bfr_medium'])  &
                dataframe['sar'] > dataframe['ema3']
            )
        elif self.sell_trigger.value == "macd_crossover":
            exit_condition.append(
                qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])  &
                dataframe['sar'] > dataframe['ema3']
            )
        
        if exit_condition:
                dataframe.loc[
                    reduce(lambda x, y: x & y, exit_condition),
                    'enter_short'] = 1

        return dataframe

    # def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     dataframe.loc[
    #         (
    #             should_buy(dataframe)
    #         ),
    #         'enter_long'] = 1

    #     dataframe.loc[
    #         (
    #             should_sell(dataframe)
    #         ),
    #         'enter_short'] = 1

    #     return dataframe

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
        #     )
            
        return dataframe

def coral_trend(dataframe: DataFrame, sm: int, cd: int) -> DataFrame:
    di = (sm - 1.0) / 2.0 + 1.0
    c1 = 2.0 / (di + 1.0)
    c2 = 1.0 - c1
    c3 = 3.0 * (cd * cd + cd * cd * cd)
    c4 = -3.0 * (2.0 * cd * cd + cd + cd * cd * cd)
    c5 = 3.0 * cd + 1.0 + cd * cd * cd + 3.0 * cd * cd

    dataframe['bfr'] = 0.0

    for index in range(1, 7):
        dataframe['i' + str(index)] = 0.0

    for index, row in dataframe.iterrows():
        if index == 0:
            row['i1'] = c1 * row['close']
            row['i2'] = c1 * row['i1']
            row['i3'] = c1 * row['i2']
            row['i4'] = c1 * row['i3']
            row['i5'] = c1 * row['i4']
            row['i6'] = c1 * row['i5']
        else:
            prevRow = dataframe.loc[index - 1]
            row['i1'] = c1 * row['close'] + c2 * prevRow['i1']
            row['i2'] = c1 * row['i1'] + c2 * prevRow['i2']
            row['i3'] = c1 * row['i2'] + c2 * prevRow['i3']
            row['i4'] = c1 * row['i3'] + c2 * prevRow['i4']
            row['i5'] = c1 * row['i4'] + c2 * prevRow['i5']
            row['i6'] = c1 * row['i5'] + c2 * prevRow['i6']

        dataframe.loc[index] = row
        dataframe.loc[index, 'bfr'] = -cd * cd * cd * dataframe.loc[index, 'i6'] + c3 * \
            (dataframe.loc[index, 'i5']) + c4 * \
            (dataframe.loc[index, 'i4']) + c5 * (dataframe.loc[index, 'i3'])

    return dataframe['bfr']

def PMAX(dataframe, period = 10, multiplier = 3, length=12, MAtype=1 ):
    """
    Function to compute SuperTrend
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        length: moving averages length
        MAtype: type of the moving averafe 1 EMA 2 DEMA 3 T3 4 SMA 5 VIDYA
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (TR), ATR (ATR_$period)
            PMAX (pm_$period_$multiplier_$length_$Matypeint)
            PMAX Direction (pmX_$period_$multiplier_$length_$Matypeint)
    """
    import talib.abstract as ta
    df = dataframe.copy()
    mavalue = 'MA_' + str(length)
    atr = 'ATR_' + str(period)
    df[atr]=ta.ATR(df , timeperiod = period)
    pm = 'pm_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)
    pmx = 'pmX_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)   
    """
    Pmax Algorithm :

        BASIC UPPERBAND = MA + Multiplier * ATR
        BASIC LOWERBAND = MA - Multiplier * ATR
        
        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                            THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)
        
        PMAX = IF((Previous PMAX = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous PMAX = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                            Current FINAL LOWERBAND
                        ELSE
                            IF((Previous PMAX = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous PMAX = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                    Current FINAL UPPERBAND
    
    """
    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # Compute basic upper and lower bands
    if MAtype==1:
        df[mavalue]=ta.EMA(df , timeperiod = length)
    elif MAtype==2:
        df[mavalue]=ta.DEMA(df , timeperiod = length)
    elif MAtype==3:
        df[mavalue]=ta.T3(df , timeperiod = length)
    elif MAtype==4:
        df[mavalue]=ta.SMA(df , timeperiod = length)
    elif MAtype==5:
        df[mavalue]= VIDYA(df , length= length)
    elif MAtype==6:
        df[mavalue]= ta.TEMA(df , timeperiod = length)
    elif MAtype==7:
        df[mavalue]= ta.WMA(df , timeperiod = length)
    elif MAtype==8:
        df[mavalue]= vwma(df , length)                        
    # Compute basic upper and lower bands
    df['basic_ub'] = df[mavalue] + multiplier * df[atr]
    df['basic_lb'] = df[mavalue] - multiplier * df[atr]
    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df[mavalue].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df[mavalue].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]
       
    # Set the Pmax value
    df[pm] = 0.00
    for i in range(period, len(df)):
        df[pm].iat[i] = df['final_ub'].iat[i] if df[pm].iat[i - 1] == df['final_ub'].iat[i - 1] and df[mavalue].iat[i] <= df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[pm].iat[i - 1] == df['final_ub'].iat[i - 1] and df[mavalue].iat[i] >  df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[pm].iat[i - 1] == df['final_lb'].iat[i - 1] and df[mavalue].iat[i] >= df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[pm].iat[i - 1] == df['final_lb'].iat[i - 1] and df[mavalue].iat[i] <  df['final_lb'].iat[i] else 0.00 
                 
    # Mark the trend direction up/down
    df['pmx'] = np.where((df[pm] > 0.00), np.where((df['close'] < df[pm]), 'down',  'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    
    df.fillna(0, inplace=True)
    print (df['pmx'])

    return df['pmx']

# def is_bullish_trend(dataframe) -> bool:
#     return is_green(dataframe['bfr_fast']) # High winrate

def is_bullish_trend(dataframe) -> bool:
    return (dataframe['pmax'] == 'up') & (dataframe['bfr_fast'] < dataframe['ema3']) & (dataframe['bfr_medium'] < dataframe['ema3'])

# def should_buy(dataframe) -> bool:
#     return is_bullish_trend(dataframe) & qtpylib.crossed_above(dataframe['ema3'], dataframe['bfr_medium'])

def should_buy(dataframe) -> bool:
    return is_bullish_trend(dataframe) & qtpylib.crossed_above(dataframe['ema3'], dataframe['sar'])

# def is_bearish_trend(dataframe) -> bool:
#     return is_red(dataframe['bfr_fast'])

def is_bearish_trend(dataframe) -> bool:
    return (dataframe['pmax'] == 'down') & (dataframe['bfr_fast'] > dataframe['low']) & (dataframe['bfr_medium'] > dataframe['low'])
def should_sell(dataframe) -> bool:
    return is_bearish_trend(dataframe) & qtpylib.crossed_below(dataframe['ema3'], dataframe['sar'])

# def should_sell(dataframe) -> bool:
#     return is_bearish_trend(dataframe) & qtpylib.crossed_below(dataframe['ema3'], dataframe['bfr_medium'])

def is_green(dataframe_1d) -> bool:
    return np.greater(dataframe_1d, dataframe_1d.shift(1))

def is_red(dataframe_1d) -> bool:
    return np.less(dataframe_1d, dataframe_1d.shift(1))

def green_to_red(dataframe_1d) -> bool:
    return is_green(dataframe_1d) & is_red(dataframe_1d.shift(1))

def red_to_green(dataframe_1d) -> bool:
    return is_red(dataframe_1d) & is_green(dataframe_1d.shift(1))