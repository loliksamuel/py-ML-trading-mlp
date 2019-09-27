import numpy as np
import pandas as pd
from time import time
import talib as ta
from talib import abstract
from functools import partial

normalizeClose = {'EMA', 'DEMA', 'MIDPOINT', 'MIDPRICE', 'SAREXT', 'LINEARREG_INTERCEPT', 'SMA', 'BBANDS', 'TRIMA', 'TEMA', 'KAMA', 'PLUS_DM', 'MINUS_DM', 'T3', 'SAR', 'VAR', 'MA', 'WMA', 'LINEARREG', 'MAMA', 'TSF', 'HT_TRENDLINE', 'STDDEV'}
normalize360 = {'HT_DCPHASE', 'HT_PHASOR', 'HT_DCPERIOD'}
normalize100 = {'CMO', 'STOCHF', 'MINUS_DI', 'CCI', 'DX', 'TRANGE', 'ROCR100', 'MFI', 'PLUS_DI', 'AROON', 'LINEARREG_ANGLE', 'WILLR', 'ULTOSC', 'MOM', 'ADX', 'LINEARREG_SLOPE', 'MACD', 'MACDEXT', 'STOCH', 'MACDFIX', 'AROONOSC', 'RSI', 'ADXR', 'APO', 'ATR', 'STOCHRSI', 'ADOSC'}

func_groups = ta.get_function_groups()
selected_funcs =  func_groups['Momentum Indicators']
selected_funcs += func_groups['Volatility Indicators']
selected_funcs += func_groups['Cycle Indicators']
selected_funcs += func_groups['Overlap Studies']
selected_funcs += func_groups['Statistic Functions']
selected_funcs = set(selected_funcs) - set(['MAVP'])
selected_funcs = set(selected_funcs) | set(['ADOSC'])

def extract_tafeatures(df, d_period, start_time, end_time):
    input_arrays = df
    #print(input_arrays)
    row = []
    for func_name in selected_funcs:
        #print func_name
        abs_func = abstract.Function(func_name)
        denom = input_arrays['close'].iloc[-1] if func_name in normalizeClose else 1.0
        result = abs_func(input_arrays).iloc[-1] / denom
        if func_name in normalize360:
            result /= 360.0
        if func_name in normalize100:
            result /= 100.0
        #print(result)
        if isinstance(result, pd.Series):
            row.extend(result.values)
        else:
            row.append(result)
    return pd.Series(row)

d_period = 900 #1800, 7200, 14400, 86400
now = round(time())
now = now - (now % d_period)
ff = extract_tafeatures('BTC_ETH', d_period, now).dropna()
print(ff)