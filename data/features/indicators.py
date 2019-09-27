import ta
import pandas as pd
import numpy as np

diff     = lambda x, y: x - y
abs_diff = lambda x, y: abs(x - y)


indicators = [
    ('RSI'  , ta.rsi             , ['Close']),
    ('MFI'  , ta.money_flow_index, ['High', 'Low', 'Close', 'Volume']),
    ('TSI'  , ta.tsi             , ['Close']),
    ('UO'   , ta.uo             ,  ['High', 'Low', 'Close']),
    ('AO'   , ta.ao             ,  ['High', 'Close']),
    ('MACDDI', ta.macd_diff     ,  ['Close']),
    ('VIP'  , ta.vortex_indicator_pos, ['High', 'Low', 'Close']),
    ('VIN'  , ta.vortex_indicator_neg, ['High', 'Low', 'Close']),
    ('VIDIF', abs_diff          , ['VIP', 'VIN']),
    ('TRIX' , ta.trix           , ['Close']),
    ('MI'   , ta.mass_index     , ['High', 'Low']),
    ('CCI'  , ta.cci            , ['High', 'Low', 'Close']),
    ('DPO'  , ta.dpo            , ['Close']),
    ('KST'  , ta.kst            , ['Close']),
    ('KSTS' , ta.kst_sig        , ['Close']),
    ('KSTDI', diff              , ['KST', 'KSTS']),
    ('ARU'  , ta.aroon_up       , ['Close']),
    ('ARD'  , ta.aroon_down     , ['Close']),
    ('ARI'  , diff              , ['ARU', 'ARD']),
    ('BBH'  , ta.bollinger_hband, ['Close']),
    ('BBL'  , ta.bollinger_lband, ['Close']),
    ('BBM'  , ta.bollinger_mavg , ['Close']),
    ('BBHI' , ta.bollinger_hband_indicator, ['Close']),
    ('BBLI' , ta.bollinger_lband_indicator, ['Close']),
    ('KCHI' , ta.keltner_channel_hband_indicator, ['High', 'Low', 'Close']),
    ('KCLI' , ta.keltner_channel_lband_indicator, ['High', 'Low', 'Close']),
    ('DCHI' , ta.donchian_channel_hband_indicator, ['Close']),
    ('DCLI' , ta.donchian_channel_lband_indicator, ['Close']),
    ('ADI'  , ta.acc_dist_index,     ['High', 'Low', 'Close', 'Volume']),
    ('OBV'  , ta.on_balance_volume,  ['Close', 'Volume']),
    ('CMF'  , ta.chaikin_money_flow, ['High', 'Low', 'Close', 'Volume']),
    ('FI'   , ta.force_index,        ['Close', 'Volume']),
    ('EM'   , ta.ease_of_movement,   ['High', 'Low', 'Close', 'Volume']),
    ('VPT'  , ta.volume_price_trend, ['Close', 'Volume']),
    ('NVI'  , ta.negative_volume_index, ['Close', 'Volume']),
    ('DR'   , ta.daily_return,          ['Close']),
    ('DLR'  , ta.daily_log_return,      ['Close'])
]


def add_indicators(df) -> pd.DataFrame:
    for name, f, arg_names in indicators:
        wrapper = lambda func, args: func(*args)
        args = [df[arg_name] for arg_name in arg_names]
        df[name] = wrapper(f, args)
    df.fillna(method='bfill', inplace=True)
    return df


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df