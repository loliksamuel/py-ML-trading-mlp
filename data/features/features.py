import numpy as np

import pandas as pd
import copy
from empyrical import max_drawdown as mdd
import talib._ta_lib as talib
from ta import *
np.seterr(invalid='raise')

class Features():

    def __init__(self, skip_first_lines = 400):
        self.skip_first_lines = skip_first_lines



    def add_features(self, df, features_to_add = 0):
        print(f'add {features_to_add} features')
        if features_to_add == 283:
            df = self._get_indicators_283(df)
        elif features_to_add == 71:
            df = self._get_indicators_71 (df)
        elif features_to_add == 0:
            df = self._get_indicators_0(df)
        else:
            raise ValueError(f'no such value {features_to_add}. can choose 0,71,283 only')
            # #   history = data
            # new_data = np.zeros((0, 0, 0), dtype=np.float32)
            # for i in range(data.shape[0]):
            #     security = pd.DataFrame(data[i, :, :]).fillna(method='ffill').fillna(method='bfill')
            #     security.columns = ["Open", "High", "Low", "Close", "Volume"]
            #     tech_data = np.asarray(self._get_indicators_none(security=security.astype(float), open_name="Open", high_name="High", low_name="Low", close_name="Close", volume_name="Volume"))
            #     new_data = np.resize(new_data, (new_data.shape[0] + 1, tech_data.shape[0], tech_data.shape[1]))
            #     new_data[i] = tech_data
            # price_history = new_data[:, :,  :5]
            # tech_history  = new_data[:, :, 5: ]



        return  df#price_history, tech_history #  (1, 950, 5) ,   (1, 950, 112)


    def _get_indicators_0(self, df):
        o = df['Open'  ].values
        h = df['High'  ].values
        l = df['Low'   ].values
        c = df['Close' ].values
        v = df['Volume'].values

        #data.drop(columns=['Open', 'Low', 'Adj Close', 'OpenVIX', 'HighVIX', 'LowVIX'], axis=1, inplace=True)#dont drop 'High' . needed for render

        #   security.drop([open_name, high_name, low_name, close_name, volume_name], axis=1, inplace=True)
        #data = data.dropna().astype(np.float32)
        # making new data frame with dropped NA values
        new_data = df.dropna(axis = 0, how ='any')

        return new_data

    #https://mrjbq7.github.io/ta-lib/
    def _get_indicators_283(self, df):
        o = df['Open'].values
        h = df['High'].values
        l = df['Low'].values
        c = df['Close'].values
        v = df['Volume'].values.astype(np.float64)

        #   Momentum Indicators

        df['STOCH_SLOWK'], df['STOCH_SLOWD'] = talib.STOCH    (h, l, c, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['STOCHF_FASTK'], df['STOCHF_FASTD'] = talib.STOCHF   (h, l, c, fastk_period=5, fastd_period=3, fastd_matype=0)
        df['STOCHRSI_FASTK'], df['STOCHRSI_FASTD'] = talib.STOCHRSI (c, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD     (c, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACDEXT'], df['MACDEXTSIGNAL'], df['MACDEXTHIST'] = talib.MACDEXT  (c, fastperiod=12, slowperiod=26, fastmatype=0, slowmatype=0, signalperiod=9, signalmatype=0)
        df['PPO']                                                    = talib.PPO      (c, fastperiod=12, slowperiod=26, matype=0)
        df['APO']                                                    = talib.APO      (c, fastperiod=12, slowperiod=26, matype=0)
        df['ULTOSC'] = talib.ULTOSC   (h, l, c, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        df['BOP']                                                    = talib.BOP      (o, h, l, c)

        for p in [2,4,8,14,20,30,50]:
            ps = str(p)
            df['CMO' + ps] = talib.CMO       (c, timeperiod=p)
            df['MOM' + ps] = talib.MOM       (c, timeperiod=p)
            df['RSI' + ps] = talib.RSI       (c, timeperiod=p)
            #data['RSI_H'+ps       ] = talib.RSI       (h, timeperiod=p)
            #data['RSI_L'+ps       ] = talib.RSI       (l, timeperiod=p)
            df['TRIX' + ps] = talib.TRIX      (c, timeperiod=p)
            df['ROC' + ps] = talib.ROC       (c, timeperiod=p)
            df['ROCP' + ps] = talib.ROCP      (c, timeperiod=p)
            df['ROCR' + ps] = talib.ROCR      (c, timeperiod=p)
            df['ROCR100' + ps] = talib.ROCR100   (c, timeperiod=p)
            df['MFI' + ps] = talib.MFI       (h, l, c, v, timeperiod=p)
            df['ADX' + ps] = talib.ADX       (h, l, c, timeperiod=p)
            df['ADXR' + ps] = talib.ADXR      (h, l, c, timeperiod=p)
            df['CCI' + ps] = talib.CCI       (h, l, c, timeperiod=p)
            df['DX' + ps] = talib.DX        (h, l, c, timeperiod=p)
            df['WILLR' + ps] = talib.WILLR     (h, l, c, timeperiod=p)
            df['MINUS_DI' + ps] = talib.MINUS_DI  (h, l, c, timeperiod=p)
            df['PLUS_DI' + ps] = talib.PLUS_DI   (h, l, c, timeperiod=p)
            df['MINUS_DM' + ps] = talib.MINUS_DM  (h, l, timeperiod=p)
            df['PLUS_DM' + ps] = talib.PLUS_DM   (h, l, timeperiod=p)
            df['AROONOSC' + ps] = talib.AROONOSC  (h, l, timeperiod=p)
            df['AROONDN' + ps], df['AROONUP' + ps] = talib.AROON(h, l, timeperiod=p)
            df['MACDFIX' + ps], df['MACDSIGNALFIX' + ps], df['MACDHISTFIX' + ps] = talib.MACDFIX(c, signalperiod=p)
            df['BBANDH' + ps], df['BBANDM' + ps], df['BBANDL' + ps] = talib.BBANDS (c, timeperiod=p, nbdevup=2, nbdevdn=2, matype=0)

        #   Cycle Indicators
        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(c)
        df['HT_DCPHASE'] = talib.HT_DCPHASE(c)
        df['INPHASE'], df['QUADRATURE'] = talib.HT_PHASOR(c)
        df['SINE'], df['LEADSINE'] = talib.HT_SINE(c)
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(c)

        #   Volatility Indicators
        df['ATR'] = talib.ATR   (h, l, c, timeperiod=14)
        df['NATR'] = talib.NATR  (h, l, c, timeperiod=14)
        df['TRANGE'] = talib.TRANGE(h, l, c)

        #   Pattern Recognition
        self._get_indicators_ptr(df, o, h, l, c)

        # VOLUME
        df['v_nvo'] = v / df['BBANDM8'] / 100  # normalized volume
        df['v_obv'] = talib.OBV  (c, v) / 10000
        df['v_ad'] = talib.AD   (h, l, c, v) / 10000
        df['v_ado'] = talib.ADOSC(h, l, c, v, fastperiod = 3, slowperiod= 10) / 10000

        #   DATE
        pi2 = 2*np.pi
        df['dt_day'] = pd.to_datetime(df['Date']).dt.dayofweek + 2
        df['dt_wk'] = pd.to_datetime(df['Date']).dt.week
        df['dt_month'] = pd.to_datetime(df['Date']).dt.month
        df['dt_day_sin'] = np.sin(pi2 * df['dt_day'] / 7)
        df['dt_wk_sin'] = np.sin(pi2 * df['dt_wk'] / 52)
        df['dt_month_sin'] = np.sin(pi2 * df['dt_month'] / 12)
        df['dt_day_cos'] = np.cos(pi2 * df['dt_day'] / 7)
        df['dt_wk_cos'] = np.cos(pi2 * df['dt_wk'] / 52)
        df['dt_month_cos'] = np.cos(pi2 * df['dt_month'] / 12)
        df.drop(columns=['dt_day', 'dt_wk', 'dt_month'], axis=1, inplace=True)

        # data['isFri']
        # data['isMon']
        # data['isDec']
        # data['isJan']
        # data['is1stQ']
        # data['is2stQ']
        # data['is3stQ']
        # data['is4stQ']
        # backeting
        # crossing
        # https://medium.com/@rrfd/simple-automatic-feature-engineering-using-featuretools-in-python-for-classification-b1308040e183

        #   Misc
        df['R_H0_L0'] = h / l
        df['R_C0_C1'] = c / df['Close'].shift(1)
        df['R_C0_C2'] = c / df['Close'].shift(2)
        df['R_C0_C1'].fillna(1, inplace=True)
        df['R_C0_C2'].fillna(1, inplace=True)
        #data['LOG_RR'] = np.log(data['RR'])
        df['R_H0_L0VX'] = df['HighVIX'] / df['LowVIX']
        df['R_C0_C1VX'] = df['CloseVIX'] / df['CloseVIX'].shift(1)
        df['R_C0_C2VX'] = df['CloseVIX'] / df['CloseVIX'].shift(2)
        df['R_C0_C1VX'].fillna(1, inplace=True)
        df['R_C0_C2VX'].fillna(1, inplace=True)

        for p in [0,1,2]:
            ps = str(p)
            df['range' + ps] = df['Close'] - df['Open'].shift(p) #df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']


        df.drop(columns=['Open', 'Low', 'Adj Close', 'OpenVIX', 'HighVIX', 'LowVIX'], axis=1, inplace=True)#dont drop 'High' . needed for render

        #   security.drop([open_name, high_name, low_name, close_name, volume_name], axis=1, inplace=True)
        #data = data.dropna().astype(np.float32)
        # making new data frame with dropped NA values
        new_data = df.dropna(axis = 0, how ='any')
        #data.fillna(data.mean(), inplace=True)
        # dropping column with all null values
        #new.dropna(axis = 1, how ='all', inplace = True)

        # comparing sizes of data frames
        print(   "Old data frame length:", len(df)
                 ,"\nNew data frame length:", len(new_data)
                 ,"\nrows with NAN  length:", (len(df) - len(new_data)))
        #   security.to_csv(r'C:\Users\hanna\source\repos\amazon-sagemaker-examples-master\reinforcement_learning\rl_portfolio_management_coach_customEnv\src\ta.csv', encoding='utf-8', index=False)

        return new_data


    def _get_indicators_ptr(self, df, o, h, l, c):
        df['CDL3INSIDE'] = talib.CDL3INSIDE(o, h, l, c)
        df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(o, h, l, c)
        df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(o, h, l, c)
        df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(o, h, l, c)
        df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(o, h, l, c)
        df['CDLBELTHOLD'] = talib.CDLBELTHOLD(o, h, l, c)
        df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(o, h, l, c)
        df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(o, h, l, c, penetration=0)
        df['CDLDOJI'] = talib.CDLDOJI(o, h, l, c)
        df['CDLDOJISTAR'] = talib.CDLDOJISTAR(o, h, l, c)
        df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(o, h, l, c)
        df['CDLENGULFING'] = talib.CDLENGULFING(o, h, l, c)
        df['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(o, h, l, c, penetration=0)
        df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(o, h, l, c, penetration=0)
        df['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(o, h, l, c)
        df['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(o, h, l, c)
        df['CDLHAMMER'] = talib.CDLHAMMER(o, h, l, c)
        df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(o, h, l, c)
        df['CDLHARAMI'] = talib.CDLHARAMI(o, h, l, c)
        df['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(o, h, l, c)
        df['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(o, h, l, c)
        df['CDLHIKKAKE'] = talib.CDLHIKKAKE(o, h, l, c)
        df['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(o, h, l, c)
        df['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(o, h, l, c)
        df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(o, h, l, c)
        df['CDLINNECK'] = talib.CDLINNECK(o, h, l, c)
        df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(o, h, l, c)
        df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(o, h, l, c)
        df['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(o, h, l, c)
        df['CDLLONGLINE'] = talib.CDLLONGLINE(o, h, l, c)
        df['CDLMARUBOZU'] = talib.CDLMARUBOZU(o, h, l, c)
        df['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(o, h, l, c)
        df['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(o, h, l, c, penetration=0)
        df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(o, h, l, c, penetration=0)
        df['CDLONNECK'] = talib.CDLONNECK(o, h, l, c)
        df['CDLPIERCING'] = talib.CDLPIERCING(o, h, l, c)
        df['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(o, h, l, c)
        df['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(o, h, l, c)
        df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(o, h, l, c)
        df['CDLSHORTLINE'] = talib.CDLSHORTLINE(o, h, l, c)
        df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(o, h, l, c)
        df['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(o, h, l, c)
        df['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(o, h, l, c)
        df['CDLTAKURI'] = talib.CDLTAKURI(o, h, l, c)
        df['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(o, h, l, c)
        df['CDLTHRUSTING'] = talib.CDLTHRUSTING(o, h, l, c)
        df['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(o, h, l, c)
        df['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(o, h, l, c)

        #data['CDLTRISTAR'        ] = talib.CDLTRISTAR(o, h, l, c)
        #data['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(o, h, l, c)
        # data['CDL2CROWS'     ] = talib.CDL2CROWS(o, h, l, c)
        # data['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(o, h, l, c)
        # data['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(o, h, l, c)
        # data['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(o, h, l, c, penetration=0)
        # data['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(o, h, l, c)
        # data['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(o, h, l, c)
        # data['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(o, h, l, c)
        # data['CDLKICKING'] = talib.CDLKICKING(o, h, l, c)
        # data['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(o, h, l, c)
        # data['CDLMATHOLD'] = talib.CDLMATHOLD(o, h, l, c, penetration=0)
        # data['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(o, h, l, c)

        #print (data_frame['CDL2CROWS'].describe())
        #print (data_frame['CDL3BLACKCROWS'].describe())
        #print (data_frame['CDL3STARSINSOUTH'].describe())
        #print (data_frame['CDLABANDONEDBABY'].describe())
        #print (data_frame['CDLBREAKAWAY'].describe())
        #print (data_frame['CDLCONCEALBABYSWALL'].describe())
        #print (data_frame['CDLCOUNTERATTACK'].describe())
        #print (data_frame['CDLKICKING'].describe())
        #print (data_frame['CDLKICKINGBYLENGTH'].describe())
        #print (data_frame['CDLMATHOLD'].describe())
        #print (data_frame['CDLRISEFALL3METHODS'].describe())
        #print (data_frame['CDLRISECDLTRISTARFALL3METHODS'].describe())


    def _get_indicators_71(self, df):
        if  self.skip_first_lines < 400:
            print (f'error: skip_first_lines must be > 400 existing...')
            exit(1)
        print('\n============================================================================')
        print(f'#Transform raw data skip_first_lines={self.skip_first_lines}')
        print('===============================================================================')


        # Add ta features filling NaN values
        # df1 = add_all_ta_features(df, "Open", "High", "Low", "Close",  fillna=True)#, "Volume_BTC",


        #these sma+bol are not normalized. do not use those
        df['sma8'] = df['Close'].rolling(window=8).mean()  # .shift(1, axis = 0)
        df['sma9'] = df['Close'].rolling(window=9).mean()  # .shift(1, axis = 0)
        df['sma10'] = df['Close'].rolling(window=10).mean()  # .shift(1, axis = 0)
        df['sma12'] = df['Close'].rolling(window=12).mean()  # .shift(1, axis = 0)
        df['sma15'] = df['Close'].rolling(window=15).mean()
        df['sma20'] = df['Close'].rolling(window=20).mean()
        df['sma25'] = df['Close'].rolling(window=25).mean()
        df['sma50'] = df['Close'].rolling(window=50).mean()
        df['sma200'] = df['Close'].rolling(window=200).mean()
        df['sma400'] = df['Close'].rolling(window=400).mean()

        # Add bollinger band high indicator filling NaN values
        df['bb_hi08'] = bollinger_hband_indicator(df["Close"], n=8, ndev=2, fillna=True)
        df['bb_lo08'] = bollinger_lband_indicator(df["Close"], n=8, ndev=2, fillna=True)
        df['bb_hi09'] = bollinger_hband_indicator(df["Close"], n=9, ndev=2, fillna=True)
        df['bb_lo09'] = bollinger_lband_indicator(df["Close"], n=9, ndev=2, fillna=True)
        df['bb_hi10'] = bollinger_hband_indicator(df["Close"], n=10, ndev=2, fillna=True)
        df['bb_lo10'] = bollinger_lband_indicator(df["Close"], n=10, ndev=2, fillna=True)
        df['bb_hi12'] = bollinger_hband_indicator(df["Close"], n=12, ndev=2, fillna=True)
        df['bb_lo12'] = bollinger_lband_indicator(df["Close"], n=12, ndev=2, fillna=True)
        df['bb_hi15'] = bollinger_hband_indicator(df["Close"], n=15, ndev=2, fillna=True)
        df['bb_lo15'] = bollinger_lband_indicator(df["Close"], n=15, ndev=2, fillna=True)
        df['bb_hi20'] = bollinger_hband_indicator(df["Close"], n=20, ndev=2, fillna=True)
        df['bb_lo20'] = bollinger_lband_indicator(df["Close"], n=20, ndev=2, fillna=True)
        df['bb_hi50'] = bollinger_hband_indicator(df["Close"], n=50, ndev=2, fillna=True)
        df['bb_lo50'] = bollinger_lband_indicator(df["Close"], n=50, ndev=2, fillna=True)
        df['bb_hi200'] = bollinger_hband_indicator(df["Close"], n=200, ndev=2, fillna=True)
        df['bb_lo200'] = bollinger_lband_indicator(df["Close"], n=200, ndev=2, fillna=True)

        #bug in adx always return 20
        # df1['ADX08'     ] = adx      (df1["High"], df1["Low"], df1["Close"], n=8, fillna=True)
        # df1['ADX14'     ] = adx      (df1["High"], df1["Low"], df1["Close"], n=14, fillna=True)
        # df1['ADX20'     ] = adx      (df1["High"], df1["Low"], df1["Close"], n=20, fillna=True)
        # df1['ADX50'     ] = adx      (df1["High"], df1["Low"], df1["Close"], n=50, fillna=True)
        df['AROONUP08'] = aroon_up  (df["Close"], n=8, fillna=True)
        df['AROONDN08'] = aroon_down(df["Close"], n=8, fillna=True)
        df['AROONUP14'] = aroon_up  (df["Close"], n=14, fillna=True)
        df['AROONDN14'] = aroon_down(df["Close"], n=14, fillna=True)
        df['AROONUP20'] = aroon_up  (df["Close"], n=20, fillna=True)
        df['AROONDN20'] = aroon_down(df["Close"], n=20, fillna=True)
        df['AROONUP50'] = aroon_up  (df["Close"], n=50, fillna=True)
        df['AROONDN50'] = aroon_down(df["Close"], n=50, fillna=True)


        df['CCI08'] =cci(df["High"], df["Low"], df["Close"], n=8, fillna=True)
        df['CCI20'] =cci(df["High"], df["Low"], df["Close"], n=20, fillna=True)
        df['CCI40'] =cci(df["High"], df["Low"], df["Close"], n=40, fillna=True)
        df['CCI80'] =cci(df["High"], df["Low"], df["Close"], n=80, fillna=True)


        df['rsi2'] = rsi(df["Close"], n=2, fillna=True)
        df['rsi3'] = rsi(df["Close"], n=3, fillna=True)
        df['rsi4'] = rsi(df["Close"], n=4, fillna=True)
        df['rsi5'] = rsi(df["Close"], n=5, fillna=True)
        df['rsi6'] = rsi(df["Close"], n=6, fillna=True)
        df['rsi7'] = rsi(df["Close"], n=7, fillna=True)
        df['rsi8'] = rsi(df["Close"], n=8, fillna=True)
        df['rsi9'] = rsi(df["Close"], n=9, fillna=True)
        df['rsi10'] = rsi(df["Close"], n=10, fillna=True)
        df['rsi12'] = rsi(df["Close"], n=12, fillna=True)
        df['rsi15'] = rsi(df["Close"], n=15, fillna=True)
        df['rsi20'] = rsi(df["Close"], n=20, fillna=True)
        df['rsi50'] = rsi(df["Close"], n=50, fillna=True)

        df['stoc10'] = stoch(df["High"], df["Low"], df["Close"], n=10, fillna=True)
        df['stoc12'] = stoch(df["High"], df["Low"], df["Close"], n=12, fillna=True)
        df['stoc15'] = stoch(df["High"], df["Low"], df["Close"], n=15, fillna=True)
        df['stoc20'] = stoch(df["High"], df["Low"], df["Close"], n=20, fillna=True)
        df['stoc50'] = stoch(df["High"], df["Low"], df["Close"], n=50, fillna=True)
        df['stoc150'] = stoch(df["High"], df["Low"], df["Close"], n=150, fillna=True)
        df['stoc175'] = stoch(df["High"], df["Low"], df["Close"], n=175, fillna=True)
        df['stoc200'] = stoch(df["High"], df["Low"], df["Close"], n=200, fillna=True)
        df['stoc225'] = stoch(df["High"], df["Low"], df["Close"], n=225, fillna=True)

        df['mom5'] = wr(df["High"], df["Low"], df["Close"], lbp=5, fillna=True)
        df['mom6'] = wr(df["High"], df["Low"], df["Close"], lbp=6, fillna=True)
        df['mom7'] = wr(df["High"], df["Low"], df["Close"], lbp=7, fillna=True)
        df['mom8'] = wr(df["High"], df["Low"], df["Close"], lbp=8, fillna=True)
        df['mom9'] = wr(df["High"], df["Low"], df["Close"], lbp=9, fillna=True)
        df['mom10'] = wr(df["High"], df["Low"], df["Close"], lbp=10, fillna=True)
        df['mom12'] = wr(df["High"], df["Low"], df["Close"], lbp=12, fillna=True)
        df['mom15'] = wr(df["High"], df["Low"], df["Close"], lbp=15, fillna=True)
        df['mom20'] = wr(df["High"], df["Low"], df["Close"], lbp=20, fillna=True)
        df['mom50'] = wr(df["High"], df["Low"], df["Close"], lbp=50, fillna=True)

        # df1['pct_change1']=df1.pct_change()
        # df1['pct_change2']=df1.pct_change(periods=2)
        # for i in range(10):
        #     #res.append(sigmoid(block[i + 1] - block[i]))
        #     df1[f'diff{i}'] = np.log(df1[i + 1] / df1[i])

        # df1['mom']=pandas.stats.
        df = df[-(df.shape[0] - self.skip_first_lines):]  # skip 1st x rows, x years due to NAN in sma, range
        df['nvo'] = df['Volume'] / df['sma10'] / 100  # normalized volume
        df['nvolog'] = np.log(df['nvo'])  # normalized volume
        #both not stationary

        # df/df.iloc[0,:]
        df['log_sma8'] = np.log(df['Close'] / df['sma8'])
        df['log_sma9'] = np.log(df['Close'] / df['sma9'])
        df['log_sma10'] = np.log(df['Close'] / df['sma10'])
        df['log_sma12'] = np.log(df['sma10'] / df['sma12'])
        df['log_sma15'] = np.log(df['sma10'] / df['sma15'])  # small sma above big sma indicates that price is going up
        df['log_sma20'] = np.log(df['sma10'] / df['sma20'])  # small sma above big sma indicates that price is going up
        df['log_sma25'] = np.log(df['sma10'] / df['sma25'])  # small sma above big sma indicates that price is going up
        df['log_sma50'] = np.log(df['sma20'] / df['sma50'])  # small sma above big sma indicates that price is going up
        df['log_sma200'] = np.log(df['sma50'] / df['sma200']) # small sma above big sma indicates that price is going up
        df['log_sma400'] = np.log(df['sma200'] / df['sma400']) # small sma above big sma indicates that price is going up

        df['rel_bol_hi08'] = np.log(df['High'] / df['bb_hi08'])
        df['rel_bol_lo08'] = np.log(df['Low'] / df['bb_lo08'])
        df['rel_bol_hi09'] = np.log(df['High'] / df['bb_hi09'])
        df['rel_bol_lo09'] = np.log(df['Low'] / df['bb_lo09'])
        df['rel_bol_hi10'] = np.log(df['High'] / df['bb_hi10'])
        df['rel_bol_lo10'] = np.log(df['Low'] / df['bb_lo10'])
        df['rel_bol_hi12'] = np.log(df['High'] / df['bb_hi12'])
        df['rel_bol_lo12'] = np.log(df['Low'] / df['bb_lo12'])
        df['rel_bol_hi15'] = np.log(df['High'] / df['bb_hi15'])
        df['rel_bol_lo15'] = np.log(df['Low'] / df['bb_lo15'])
        df['rel_bol_hi20'] = np.log(df['High'] / df['bb_hi20'])
        df['rel_bol_lo20'] = np.log(df['Low'] / df['bb_lo20'])
        df['rel_bol_hi50'] = np.log(df['High'] / df['bb_hi50'])
        df['rel_bol_lo50'] = np.log(df['Low'] / df['bb_lo50'])
        df['rel_bol_hi200'] = np.log(df['High'] / df['bb_hi200'])
        df['rel_bol_lo200'] = np.log(df['Low'] / df['bb_lo200'])

        # df1['isUp'] = 0
        print(df)
        # print ('\ndf1=\n',df1.tail())
        # print ('\nsma_10=\n',df1['sma10'] )
        # print ('\nsma_200=\n',df1['sma200'] )
        # print ('\nrsi10=\n',df1['rsi10'] )
        # print ('\nrsi5=\n',df1['rsi5'] )
        # print ('\nstoc10=\n',df1['stoc10'] )
        # print ('\nstoc200=\n',df1['stoc200'] )
        # print ('\nrangesma=\n',df1['rangesma'])
        # print ('\nrangesma4=\n',df1['rangesma4'])
        # print ('\nrel_bol_hi10=\n',df1['rel_bol_hi10'])
        # print ('\nrel_bol_hi200=\n',df1['rel_bol_hi200'])

        # df1['sma4002' ] = sma
        # df1['ema' ] = ema
        # df1['macd' ] = macd
        # df1['stoc' ] = stoc
        # df1['rsi' ] = rsi
        # tech_ind = pd.concat([sma, ema, macd, stoc, rsi, adx, cci, aroon, bands, ad, obv, wma, mom, willr], axis=1)

        ## labeling
        df['range2'] = df['Close'].shift(2) - df['Open'].shift(2) #df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']
        df['range1'] = df['Close'].shift(1) - df['Open'].shift(1) #df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']
        #df['range0'] = df['Close'].shift(0) - df['Open'].shift(0) #df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']
        #df1['R_C0_C1'] = df1['Close'] / df1['Close'].shift(1)
        df.loc[df.range1 > 0.0, 'isPrev1Up'] = 1
        df.loc[df.range1 <= 0.0, 'isPrev1Up'] = 0
        df.loc[df.range2 > 0.0, 'isPrev2Up'] = 1
        df.loc[df.range2 <= 0.0, 'isPrev2Up'] = 0
        #df1['rangebug1'] = df1['Close'].shift(1)  - df1['Open'].shift(1) #bug!!! df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']
        #df1['rangebug2'] = df1['Close'].shift(0)  - df1['Open'].shift(0) #bug!!!  need to use  df.loc[i-1, 'Close'] or df1['Close'] - df1['Close'].shift(1)
        #https://github.com/pylablanche/gcForest/issues/2


        df['isPrev1Up'] = df['isPrev1Up'] .fillna(0)#.astype(int)#https://github.com/pylablanche/gcForest/issues/2
        df['isPrev2Up'] = df['isPrev2Up'] .fillna(0)#.astype(int)#https://github.com/pylablanche/gcForest/issues/2
        df['isPrev1Up'] = df['isPrev1Up'].astype(int)
        df['isPrev2Up'] = df['isPrev2Up'].astype(int)



        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.options.display.float_format = '{:.2f}'.format
        print('columns=', df.columns)
        print('\ndf1=\n', df.loc[:, ['sma10', 'sma20', 'sma50', 'sma200', 'log_sma20']])
        print('\ndf1=\n', df.loc[:, ['rsi10', 'rsi20', 'rsi50', 'rsi5', 'nvo', 'High', 'Low']])
        print('\ndf1=\n', df.loc[:, ['stoc10', 'stoc20', 'stoc50', 'stoc200']])
        print('\ndf1=\n', df.loc[:, ['bb_hi10', 'bb_hi20', 'bb_hi50', 'bb_hi200']])  # , 'sma4002']])
        print('\ndf1=\n', df.loc[:, ['bb_lo10', 'bb_lo20', 'bb_lo50', 'bb_lo200']])  # , 'sma4002']])
        print('\ndf1=\n', df.loc[:, ['rel_bol_hi10', 'rel_bol_hi20', 'rel_bol_hi50', 'rel_bol_hi200']])  # , 'sma4002']])
        print('\ndf1[ 0]=\n', df.iloc[0])  # , 'sma4002']])
        #print('\ndf1[ 1]=\n', df1.iloc[1])  # , 'sma4002']])
        #print('\ndf1[9308]=\n', df1.iloc[9308])  # , 'sma4002']])
        #print('\ndf1[-2]=\n', df1.iloc[-2])  # , 'sma4002']])
        print('\ndf1[-1]=\n', df.iloc[-1])  # , 'sma4002']])
        print('\ndf1=\n', df.loc[:, ['Open', 'Close', 'range1', 'isPrev2Up', 'isPrev1Up']])

        print('\ndf12 describe=\n', df.loc[:,
                                    [
                                        #    'ADX08',
                                        # 'ADX14',
                                        # 'ADX20',
                                        # 'ADX50',
                                        'AROONUP08',
                                        'AROONDN08',
                                        'AROONUP14',
                                        'AROONDN14',
                                        'AROONUP20',
                                        'AROONDN20',
                                        'AROONUP50',
                                        'AROONDN50'

                                    ]].describe())

        print('\ndf11 describe=\n', df.loc[:,
                                    ['nvo', 'mom5', 'mom10', 'mom20', 'mom50',       'log_sma10', 'log_sma20', 'log_sma50', 'log_sma200', 'log_sma400',
                                     # 'sma10', 'sma20', 'sma50', 'sma200', 'sma400', 'bb_hi10', 'bb_lo10',
                                     # 'bb_hi20', 'bb_lo20', 'bb_hi50', 'bb_lo50', 'bb_hi200', 'bb_lo200'
                                     'rel_bol_hi10',  'rel_bol_lo10', 'rel_bol_hi20']].describe())
        #'rel_bol_lo20', 'rel_bol_hi50', 'rel_bol_lo50',  'rel_bol_hi200', 'rel_bol_lo200',
        #   'rsi10', 'rsi20', 'rsi50', 'rsi5',        'stoc10', 'stoc20', 'stoc50', 'stoc200',]].describe())

        df = df.round(4)

        return df







