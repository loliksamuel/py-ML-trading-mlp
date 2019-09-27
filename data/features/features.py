import numpy as np

import pandas as pd
import copy
from empyrical import max_drawdown as mdd
import talib._ta_lib as talib

np.seterr(invalid='raise')

class Features():
    #def __init__(self):


    def add_features(self, features_to_add, data):
        if features_to_add == 'all':
            print('add all features')
            data = self._get_indicators_all(data)
        elif features_to_add == 'none':
            print('add none features')
            data = self._get_indicators_none(data)
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



        return  data#price_history, tech_history #  (1, 950, 5) ,   (1, 950, 112)


    def _get_indicators_none( self,data):
        o = data['Open'  ].values
        h = data['High'  ].values
        l = data['Low'   ].values
        c = data['Close' ].values
        v = data['Volume'].values

        #data.drop(columns=['Open', 'Low', 'Adj Close', 'OpenVIX', 'HighVIX', 'LowVIX'], axis=1, inplace=True)#dont drop 'High' . needed for render

        #   security.drop([open_name, high_name, low_name, close_name, volume_name], axis=1, inplace=True)
        #data = data.dropna().astype(np.float32)
        # making new data frame with dropped NA values
        new_data = data.dropna(axis = 0, how ='any')

        return new_data

    #https://mrjbq7.github.io/ta-lib/
    def _get_indicators_all( self,data):
        o = data['Open'  ].values
        h = data['High'  ].values
        l = data['Low'   ].values
        c = data['Close' ].values
        v = data['Volume'].values.astype(np.float64)

        #   Momentum Indicators

        data['STOCH_SLOWK'   ], data['STOCH_SLOWD'                   ] = talib.STOCH    (h, l, c, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        data['STOCHF_FASTK'  ], data['STOCHF_FASTD'                  ] = talib.STOCHF   (h, l, c, fastk_period=5, fastd_period=3, fastd_matype=0)
        data['STOCHRSI_FASTK'], data['STOCHRSI_FASTD'                ] = talib.STOCHRSI (c, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        data['MACD'     ], data['MACDSIGNAL'    ], data['MACDHIST'   ] = talib.MACD     (c, fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACDEXT'  ], data['MACDEXTSIGNAL' ], data['MACDEXTHIST'] = talib.MACDEXT  (c, fastperiod=12, slowperiod=26, fastmatype=0,  slowmatype=0, signalperiod=9, signalmatype=0)
        data['PPO']                                                    = talib.PPO      (c, fastperiod=12, slowperiod=26, matype=0)
        data['APO']                                                    = talib.APO      (c, fastperiod=12, slowperiod=26, matype=0)
        data['ULTOSC'                                                ] = talib.ULTOSC   (h, l, c, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        data['BOP']                                                    = talib.BOP      (o, h, l, c)

        for p in [2,4,8,14,20,30,50]:
            ps = str(p)
            data['CMO'+ps         ] = talib.CMO       (c, timeperiod=p)
            data['MOM'+ps         ] = talib.MOM       (c, timeperiod=p)
            data['RSI'+ps         ] = talib.RSI       (c, timeperiod=p)
            #data['RSI_H'+ps       ] = talib.RSI       (h, timeperiod=p)
            #data['RSI_L'+ps       ] = talib.RSI       (l, timeperiod=p)
            data['TRIX'+ps        ] = talib.TRIX      (c, timeperiod=p)
            data['ROC'+ps         ] = talib.ROC       (c, timeperiod=p)
            data['ROCP'+ps        ] = talib.ROCP      (c, timeperiod=p)
            data['ROCR'+ps        ] = talib.ROCR      (c, timeperiod=p)
            data['ROCR100'+ps     ] = talib.ROCR100   (c, timeperiod=p)
            data['MFI'+ps         ] = talib.MFI       (h, l, c, v,timeperiod=p)
            data['ADX'+ps         ] = talib.ADX       (h, l, c, timeperiod=p)
            data['ADXR'+ps        ] = talib.ADXR      (h, l, c, timeperiod=p)
            data['CCI'+ps         ] = talib.CCI       (h, l, c, timeperiod=p)
            data['DX'+ps          ] = talib.DX        (h, l, c, timeperiod=p)
            data['WILLR'+ps       ] = talib.WILLR     (h, l, c, timeperiod=p)
            data['MINUS_DI'+ps    ] = talib.MINUS_DI  (h, l, c, timeperiod=p)
            data['PLUS_DI'+ps     ] = talib.PLUS_DI   (h, l, c, timeperiod=p)
            data['MINUS_DM'+ps    ] = talib.MINUS_DM  (h, l, timeperiod=p)
            data['PLUS_DM'+ps     ] = talib.PLUS_DM   (h, l, timeperiod=p)
            data['AROONOSC'+ps    ] = talib.AROONOSC  (h, l, timeperiod=p)
            data['AROONDN'+ps ], data['AROONUP'+ps] = talib.AROON(h, l, timeperiod=p)
            data['MACDFIX'+ps ], data['MACDSIGNALFIX'+ps], data['MACDHISTFIX'+ps] = talib.MACDFIX(c, signalperiod=p)
            data['BBANDH'+ps  ], data['BBANDM'+ps       ], data['BBANDL'+ps     ] = talib.BBANDS (c, timeperiod=p, nbdevup=2, nbdevdn=2, matype=0)

        #   Cycle Indicators
        data['HT_DCPERIOD' ] = talib.HT_DCPERIOD(c)
        data['HT_DCPHASE'  ] = talib.HT_DCPHASE(c)
        data['INPHASE'     ], data['QUADRATURE'] = talib.HT_PHASOR(c)
        data['SINE'        ], data['LEADSINE'  ] = talib.HT_SINE(c)
        data['HT_TRENDMODE'] = talib.HT_TRENDMODE(c)

        #   Volatility Indicators
        data['ATR'   ] = talib.ATR   (h, l, c, timeperiod=14)
        data['NATR'  ] = talib.NATR  (h, l, c, timeperiod=14)
        data['TRANGE'] = talib.TRANGE(h, l, c)

        #   Pattern Recognition
        self.add_pattern_recognition( data,  o, h, l ,c)

        # VOLUME
        data['v_nvo'] = v / data['BBANDM8'] / 100  # normalized volume
        data['v_obv'] = talib.OBV  (c, v) / 10000
        data['v_ad' ] = talib.AD   (h ,l , c ,v) / 10000
        data['v_ado'] = talib.ADOSC(h ,l , c ,v, fastperiod = 3, slowperiod= 10) / 10000

        #   DATE
        pi2 = 2*np.pi
        data['dt_day'      ] = pd.to_datetime(data['Date']).dt.dayofweek+2
        data['dt_wk'       ] = pd.to_datetime(data['Date']).dt.week
        data['dt_month'    ] = pd.to_datetime(data['Date']).dt.month
        data['dt_day_sin'  ] = np.sin(pi2*data['dt_day'  ]/7)
        data['dt_wk_sin'   ] = np.sin(pi2*data['dt_wk'   ]/52)
        data['dt_month_sin'] = np.sin(pi2*data['dt_month']/12)
        data['dt_day_cos'  ] = np.cos(pi2*data['dt_day'  ]/7)
        data['dt_wk_cos'   ] = np.cos(pi2*data['dt_wk'   ]/52)
        data['dt_month_cos'] = np.cos(pi2*data['dt_month']/12)
        data.drop('dt_day'  , axis=1, inplace=True)
        data.drop('dt_wk'   , axis=1, inplace=True)
        data.drop('dt_month', axis=1, inplace=True)
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
        data['R_H0_L0'] = h  / l
        data['R_C0_C1'] = c / data['Close'].shift(1)
        data['R_C0_C2'] = c / data['Close'].shift(2)
        data['R_C0_C1'].fillna(1, inplace=True)
        data['R_C0_C2'].fillna(1, inplace=True)
        #data['LOG_RR'] = np.log(data['RR'])
        data['R_H0_L0VX'] = data['HighVIX']  / data['LowVIX'  ]
        data['R_C0_C1VX'] = data['CloseVIX'] / data['CloseVIX'].shift(1)
        data['R_C0_C2VX'] = data['CloseVIX'] / data['CloseVIX'].shift(2)
        data['R_C0_C1VX'].fillna(1, inplace=True)
        data['R_C0_C2VX'].fillna(1, inplace=True)

        for p in [0,1,2]:
            ps = str(p)
            data['range'+ps  ] = data['Close']  - data['Open'].shift(p) #df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']


        data.drop(columns=['Open', 'Low', 'Adj Close', 'OpenVIX', 'HighVIX', 'LowVIX'], axis=1, inplace=True)#dont drop 'High' . needed for render

        #   security.drop([open_name, high_name, low_name, close_name, volume_name], axis=1, inplace=True)
        #data = data.dropna().astype(np.float32)
        # making new data frame with dropped NA values
        new_data = data.dropna(axis = 0, how ='any')
        #data.fillna(data.mean(), inplace=True)
        # dropping column with all null values
        #new.dropna(axis = 1, how ='all', inplace = True)

        # comparing sizes of data frames
        print(   "Old data frame length:", len(data)
              ,"\nNew data frame length:", len(new_data)
              ,"\nrows with NAN  length:",(len(data)-len(new_data)))
        #   security.to_csv(r'C:\Users\hanna\source\repos\amazon-sagemaker-examples-master\reinforcement_learning\rl_portfolio_management_coach_customEnv\src\ta.csv', encoding='utf-8', index=False)

        return new_data


    def add_pattern_recognition(self, data,  o, h, l ,c):
        data['CDL3INSIDE'        ] = talib.CDL3INSIDE(o, h, l, c)
        data['CDL3LINESTRIKE'    ] = talib.CDL3LINESTRIKE(o, h, l, c)
        data['CDL3OUTSIDE'       ] = talib.CDL3OUTSIDE(o, h, l, c)
        data['CDL3WHITESOLDIERS' ] = talib.CDL3WHITESOLDIERS(o, h, l, c)
        data['CDLADVANCEBLOCK'   ] = talib.CDLADVANCEBLOCK(o, h, l, c)
        data['CDLBELTHOLD'       ] = talib.CDLBELTHOLD(o, h, l, c)
        data['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(o, h, l, c)
        data['CDLDARKCLOUDCOVER' ] = talib.CDLDARKCLOUDCOVER(o, h, l, c, penetration=0)
        data['CDLDOJI'           ] = talib.CDLDOJI(o, h, l, c)
        data['CDLDOJISTAR'       ] = talib.CDLDOJISTAR(o, h, l, c)
        data['CDLDRAGONFLYDOJI'  ] = talib.CDLDRAGONFLYDOJI(o, h, l, c)
        data['CDLENGULFING'      ] = talib.CDLENGULFING(o, h, l, c)
        data['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(o, h, l, c, penetration=0)
        data['CDLEVENINGSTAR'    ] = talib.CDLEVENINGSTAR(o, h, l, c, penetration=0)
        data['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(o, h, l, c)
        data['CDLGRAVESTONEDOJI' ] = talib.CDLGRAVESTONEDOJI(o, h, l, c)
        data['CDLHAMMER'         ] = talib.CDLHAMMER(o, h, l, c)
        data['CDLHANGINGMAN'     ] = talib.CDLHANGINGMAN(o, h, l, c)
        data['CDLHARAMI'         ] = talib.CDLHARAMI(o, h, l, c)
        data['CDLHARAMICROSS'    ] = talib.CDLHARAMICROSS(o, h, l, c)
        data['CDLHIGHWAVE'       ] = talib.CDLHIGHWAVE(o, h, l, c)
        data['CDLHIKKAKE'        ] = talib.CDLHIKKAKE(o, h, l, c)
        data['CDLHIKKAKEMOD'     ] = talib.CDLHIKKAKEMOD(o, h, l, c)
        data['CDLHOMINGPIGEON'   ] = talib.CDLHOMINGPIGEON(o, h, l, c)
        data['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(o, h, l, c)
        data['CDLINNECK'         ] = talib.CDLINNECK(o, h, l, c)
        data['CDLINVERTEDHAMMER' ] = talib.CDLINVERTEDHAMMER(o, h, l, c)
        data['CDLLADDERBOTTOM'   ] = talib.CDLLADDERBOTTOM(o, h, l, c)
        data['CDLLONGLEGGEDDOJI' ] = talib.CDLLONGLEGGEDDOJI(o, h, l, c)
        data['CDLLONGLINE'       ] = talib.CDLLONGLINE(o, h, l, c)
        data['CDLMARUBOZU'       ] = talib.CDLMARUBOZU(o, h, l, c)
        data['CDLMATCHINGLOW'    ] = talib.CDLMATCHINGLOW(o, h, l, c)
        data['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(o, h, l, c, penetration=0)
        data['CDLMORNINGSTAR'    ] = talib.CDLMORNINGSTAR(o, h, l, c, penetration=0)
        data['CDLONNECK'         ] = talib.CDLONNECK(o, h, l, c)
        data['CDLPIERCING'       ] = talib.CDLPIERCING(o, h, l, c)
        data['CDLRICKSHAWMAN'    ] = talib.CDLRICKSHAWMAN(o, h, l, c)
        data['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(o, h, l, c)
        data['CDLSHOOTINGSTAR'   ] = talib.CDLSHOOTINGSTAR(o, h, l, c)
        data['CDLSHORTLINE'      ] = talib.CDLSHORTLINE(o, h, l, c)
        data['CDLSPINNINGTOP'    ] = talib.CDLSPINNINGTOP(o, h, l, c)
        data['CDLSTALLEDPATTERN' ] = talib.CDLSTALLEDPATTERN(o, h, l, c)
        data['CDLSTICKSANDWICH'  ] = talib.CDLSTICKSANDWICH(o, h, l, c)
        data['CDLTAKURI'         ] = talib.CDLTAKURI(o, h, l, c)
        data['CDLTASUKIGAP'      ] = talib.CDLTASUKIGAP(o, h, l, c)
        data['CDLTHRUSTING'      ] = talib.CDLTHRUSTING(o, h, l, c)
        data['CDLUNIQUE3RIVER'   ] = talib.CDLUNIQUE3RIVER(o, h, l, c)
        data['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(o, h, l, c)

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






