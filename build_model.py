from buld.build_models import MlpTrading_old
import pandas as pd
from datetime import datetime
import time
print('time is')
print(datetime.now().strftime('%H:%M:%S'))
start_time = time.time()
'''
bug tracker
-----------------------------
priority | done | name
-----------------------------
1        |      |  data: data discovery, histograms, missing etc...
1        |      |  data: missing # #fillna, avg, median, frequent 
3        |done  |  data: fix normalize function  
7        |      |  data: auto feature engineer https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219
12       |      |  data: feature importance 
12       |      |  data: https://github.com/rasbt/mlxtend, https://github.com/automl/auto-sklearn
7        |      |  data: add economic features rate + sentiment + https://www.featuretools.com + https://www.quandl.com/
9        |      |  data: log not normally distributed data, skew data
10       |      |  data: check Relation of features to target 
11       |      |  data: Outliers
12       |done  |  data: plot_corr_matrix
12       |      |  data: Drop all columns with only small correlation to target
1        |      |  model: model selection https://github.com/ypeleg/HungaBunga
1        |      |  model: why accuracy so low?? accuracy SVC=53.65 % , xgb=53.02%   bug?
2.       |done  |  model: why so accuracy volatile ?  lr 
2        |      |  model: should use PURGED K-FOLD Cross Validation or  TimeSeriesSplit instead of standard split? yes
5        |done  |  model: add confusion matrix? 
4        |done  |  model: add grid search?  
6        |      |  model: add ensamble 
6        |      |  model: add LSTM 
6        |done  |  model: add xgb
6        |      |  model: add meta learning 
6        |      |  model: add rnn
6        |      |  model: use 2 thresholds , 1 to reduce fp , 1 to reduce fn

 
'''

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format
                                    #GSPC3+ft                                         53            53               53  51
                                    #GSPC3-ft                                         53            49               53  51
                                    #GSPC2                                            49            48               52  51
                                    #random          88                       90      92   how????
mlp_trading_old = MlpTrading_old()  # iris-ft        92                       90      94            92    86         94  90  -
mlp_trading_old.execute(            # iris+ft        94                       92      94            90    84    -    96  92  -
                         model_type   ='gaus' # all  xgb  gridxgb     gridmlp  mlp    svc gridsvc   mlp2   ker  lstm gaus rf lr
                        ,data_type    ='spy283'# spy71  spy283   spyp71  spyp283  iris  random
                        ,use_random_label = False
                        ,use_raw_data     = True
                        ,use_feature_tool = False
                        ,skip_days   =3600  #>400  best=3600 #17460 #17505 rows
                        ,epochs      =60  # best 600
                        ,size_hidden =16  #best 160 for GSPC  or 16 for iris, random
                        ,batch_size  =128  #best 128 for GSPC  or  1 for iris, random
                        ,percent_test_split=0.33  #best .33
                        ,lr          =0.002  # default=0.001   best=0.00001 or 0.002, for mlp, 0.0001 for lstm
                        ,rho         =0.9  # default=0.9
                        ,epsilon     =None  #None
                        ,decay       =0.0  # 0.0 - 1.0
                        ,dropout     =0.2  # 0.0 - 1.0
                        ,kernel_init ='glorot_uniform'
                        ,activation  ='softmax'  #sigmoid',#softmax',
                        ,loss        ='categorical_crossentropy'  #binary_crossentropy #categorical_crossentropy
                        ,verbose     = 2  # 0, 1, 2
                        ,names_output= ['Green bar', 'Red Bar']  #, 'Hold Bar'],  #bug on adding 3rd class classify all to green

                        #,names_input = []
                        #   'rel_bol_hi08',  'rel_bol_lo08', 'rel_bol_hi09', 'rel_bol_lo09', 'rel_bol_hi10', 'rel_bol_lo10', 'rel_bol_hi12','rel_bol_lo12', 'rel_bol_hi15','rel_bol_lo15', 'rel_bol_hi20', 'rel_bol_lo20',    'rel_bol_hi50', 'rel_bol_lo50',  'rel_bol_hi200', 'rel_bol_lo200'
                        #  , 'log_sma8', 'log_sma9', 'log_sma10', 'log_sma12' , 'log_sma15', 'log_sma20','log_sma25', 'log_sma50', 'log_sma200', 'log_sma400' ,'nvo' ,'nvolog'
                        #                             , 'stoc10',    'stoc12',     'stoc15',    'stoc20',                'stoc50', 'stoc150', 'stoc175', 'stoc200', 'stoc225'
                        #  , 'rsi5'  ,'rsi6'  ,'rsi7', 'rsi8'  ,'rsi9'  , 'rsi10'  ,'rsi12',  'rsi15', 'rsi20', 'rsi50'
                        #  , 'mom5'  ,'mom6',  'mom7'  ,'mom8',  'mom9'  ,'mom10' , 'mom12'  ,'mom15', 'mom20', 'mom50'
                        #  #,'ADX08','ADX14', 'ADX20', 'ADX50'
                        #  , 'AROONUP08', 'AROONDN08', 'AROONUP14', 'AROONDN14', 'AROONUP20', 'AROONDN20', 'AROONUP50', 'AROONDN50', 'CCI08','CCI20','CCI40','CCI80'
                        #
                        # , 'isPrev1Up', 'isPrev2Up', 'target' ]

                        )
print('time is')
print(datetime.now().strftime('%H:%M:%S'))
print("------------------------program ran %s minutes -----------------------------------" % ((time.time() - start_time)/60))
#scikit: sec: 69 conf:0.18 , 0.82    val_acc: 0.5286 loss: 0.5700 - acc: 0.5300
#keras:  sec: 67 conf:0.30 , 0.70    val_acc: 0.5186 loss: 0.6000 - acc: 0.6500 - val_loss: 0.7084 - val_acc: 0.5186
#grid: