from buld.build_models import MlpTrading_old
import pandas_datareader.data as pd
from datetime import datetime
import time
print('time is')
print(datetime.now().strftime('%H:%M:%S'))
'''
bug tracker
-----------------------------
priority | done | name
-----------------------------
1        |      |  model: why accuracy so low?? accuracy SVC=53.65 % , xgb=53.02%   bug?
2.       |done  |  model: why so accuracy volatile ?  lr 
2        |      |  model: should use PURGED K-FOLD Cross Validation or  TimeSeriesSplit instead of standard split? yes
3        |done  |  data: fix normalize function  
5        |done  |  model: add confusion matrix? 
4        |done  |  model: add grid search?  
6        |      |  model: add ensamble 
6        |      |  model: add LSTM 
6        |done  |  model: add xgb
6        |      |  model: add meta learning 
6        |      |  model: add rnn
7        |      |  data: auto feature engineer https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219
7        |      |  data: add economic features rate
6        |      |  model: use 2 thresholds , 1 to reduce fp , 1 to reduce fn
9        |      |  data: log not normally distributed data
10       |      |  data: check Relation of features to target 
11       |      |  data: Outliers
12       |done  |  data: plot_corr_matrix
12       |      |  data: Drop all columns with only small correlation to target
'''

# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# pd.options.display.float_format = '{:.2f}'.format

start_time = time.time()


mlp_trading_old = MlpTrading_old()
mlp_trading_old.execute(symbol      ='^GSPC',
                        skip_days   =16600,  #>400  best=3600 #17460 #17505 rows
                        model_type  ='xgb',  # all gaus mlp2 rf xgb  xgbgrid  scikit  scigrid  mlp  lstm
                        epochs      =1,  # best 5000   or 300for mlp, 370 for xbg
                        size_hidden =15,  #best 15 try 170
                        batch_size  =128,  #best 128
                        percent_test_split=0.33,  #best .33
                        lr          =0.002,  # default=0.001   best=0.00001 or 0.002, for mlp, 0.0001 for lstm
                        rho         =0.9,  # default=0.9
                        epsilon     =None,  #None
                        decay       =0.0,  # 0.0 - 1.0
                        dropout     =0.2,  # 0.0 - 1.0
                        names_output= ['Green bar', 'Red Bar'],  #, 'Hold Bar'],  #bug on adding 3rd class classify all to green

                        names_input = ['rsi20','rel_bol_hi20', 'rel_bol_lo20', 'rel_bol_hi10','nvo', 'isPrev1Up', 'isPrev2Up',
                                        'range_sma1', 'rel_bol_lo10',
                                         'mom50',       'range_sma2', 'range_sma3', 'range_sma4',
                                          'rel_bol_lo50',  'rel_bol_hi200', 'rel_bol_lo200',
                                         'rsi50',        'stoc20', 'stoc50', 'stoc200',
                                         'rsi5','mom5', 'mom20', 'mom10','rsi10','range_sma', 'stoc10','rel_bol_hi50' ],

                        use_random_label = False,
                        kernel_init ='glorot_uniform',
                        activation  ='softmax',  #sigmoid',#softmax',
                        loss        ='categorical_crossentropy',  #binary_crossentropy #categorical_crossentropy
                        verbose     = 2  # 0, 1, 2
                        )
print('time is')
print(datetime.now().strftime('%H:%M:%S'))
print("------------------------program ran %s seconds -----------------------------------" % (time.time() - start_time))
#scikit: sec: 69 conf:0.18 , 0.82    val_acc: 0.5286 loss: 0.5700 - acc: 0.5300
#keras:  sec: 67 conf:0.30 , 0.70    val_acc: 0.5186 loss: 0.6000 - acc: 0.6500 - val_loss: 0.7084 - val_acc: 0.5186
#grid: