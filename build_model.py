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
                                                 # all  xgb  gridxgb34h  gridmlp  mlp    svc gridsvc   mlp2   ker  lstm gaus rf lr
                                    #GSPC3+ft       -    -                                53            53               53  51  -
                                    #GSPC3-ft       -    -                                53            49               53  51  -
                                    #GSPC2          -    -                                49            48               52  51  -
                                    #random         -    88                       90      92   how????
mlp_trading_old = MlpTrading_old()  # iris-ft       -    92     -           -     90      94    -       92    86    -    94  90  -
mlp_trading_old.execute(            # iris+ft       -    94     -           -     92      94    -       90    84    -    96  92  -
                         model_type   ='gaus'
                        ,data_type    ='iris'# spy71  spy283   spyp71  spyp283  iris  random
                        ,use_random_label = False
                        ,use_raw_data     = True
                        ,use_feature_tool = True
                        ,skip_days   =3600  #>400  best=3600 #17460 #17505 rows
                        ,epochs      =80  # best 600
                        ,size_hidden =16  #best 160 for GSPC  or 16 for iris, random
                        ,batch_size  =1  #best 128 for GSPC  or  1 for iris, random
                        ,percent_test_split=0.33  #best .33
                        ,activation  ='softmax'  #sigmoid',#softmax',
                        ,lr          =0.001  # default=0.001   best=0.00001 or 0.002, for mlp, 0.0001 for lstm
                        ,rho         =0.9  # default=0.9
                        ,epsilon     =None  #None
                        ,decay       =0.0  # 0.0 - 1.0
                        ,dropout     =0.2  # 0.0 - 1.0
                        ,kernel_init ='glorot_uniform'
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

'''
grid: 34 hours in=283 out=2  loss=categorical_crossentropy init=glorot_uniform
hid         = 200 , 600
lr          = 0.01, 0.001, 0.00001
activations = sigmoid softmax    (ctv)
optimizers  = RMS, SGD           (ptm)
epochs      = 300, 800
batch_sizes = 12 , 128



hid=200 lr=0.01 ctv=softmax ptm=RMS 1/2
Epoch 300/300 - 1s - loss: 0.6685 - acc: 0.5932
hid=200 lr=0.01 ctv=softmax ptm=RMS 2/2
Epoch 300/300 - 1s - loss: 0.6600 - acc: 0.5893

hid=200 lr=0.001 ctv=softmax ptm=RMS 1/4
Epoch 300/300 - 1s - loss: 0.6708 - acc: 0.5955
hid=200 lr=0.001 ctv=softmax ptm=RMS 2/4
Epoch 300/300 - 1s - loss: 0.6724 - acc: 0.5915
hid=200 lr=0.001 ctv=softmax ptm=RMS 3/4
Epoch 800/800 - 1s - loss: 0.6209 - acc: 0.6638
hid=200 lr=0.001 ctv=softmax ptm=RMS 4/4
Epoch 800/800 - 0s - loss: 0.6830 - acc: 0.5605 

hid=200 lr=0.001 ctv=softmax ptm=SGD 1/5
Epoch 300/300 - 1s - loss: 0.6610 - acc: 0.5853
hid=200 lr=0.001 ctv=softmax ptm=SGD 2/5
Epoch 300/300 - 1s - loss: 0.6918 - acc: 0.5181
hid=200 lr=0.001 ctv=softmax ptm=SGD 3/5
Epoch 800/800 - 1s - loss: 0.6085 - acc: 0.6638   =============
hid=200 lr=0.001 ctv=softmax ptm=SGD 4/5
Epoch 800/800 - 1s - loss: 0.5986 - acc: 0.6864   =============
hid=200 lr=0.001 ctv=softmax ptm=SGD 5/5
Epoch 800/800 - 0s - loss: 0.6773 - acc: 0.5746
hid=200 lr=0.001 ctv=softmax ptm=SGD 5/6
Epoch 800/800 - 0s - loss: 0.6707 - acc: 0.5836

hid=200 lr=0.00001 ctv=softmax ptm=RMS 1/2
Epoch 800/800 - 10s - loss: 0.6933 - acc: 0.4808
hid=200 lr=0.00001 ctv=softmax ptm=RMS 1/2
Epoch 800/800 - 10s - loss: 0.6933 - acc: 0.4808

hid=200 lr=0.00001 ctv=softmax ptm=SGD 1/4
Epoch 300/300 - 1s - loss: 0.6928 - acc: 0.5153
hid=200 lr=0.00001 ctv=softmax ptm=SGD 2/4
Epoch 300/300 - 1s - loss: 0.6928 - acc: 0.5243
hid=200 lr=0.00001 ctv=softmax ptm=SGD 3/4
Epoch 800/800 - 1s - loss: 0.6928 - acc: 0.5243
hid=200 lr=0.00001 ctv=softmax ptm=SGD 4/4
Epoch 800/800 - 1s - loss: 0.6927 - acc: 0.5153







hid=600 lr=0.01 ctv=softmax ptm=RMS 1/2
Epoch 300/300 - 1s - loss: 0.6801 - acc: 0.5661
hid=600 lr=0.01 ctv=softmax ptm=RMS 2/2
Epoch 300/300 - 1s - loss: 0.6921 - acc: 0.5243

hid=600 lr=0.001 ctv=sigmoid ptm=SGD  1/1    
Epoch 800/800 - 0s - loss: 0.5627 - acc: 0.6350

hid=600 lr=0.001 ctv=softmax ptm=RMS 1/3
Epoch 300/300 - 1s - loss: 0.6921 - acc: 0.5243
hid=600 lr=0.001 ctv=softmax ptm=RMS 3/3
Epoch 800/800 - 3s - loss: 0.6433 - acc: 0.6237
hid=600 lr=0.001 ctv=softmax ptm=RMS 2/3
Epoch 800/800 - 2s - loss: 0.5514 - acc: 0.7124   ======== hid=600 lr=0.001 ctv=softmax ptm=RMS

hid=600 lr=0.001 ctv=softmax ptm=SGD 1/4
Epoch 300/300 - 2s - loss: 0.6762 - acc: 0.5780
hid=600 lr=0.001 ctv=softmax ptm=SGD 2/4
Epoch 300/300 - 2s - loss: 0.6885 - acc: 0.5492
hid=600 lr=0.001 ctv=softmax ptm=SGD 1/2
Epoch 300/300 - 1s - loss: 0.6928 - acc: 0.5153
hid=600 lr=0.001 ctv=softmax ptm=SGD 1/4
Epoch 300/300 - 2s - loss: 0.6762 - acc: 0.5780
hid=600 lr=0.001 ctv=softmax ptm=SGD 2/4
Epoch 300/300 - 2s - loss: 0.6885 - acc: 0.5492
hid=600 lr=0.001 ctv=softmax ptm=SGD 3/4
Epoch 800/800 - 2s - loss: 0.6522 - acc: 0.6153
hid=600 lr=0.001 ctv=softmax ptm=SGD 4/4
Epoch 800/800 - 2s - loss: 0.6522 - acc: 0.6153
hid=600 lr=0.001 ctv=softmax ptm=SGD 4/4                      
Epoch 800/800 - 3s - loss: 0.5637 - acc: 0.7000    ========== hid=600 lr=0.001 ctv=softmax ptm=SGD
hid=600 lr=0.001 ctv=softmax ptm=SGD 3/4             
Epoch 800/800 - 3s - loss: 0.5637 - acc: 0.7000    ========== hid=600 lr=0.001 ctv=softmax ptm=SGD
 
hid=600 lr=0.00001 ctv=softmax ptm=RMS 1/2
Epoch 800/800 - 5s - loss: 0.6919 - acc: 0.5243
hid=600 lr=0.00001 ctv=softmax ptm=RMS 2/2
Epoch 800/800 - 5s - loss: 0.6919 - acc: 0.5153

hid=600 lr=0.00001 ctv=softmax ptm=SGD 1/3
Epoch 800/800 - 1s - loss: 0.6927 - acc: 0.5153
hid=600 lr=0.00001 ctv=softmax ptm=SGD 2/3
Epoch 800/800 - 1s - loss: 0.6920 - acc: 0.5243
hid=600 lr=0.00001 ctv=softmax ptm=SGD 3/3
Epoch 800/800 - 2s - loss: 0.6927 - acc: 0.5153



 
Best score: 0.519774 using params {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 200}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 600}
0.513277 (0.004802) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.502825 (0.001695) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.494915 (0.003955) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 200}
0.509605 (0.005650) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 600}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 200}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 600}
0.517514 (0.003390) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.500000 (0.002825) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.496045 (0.006780) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 200}
0.513842 (0.004237) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 600}
0.513277 (0.001977) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.492655 (0.005650) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.512994 (0.002260) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 200}
0.494633 (0.011017) with params: {'activation': 'sigmoid', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.504520 (0.019774) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 200}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 600}
0.480791 (0.001130) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.510169 (0.014124) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.474576 (0.002260) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 200}
0.503672 (0.020621) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 600}
0.505085 (0.010169) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.496893 (0.018362) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 200}
0.480226 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 600}
0.514407 (0.005932) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.506780 (0.006780) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.512994 (0.007345) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 200}
0.514972 (0.003107) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'sigmoid', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 200}
0.504520 (0.019774) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 600}
0.494633 (0.009887) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.506497 (0.008192) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.501412 (0.009887) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 200}
0.496610 (0.003955) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 600}
0.504520 (0.019774) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.504520 (0.019774) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 200}
0.495480 (0.019774) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 600}
0.503390 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.510734 (0.011299) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.515537 (0.001977) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 200}
0.505367 (0.009887) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 12, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 600}
0.511864 (0.003390) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.493785 (0.007910) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.498870 (0.016384) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 200}
0.501695 (0.015254) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 300, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 600}
0.519492 (0.004802) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.497740 (0.001695) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.512994 (0.013559) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 200}
0.507345 (0.009040) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.01, 'optimizer': 'SGD', 'size_hidden': 600}
0.500565 (0.002260) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.500565 (0.002825) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 0.001, 'optimizer': 'SGD', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'RMSprop', 'size_hidden': 600}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 200}
0.519774 (0.004520) with params: {'activation': 'softmax', 'batch_size': 128, 'epochs': 800, 'init': 'glorot_uniform', 'loss': 'categorical_crossentropy', 'lr': 1e-05, 'optimizer': 'SGD', 'size_hidden': 600}

'''