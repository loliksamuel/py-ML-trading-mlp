from data_splitter.splitter_simple import TrainTestPercentageDataSplitter
from model.enum import MlModel
from buld.build_models import MlpTrading
from buld.utils import get_data_from_disc, data_transform, data_clean
import numpy as np






def execute_model_train_and_test(full_data_frame, data_splitter, epochs, verbose=2):
    cv_scores = []
    for train_indices, test_indices, iteration_id in data_splitter.split(full_data_frame.values):
        mlp_trading = MlpTrading(symbol='^GSPC')

        (scores, model, params) = mlp_trading\
                                                .execute(df_all=full_data_frame,
                                                         train_data_indices=train_indices,
                                                         test_data_indices=test_indices,
                                                         iteration_id=iteration_id,
                                                         model_type=MlModel.MLP,#bug with MlModel.LSTM


                                                         epochs=epochs,#  iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
                                                         size_hidden=2048,# 1 to 1024
                                                         batch_size=128,#1 to 1024  # we cannot pass the entire data into network at once , so we divide it to batches . number of samples that we will pass through the network at 1 time and use for each epoch. default is 32
                                                         loss='categorical_crossentropy',#binary_crossentropy categorical_crossentropy
                                                         lr=0.1, #1 to 0.000000001 if loss is diminishing very slowly increase lr
                                                         rho=0.9,#default 0.9
                                                         epsilon=None,
                                                         decay=0.5,
                                                         kernel_init='glorot_uniform',#  'glorot_uniform'(default),'normal', 'uniform'
                                                         dropout=0.2,#for genralization
                                                         verbose=verbose)#0 1 2
        cv_scores.append(scores[1] * 100)


    return cv_scores, model, params
'''
50 epochs benchmark
val_loss   val_acc      loss       acc  epoch
45  0.692096  0.533362  0.691158  0.525999     45
46  0.692106  0.532926  0.690589  0.532338     46
47  0.692108  0.530092  0.690019  0.533949     47
48  0.692096  0.530528  0.690671  0.529007     48
49  0.692062  0.532490  0.691352  0.526321     49

Actual\Predics    0          1
  0              127        1999
  1              145        2315
'''
cv_scores = [0]
symbol    = '^GSPC'
epochs    = 100 #iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
skip_first_lines=3600 #>400  total 13894 daily bars
size_output = 2 #choose :-3 = random (-1 0 +1), -2=random(0 or 1), 2 for up/dn , 3 for up/dn/hold
verbose     = 2 #0=none, 1=all, 2=mid
data_raw    = get_data_from_disc (symbol=symbol, usecols=['Date', 'Close', 'Open', 'High', 'Low', 'Volume'])
data_clean  = data_clean(data_raw)
df_all      = data_transform     (data_clean, skip_first_lines=skip_first_lines, size_output=size_output)


# df_features      = data_select     (df_all, names_input)
# df_features_norm = data_normalize0 (df_features.values, axis=1)
# df_y_observed    = data_select     (df_all, 'isUp')


#
# n_splits  = 5
# verbose = 0
# print('\n\n\n============================================================================')
# print(f'==        {n_splits}    Cross      Validation       MODE  ')
# print('============================================================================')
# (cv_scores, _, _)  = execute_model_train_and_test(df_all,  epochs=epochs, verbose=verbose, data_splitter=PurgedKFoldDataSplitter(n_splits=n_splits, gap_percentage=1.0))



print('\n\n\nֿ\n\n\nֿ\n\n\nֿ============================================================================')
print('#      1  SIMPLE   SPLIT   MODE      ')
print('============================================================================')
(_, model, params)  = execute_model_train_and_test(df_all, epochs=epochs, verbose=verbose, data_splitter=TrainTestPercentageDataSplitter(33))



print('\n======================================')
print('Total Accuracy Cross Validation: ')
print(*cv_scores, sep = ", ")
print(f'{np.mean(cv_scores):.2f}% (+/- {np.std(cv_scores):.2f}%)')
print('======================================')


print('\n\n\n\n\nֿ\n\n\nֿֿ===============================================================================')
print('#Save   model')
print('===============================================================================')
model.summary()
model.save(folder='files/output/', filename=params, iteration_id='')

