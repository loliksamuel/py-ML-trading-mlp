from data_splitter.splitter_k_fold_purged import PurgedKFoldDataSplitter
from data_splitter.splitter_simple import TrainTestPercentageDataSplitter
from model.enum import MlModel
from mlp_trading import MlpTrading
from utils.utils import get_data_from_disc
import numpy as np


def load_data():
    print('\n\n\n============================================================================')
    print('#Loading      data')
    print('===============================================================================')
    df_all = get_data_from_disc(symbol='^GSPC', skip_first_lines=3600, size_output=2)
    return df_all


def execute_model_train_and_test(full_data_frame, data_splitter, epochs):
    cv_scores = []
    for train_indices, test_indices, iteration_id in data_splitter.split(full_data_frame.values):
        mlp_trading = MlpTrading(symbol='^GSPC')

        (scores, model, params) = mlp_trading\
                                                .execute(df_all=full_data_frame,
                                                         train_data_indices=train_indices,
                                                         test_data_indices=test_indices,
                                                         iteration_id=iteration_id,
                                                         model_type=MlModel.MLP,
                                                         epochs=epochs,
                                                         size_hidden=15,
                                                         batch_size=128,
                                                         loss='categorical_crossentropy',
                                                         lr=0.00001,
                                                         rho=0.9,
                                                         epsilon=None,
                                                         decay=0.0,
                                                         kernel_init='glorot_uniform',
                                                         dropout=0.2,
                                                         verbose=0)
        cv_scores.append(scores[1] * 100)


    return cv_scores, model, params




epochs    = 1#5000
cv_scores = [0]
all_data = load_data()



# n_splits  = 5
# print('\n\n\n============================================================================')
# print(f'==        {n_splits}    Cross      Validation       MODE  ')
# print('============================================================================')
# (cv_scores, _, _)  = execute_model_train_and_test(all_data,  epochs=epochs, data_splitter=PurgedKFoldDataSplitter(n_splits=n_splits, gap_percentage=1.0))



print('\n\n\nֿ\n\n\nֿ\n\n\nֿ============================================================================')
print('#      1  SIMPLE   SPLIT   MODE      ')
print('============================================================================')
(_, model, params)  = execute_model_train_and_test(all_data, epochs=epochs, data_splitter=TrainTestPercentageDataSplitter(33))



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

