from data_splitter.purged_k_fold_data_splitter import PurgedKFoldDataSplitter
from data_splitter.train_test_percentage_data_splitter import TrainTestPercentageDataSplitter
from model.ml_model import MlModel
from mlp_trading import MlpTrading
from utils.utils import get_data_from_disc
import numpy as np


def load_data():
    print('\n======================================')
    print('Loading the data')
    print('======================================')
    df_all = get_data_from_disc(symbol='^GSPC', skip_first_lines=3600, size_output=2)
    return df_all


def execute_model_train_and_test(full_data_frame, data_splitter, epochs):
    cv_scores = []
    for train_indices, test_indices, iteration_id in data_splitter.split(full_data_frame.values):
        mlp_trading = MlpTrading(symbol='^GSPC')
        scores = mlp_trading.execute(df_all=full_data_frame,
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
                                     verbose=2)
        cv_scores.append(scores[1] * 100)
    return cv_scores


all_data = load_data()

print('\n======================================')
print('Cross Validation Test')
print('======================================')
cv_accuracies = execute_model_train_and_test(all_data,
                                             PurgedKFoldDataSplitter(n_splits=5, gap_percentage=1.0),
                                             epochs=50)
print('\n======================================')
print('Total Cross Validation Accuracy')
print(f'{np.mean(cv_accuracies):.2f}% (+/- {np.std(cv_accuracies):.2f}%)')
print('======================================')

print('\n======================================')
print('Model Development')
print('======================================')
execute_model_train_and_test(all_data, TrainTestPercentageDataSplitter(33), epochs=2000)
