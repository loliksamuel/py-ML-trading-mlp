import numpy as np

from data_splitter.splitter_abstract import AbstractDataSplitter


class TrainTestPercentageDataSplitter(AbstractDataSplitter):
    def __init__(self, percent_test_split=33):
        self.percent_test_split = percent_test_split

    def split(self, x):
        min_index = 0
        max_index = len(x) - 1
        split_index = int(x.shape[0] * ((100.0 - self.percent_test_split) / 100.0))
        indices = np.arange(x.shape[0])
        train_indices = indices[min_index:split_index]
        test_indices = indices[split_index: max_index + 1]
        yield train_indices, test_indices, ''
