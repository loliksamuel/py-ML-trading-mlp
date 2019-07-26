import numpy as np


class PurgedKFold:
    def __init__(self, n_splits=3, gap_percentage=1.0):
        self.n_splits = n_splits
        self.gap_percentage = gap_percentage

    def split(self, x):
        indices = np.arange(x.shape[0])
        test_starts = [
            (i[0], i[-1] + 1) for i in np.array_split(np.arange(x.shape[0]),
                                                      self.n_splits)
        ]
        for i, j in test_starts:
            gap = int(((j - i) * self.gap_percentage) / 100)

            min_index = 0
            max_index = len(x) - 1

            if i == min_index:
                test_indices = indices[min_index:j - gap]
                train_indices = indices[j + 2 * gap: max_index]
            elif j == max_index:
                test_indices = indices[i + gap:max_index]
                train_indices = indices[min_index: i - 2 * gap]
            else:
                test_indices = indices[i + gap:j - gap]
                train_indices = np.concatenate((indices[min_index: i - 2 * gap], indices[j + 2 * gap: max_index]))

            yield train_indices, test_indices
