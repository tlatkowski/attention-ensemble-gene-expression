import numpy as np


class BatchCreator:
    def __init__(self, y_actual, x_input, batch_size, selection_methods):
        num_instances = y_actual.shape[0]
        self._shuffle_idxs = np.random.permutation(range(num_instances))
        self._y_actual = y_actual
        self._x_input = x_input
        self._batch_size = batch_size
        self._selection_methods = selection_methods

    def next_batch(self, batch):
        batch_dict = dict()
        shuffle_idxs_batch = self._shuffle_idxs[batch * self._batch_size: (batch + 1) * self._batch_size]
        batch_dict['y'] = self._y_actual[shuffle_idxs_batch]
        for method in self._selection_methods:
            batch_dict[method] = self._x_input[method][:, shuffle_idxs_batch].T
        return batch_dict

    def full_data(self):
        full_dict = dict()
        full_dict['y'] = self._y_actual
        for method in self._selection_methods:
            full_dict[method] = self._x_input[method].T
        return full_dict
