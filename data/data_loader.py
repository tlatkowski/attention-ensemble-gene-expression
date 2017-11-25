import numpy as np
import pandas as pd
from nn_utils import norm_data


def load_data():
    num_autism = 82
    num_control = 64
    a = np.array([0.0] * num_autism)
    b = np.array([1.0] * num_control)
    c = np.concatenate((a, b), axis=0)
    y_actual = np.reshape(c, (-1, 1))

    data_file = 'data/data.tsv'
    df = pd.read_csv(data_file, sep='\t', header=None, index_col=0).T
    X = norm_data(df)
    X['Case'] = ['AUTISM'] * num_autism + ['CONTROL'] * num_control
    X['Labels'] = y_actual
    return X