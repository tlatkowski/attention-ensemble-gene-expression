import logging
import os

import pandas as pd

from selection import feature_selection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select(df: pd.DataFrame, num_features=100, force=True):

    feature_fn = 'features/ttest.csv'
    if not os.path.exists(feature_fn) or force:
        _, t_test_features = feature_selection.t_test(df, num_features)
        pd.DataFrame(t_test_features).to_csv(feature_fn)
        X = df[t_test_features].T.values  # input size x batch size
    else:
        t_test_features = pd.read_csv('features/ttest.csv', index_col=1)
        X = df[t_test_features.index].T.values  # input size x batch size
    return X
