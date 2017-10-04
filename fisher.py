import logging
import os
import pandas as pd

import feature_selection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select(df: pd.DataFrame, num_features=100, force=True):
    feature_fn = 'features/Fisher.csv'
    if not os.path.exists(feature_fn) or force:
        _, fisher_features = feature_selection.fisher(df, num_features)
        pd.DataFrame(fisher_features).to_csv(feature_fn)
        X = df[fisher_features].T.values  # input size x batch size
    else:
        fisher_features = pd.read_csv(feature_fn, index_col=1)
        X = df[fisher_features.index].T.values  # input size x batch size
    return X
