import pandas as pd


def normalize(data: pd.DataFrame):
    mean = data.mean()
    var = data.var()
    norm_data = (data - mean) / var
    return norm_data


def create_feed_dict(model, batch, selection_methods):
    feed_dict = dict()
    feed_dict[model.labels] = batch['y']
    for method in selection_methods:
        feed_dict[model.nn_inputs[method]] = batch[method]
    return feed_dict
