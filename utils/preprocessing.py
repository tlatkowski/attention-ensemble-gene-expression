import pandas as pd


def normalize(data: pd.DataFrame):
    mean = data.mean()
    var = data.var()
    norm_data = (data - mean) / var
    return norm_data
