import pandas as pd
import tensorflow as tf


def norm_data(X: pd.DataFrame):
    mean = X.mean()
    var = X.var()
    X_norm = (X - mean) / var
    return X_norm


def init_inputs(num_features, selection_methods):
    inputs = dict()
    for selection_method in selection_methods:
        inputs[selection_method] = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name=selection_method + '_input')
    return inputs


def dense_feed_forward(nn_inputs: dict, units, activation):
    dense_ff = dict()
    for name, nn_input in nn_inputs.items():
        dense_ff[name] = tf.layers.dense(nn_input, units=units, activation=activation, name=name + '_ff')
    return dense_ff


def logits_layer(feed_forward, units):
    logits_layers = dict()
    for name, ff in feed_forward.items():
        logits_layers[name] = tf.layers.dense(ff, units=units, name=name + '_logits')
    return logits_layers
