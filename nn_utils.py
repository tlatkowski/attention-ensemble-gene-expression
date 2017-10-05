import pandas as pd
import tensorflow as tf


def norm_data(X: pd.DataFrame):
    mean = X.mean()
    var = X.var()
    X_norm = (X - mean) / var
    return X_norm


def init_weights(num_features, num_methods = 4):
    parameters = dict()
    for i in range(1, num_methods + 1):
        parameters['W' + str(i)] = tf.get_variable('W' + str(i),
                                                   shape=[150, num_features],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(i)] = tf.get_variable('b' + str(i),
                                                   shape=[150, 1],
                                                   initializer=tf.zeros_initializer)
        parameters['W' + str(i) + '_2'] = tf.get_variable('W' + str(i) + '_2',
                                                         shape=[1, 150],
                                                         initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(i) + '_2'] = tf.get_variable('b' + str(i) + '_2',
                                                         shape=[1, 1],
                                                         initializer=tf.zeros_initializer)
    return parameters


def forward_pass(parameters, inputs):
    m = len(parameters) // 4
    f_pass = dict()
    for i in range(1, m+1):
        f_pass['A' + str(i)] = tf.nn.tanh(
            tf.matmul(
                parameters['W' + str(i)],
                inputs['X' + str(i)])
            + parameters['b' + str(i)])
        f_pass['A' + str(i) + '_2'] = tf.nn.sigmoid(
            tf.matmul(
                parameters['W' + str(i) + '_2'],
                f_pass['A' + str(i)])
            + parameters['b' + str(i) + '_2'])
    return f_pass


def init_inputs(num_features, selection_methods):
    inputs = dict()
    for selection_method in selection_methods:
        inputs[selection_method] = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name=selection_method)
    return inputs


def dense_feed_forward(nn_inputs: dict, units, activation):
    dense_ff = dict()
    for name, nn_input in nn_inputs.items():
        dense_ff[name] = tf.layers.dense(nn_input, units=units, activation=activation)
    return dense_ff


def logits_layer(feed_forward, units):
    logits_layers = dict()
    for name, ff in feed_forward.items():
        logits_layers[name] = tf.layers.dense(ff, units=1)
    return logits_layers
