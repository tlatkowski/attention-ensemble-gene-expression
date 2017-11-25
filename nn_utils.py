import pandas as pd
import tensorflow as tf

from hyperparams import Hyperparams as hp


def norm_data(X: pd.DataFrame):
    mean = X.mean()
    var = X.var()
    X_norm = (X - mean) / var
    return X_norm


def init_inputs(num_features, selection_methods):
    inputs = dict()
    for selection_method in selection_methods:
        with tf.name_scope('input_' + selection_method):
            inputs[selection_method] = tf.placeholder(dtype=tf.float32,
                                                      shape=[None, num_features],
                                                      name=selection_method + '_input')
    return inputs


def dense_feed_forward(nn_inputs: dict, units, activation):
    dense_ff = dict()
    for name, nn_input in nn_inputs.items():
        with tf.name_scope('ff_' + name):
            dense_ff[name] = tf.layers.dense(nn_input, units=units, activation=activation, name=name + '_ff')
    return dense_ff


def attention_layer(nn_outcomes, attention_size=50):
    with tf.name_scope('attention'):
        attention_W = tf.layers.dense(nn_outcomes, units=attention_size, name='attention_W')
        attention_v = tf.layers.dense(attention_W, units=1, name='attention_v')
        attentions_weights = tf.nn.softmax(attention_v, 1, name='attention_weights')
        attentions_weights = tf.reshape(attentions_weights, [-1, hp.num_methods])
        out = attentions_weights * tf.reshape(nn_outcomes, [-1, hp.num_methods])
    return out


def logits_layer(feed_forward, units):
    logits_layers = dict()
    for name, ff in feed_forward.items():
        logits_layers[name] = tf.layers.dense(ff, units=units, name=name + '_logits')
    return logits_layers
