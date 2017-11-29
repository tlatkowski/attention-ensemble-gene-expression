from constants import ACTIVATIONS as ACTIV
import tensorflow as tf


def feed_forward_diff_features(nn_inputs: dict, units, activation):
    dense_ff = dict()
    for name, nn_input in nn_inputs.items():
        with tf.name_scope('ff_' + name):
            dense_ff[name] = tf.layers.dense(nn_input, units=units, activation=activation, name=name + '_ff')

    nn_outcomes = tf.concat(list(dense_ff.values()), 1)
    nn_outcomes = tf.reshape(nn_outcomes, [-1, len(nn_inputs), units])
    return nn_outcomes


def feed_forward_diff_activations(nn_input, units, activations=[ACTIV.RELU, ACTIV.SIGMOID, ACTIV.TANH]):
    dense_ff = dict()
    for name, activation in activations:
        with tf.name_scope('ff_' + name):
            dense_ff['ff_' + name] = tf.layers.dense(nn_input,
                                                     units=units,
                                                     activation=activation,
                                                     name=name + '_ff')
    return dense_ff
