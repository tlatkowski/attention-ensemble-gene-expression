import tensorflow as tf

from utils.constants import ACTIVATIONS as ACT


def feed_forward_diff_features(nn_inputs: dict, units, activation):
    dense_ff = dict()
    for name, nn_input in nn_inputs.items():
        with tf.name_scope('ff_' + name):
            dense_ff[name] = tf.layers.dense(nn_input, units=units, activation=activation, name=name + '_ff')

    nn_outcomes = tf.concat(list(dense_ff.values()), 1)
    nn_outcomes = tf.reshape(nn_outcomes, [-1, len(nn_inputs), units])
    return nn_outcomes


def feed_forward_diff_activations(nn_input, units, activations=[ACT.RELU, ACT.SIGMOID, ACT.TANH]):
    dense_ff = dict()
    for name, activation in activations:
        with tf.name_scope('ff_' + name):
            dense_ff['ff_' + name] = tf.layers.dense(nn_input,
                                                     units=units,
                                                     activation=activation,
                                                     name=name + '_ff')
    return dense_ff


def feed_forward_diff_layers(nn_inputs: dict, nets: dict, activation=ACT.RELU):
    dense_ff = dict()
    for name, layers in nets.items():
        with tf.name_scope('ff_' + name):
            for i, units in enumerate(layers):
                if i == 0:
                    nn_input = nn_inputs[name]
                else:
                    nn_input = dense_ff['ff_{}_{}'.format(name, i - 1)]

                dense_ff['ff_{}_{}'.format(name, i)] = tf.layers.dense(nn_input,
                                                                           units=units,
                                                                           activation=activation[1])
    return dense_ff