import tensorflow as tf
from hyperparams import Hyperparams as hp


def attention_layer(nn_outcomes, attention_size=50):
    with tf.name_scope('attention'):
        u = tf.layers.dense(nn_outcomes, units=attention_size, activation=tf.nn.tanh, name='attention_W')
        vu = tf.layers.dense(u, units=1, name='attention_v')
        alphas = tf.nn.softmax(vu, 1, name='alphas')
        out = tf.reduce_sum(alphas * nn_outcomes, 1)
    return out


def attention_layer_tensordot(nn_outcomes, attention_size=50):
    with tf.name_scope('attention'):
        W = tf.Variable(tf.random_normal([hp.activation_size, attention_size]))
        v = tf.Variable(tf.random_normal([attention_size]))
        # (batch x num_methods x hidden) x (hidden x attention)
        u = tf.nn.tanh(tf.tensordot(nn_outcomes, W, axes=1))
        uv = tf.tensordot(u, v, axes=1)
        alphas = tf.nn.softmax(uv)
        out = tf.reduce_sum(tf.expand_dims(alphas, -1) * nn_outcomes, 1)
    return out