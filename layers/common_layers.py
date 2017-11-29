import tensorflow as tf


def init_inputs(num_features, selection_methods):
    inputs = dict()
    for selection_method in selection_methods:
        with tf.name_scope('input_' + selection_method):
            inputs[selection_method] = tf.placeholder(dtype=tf.float32,
                                                      shape=[None, num_features],
                                                      name=selection_method + '_input')
    return inputs


def logits_layer(feed_forward, units):
    logits_layers = dict()
    for name, ff in feed_forward.items():
        logits_layers[name] = tf.layers.dense(ff, units=units, name=name + '_logits')
    return logits_layers
