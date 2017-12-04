import tensorflow as tf

from layers.attention_layers import attention_layer
from layers.common_layers import init_inputs
from layers.feed_forward_layers import feed_forward_diff_features, feed_forward_diff_layers
from utils.hyperparams import Hyperparams as hp


class AttentionBasedEnsembleNets:

    def __init__(self, selection_methods, num_features, learning_rate=0.01):

        with tf.name_scope('input'):
            self.nn_inputs = init_inputs(num_features, selection_methods)
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='labels')

        with tf.name_scope('ff'):
            nets = {'fisher': [200, 50], 'ttest': [200, 50], 'corr': [200, 50], 'random': [200, 50]}
            feed_forward = feed_forward_diff_layers(self.nn_inputs, nets)
        with tf.name_scope('output'):
            out = attention_layer(feed_forward, attention_size=50)
            logits = tf.layers.dense(out, units=1)
            sig = tf.nn.sigmoid(logits)
            predictions = tf.round(sig)

        with tf.name_scope('train'):
            self.loss = tf.losses.sigmoid_cross_entropy(
                  multi_class_labels=self.labels, logits=logits)
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        with tf.name_scope('summaries'):
            self.acc = tf.reduce_mean((predictions * self.labels) + ((1 - predictions) * (1 - self.labels)))
            self.precision, precision_op = tf.metrics.precision(self.labels, predictions)
            # summaries
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.acc)
            tf.summary.scalar('precision_op', precision_op)
            self.merged_summary_op = tf.summary.merge_all()
