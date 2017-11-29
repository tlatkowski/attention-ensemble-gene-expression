import tensorflow as tf

from layers.common_layers import init_inputs, logits_layer
from layers.feed_forward_layers import feed_forward_diff_features
from layers.attention_layers import attention_layer, attention_layer_tensordot
from hyperparams import Hyperparams as hp


class AttentionBasedEnsemble:

    def __init__(self):

        with tf.name_scope('input'):
            self.nn_inputs = init_inputs(hp.num_features, hp.selection_methods)
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')

        with tf.name_scope('ff'):
            feed_forward = feed_forward_diff_features(self.nn_inputs, units=hp.activation_size, activation=hp.activation)

        with tf.name_scope('output'):
            out = attention_layer(feed_forward, attention_size=50)
            logits = tf.layers.dense(out, units=1)
            sig = tf.nn.sigmoid(logits)
            predictions = tf.round(sig)

        with tf.name_scope('train'):
            self.loss = tf.losses.sigmoid_cross_entropy(
                  multi_class_labels=self.labels, logits=logits)
            self.opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

        with tf.name_scope('summaries'):
            self.acc = tf.reduce_mean((predictions * self.labels) + ((1 - predictions) * (1 - self.labels)))
            self.precision, precision_op = tf.metrics.precision(self.labels, predictions)
            # summaries
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.acc)
            tf.summary.scalar('precision_op', precision_op)
            self.merged_summary_op = tf.summary.merge_all()
