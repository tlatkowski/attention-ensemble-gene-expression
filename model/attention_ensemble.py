import nn_utils
import tensorflow as tf
from hyperparams import Hyperparams as hp


class AttentionBasedEnsemble:
    def __init__(self):

        with tf.name_scope('att'):
            self.nn_inputs = nn_utils.init_inputs(hp.num_features, hp.selection_methods)

            self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')

            feed_forward = nn_utils.dense_feed_forward(self.nn_inputs, units=hp.activation_size, activation=hp.activation)

            logits = nn_utils.logits_layer(feed_forward, units=1)

            nn_outcomes = tf.concat(list(logits.values()), 1)
            nn_outcomes = tf.reshape(nn_outcomes, [-1, 4, 1])

            attention_W = tf.layers.dense(nn_outcomes, units=50, name='attention_W')
            attention_v = tf.layers.dense(attention_W, units=1, name='attention_v')
            self.attentions_weights = tf.nn.softmax(attention_v, 1, name='attention_weights')
            self.attentions_weights = tf.reshape(self.attentions_weights, [-1, hp.num_methods])
            out = self.attentions_weights*tf.reshape(nn_outcomes, [-1, hp.num_methods])
            logits_out = tf.layers.dense(out, units=1)
            sig = tf.nn.sigmoid(logits_out)
            pred = tf.round(sig)

            self.acc = tf.reduce_mean((pred * self.y) + ((1 - pred) * (1 - self.y)))

            self.loss = tf.losses.sigmoid_cross_entropy(
                  multi_class_labels=self.y, logits=logits_out)

            # summaries
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.acc)
            self.merged_summary_op = tf.summary.merge_all()

            self.opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)