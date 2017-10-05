import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from hyperparams import Hyperparams as hp
import nn_utils
from selection import features_utils

sns.set()

a = np.array([0.0] * 82)
b = np.array([1.0] * 64)
c = np.concatenate((a, b), axis=0)
y_actual = np.array(c).reshape(146, 1)

data_file = 'data/data.tsv'
df = pd.read_csv(data_file, sep='\t', header=None, index_col=0).T
X = nn_utils.norm_data(df)
X['Case'] = ['AUTISM'] * 82 + ['CONTROL'] * 64


x_input = features_utils.execute_selection(hp.selection_methods, X, num_features=hp.num_features)

nn_inputs = nn_utils.init_inputs(hp.num_features, hp.selection_methods)

y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')

feed_forward = nn_utils.dense_feed_forward(nn_inputs, units=hp.activation_size, activation=hp.activation)

logits = nn_utils.logits_layer(feed_forward, units=1)

nn_outcomes = tf.concat(list(logits.values()), 1)
nn_outcomes = tf.reshape(nn_outcomes, [-1, 4, 1])

ensemble_logits = tf.layers.dense(nn_outcomes, units=50)
ensemble_logits_att = tf.layers.dense(ensemble_logits, units=1)
attentions_weights = tf.nn.softmax(ensemble_logits_att, 1)
attentions_weights = tf.reshape(attentions_weights, [-1, hp.num_methods])
out = attentions_weights*tf.reshape(nn_outcomes, [-1, hp.num_methods])
logits_out = tf.layers.dense(out, units=1)
sig = tf.nn.sigmoid(logits_out)
pred = tf.round(sig)

acc = tf.reduce_mean((pred * y) + ((1 - pred) * (1 - y)))

loss = tf.losses.sigmoid_cross_entropy(
      multi_class_labels=y, logits=logits_out)

opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(hp.num_epochs):
        shuffle_idxs = np.random.permutation(range(146))
        num_batches = len(X) // hp.batch_size
        for batch in range(num_batches):
            y_batch = y_actual[shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]]
            x_batch_fisher = x_input['fisher'][:, shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]].T
            x_batch_corr = x_input['corr'][:, shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]].T
            x_batch_ttest = x_input['ttest'][:, shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]].T
            x_batch_random = x_input['random'][:, shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]].T
            my_opt, = sess.run([opt], feed_dict={
                nn_inputs['fisher']: x_batch_fisher,
                nn_inputs['corr']: x_batch_corr,
                nn_inputs['ttest']: x_batch_ttest,
                nn_inputs['random']: x_batch_random,
                y: y_batch})
        if i % 1000:
            my_acc, my_att = sess.run([acc, attentions_weights], feed_dict={
                nn_inputs['fisher']: x_input['fisher'].T,
                nn_inputs['corr']: x_input['corr'].T,
                nn_inputs['ttest']: x_input['ttest'].T,
                nn_inputs['random']: x_input['random'].T,
                y: y_actual})
            ax = sns.heatmap(my_att.T)
            plt.show()
