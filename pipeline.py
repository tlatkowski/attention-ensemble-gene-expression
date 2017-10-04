import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;
import tensorflow as tf

import features_utils
import nn_utils

sns.set()


a = np.array([0.0] * 82)
b = np.array([1.0] * 64)
c = np.concatenate((a, b), axis=0)
y_actual = np.array(c).reshape(146, 1)

data_file = 'data.txt'
df = pd.read_csv(data_file, sep='\t', header=None, index_col=0).T
X = nn_utils.norm_data(df)
X['Case'] = ['AUTISM'] * 82 + ['CONTROL'] * 64

num_features = 500
num_methods = 4
attention_size = 10
x_input = features_utils.execute_selection(['fisher', 'ttest', 'corr', 'random'], X, num_features=num_features)

X1 = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name='X1')
X2 = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name='X2')
X3 = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name='X3')
X4 = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name='X4')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')

dense1 = tf.layers.dense(X1, units=64, activation=tf.nn.tanh)
dense2 = tf.layers.dense(X2, units=64, activation=tf.nn.tanh)
dense3 = tf.layers.dense(X3, units=64, activation=tf.nn.tanh)
dense4 = tf.layers.dense(X4, units=64, activation=tf.nn.tanh)

logits1 = tf.layers.dense(dense1, units=1)
logits2 = tf.layers.dense(dense2, units=1)
logits3 = tf.layers.dense(dense3, units=1)
logits4 = tf.layers.dense(dense4, units=1)

sig1 = tf.nn.sigmoid(logits1)
sig2 = tf.nn.sigmoid(logits2)
sig3 = tf.nn.sigmoid(logits3)
sig4 = tf.nn.sigmoid(logits4)

nn_outcomes = tf.concat([logits1, logits2, logits3, logits4], 1)
nn_outcomes = tf.reshape(nn_outcomes, [-1, 4, 1])

ensemble_logits = tf.layers.dense(nn_outcomes, units=50)
ensemble_logits_att = tf.layers.dense(ensemble_logits, units=1)
attentions_weights = tf.nn.softmax(ensemble_logits_att, 1)
attentions_weights = tf.reshape(attentions_weights, [-1, num_methods])
out = attentions_weights*tf.reshape(nn_outcomes, [-1, num_methods])
logits_out = tf.layers.dense(out, units=1)
sig = tf.nn.sigmoid(logits_out)
pred = tf.round(sig)

acc = tf.reduce_mean((pred * Y) + ((1 - pred) * (1 - Y)))

loss = tf.losses.sigmoid_cross_entropy(
      multi_class_labels=Y, logits=logits_out)

opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(10000):
        shuffle_idxs = np.random.permutation(range(146))
        num_batch = 10
        for batch in range(num_batch):
            y_batch = y_actual[shuffle_idxs[batch*14: (batch+1)*14]]
            x_batch_fisher = x_input['fisher'][:, shuffle_idxs[batch*14: (batch+1)*14]].T
            x_batch_corr = x_input['corr'][:, shuffle_idxs[batch*14: (batch+1)*14]].T
            x_batch_ttest = x_input['ttest'][:, shuffle_idxs[batch*14: (batch+1)*14]].T
            x_batch_random = x_input['random'][:, shuffle_idxs[batch*14: (batch+1)*14]].T
            my_opt, = sess.run([opt], feed_dict={
                X1: x_batch_fisher,
                X2: x_batch_corr,
                X3: x_batch_ttest,
                X4: x_batch_random,
                Y: y_batch})
        if i % 1000:
            my_acc, my_att = sess.run([acc, attentions_weights], feed_dict={
                X1: x_input['fisher'].T,
                X2: x_input['corr'].T,
                X3: x_input['ttest'].T,
                X4: x_input['random'].T,
                Y: y_actual})
            # print(my_acc)
            # print(my_att)
            ax = sns.heatmap(my_att.T)
            plt.show()
