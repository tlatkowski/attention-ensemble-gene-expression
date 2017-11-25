import os

import numpy as np
import seaborn as sns
import tensorflow as tf

from data.data_loader import load_data
from hyperparams import Hyperparams as hp
from model.attention_ensemble import AttentionBasedEnsemble
from selection import features_utils

sns.set()
logs_path = 'logs/'

X = load_data()
y_actual = np.array(list(X['Labels']))
y_actual = np.reshape(y_actual, (-1, 1))
x_input = features_utils.execute_selection(hp.selection_methods, X, num_features=hp.num_features, force=False)

with tf.Session() as sess:
    att_ensemble_model = AttentionBasedEnsemble()
    init = tf.global_variables_initializer()
    sess.run(init)
    if not os.path.isdir(logs_path):
        os.makedirs(logs_path)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(hp.num_epochs):
        shuffle_idxs = np.random.permutation(range(146))
        num_batches = len(X) // hp.batch_size
        for batch in range(num_batches):
            y_batch = y_actual[shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]]
            x_batch_fisher = x_input['fisher'][:, shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]].T
            x_batch_corr = x_input['corr'][:, shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]].T
            x_batch_ttest = x_input['ttest'][:, shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]].T
            x_batch_random = x_input['random'][:, shuffle_idxs[batch*hp.batch_size: (batch+1)*hp.batch_size]].T
            my_opt, summary = sess.run([att_ensemble_model.opt, att_ensemble_model.merged_summary_op], feed_dict={
                att_ensemble_model.nn_inputs['fisher']: x_batch_fisher,
                att_ensemble_model.nn_inputs['corr']: x_batch_corr,
                att_ensemble_model.nn_inputs['ttest']: x_batch_ttest,
                att_ensemble_model.nn_inputs['random']: x_batch_random,
                att_ensemble_model.y: y_batch})
            summary_writer.add_summary(summary, epoch * num_batches + epoch)
        if epoch % hp.eval_every:
            acc = sess.run([att_ensemble_model.acc], feed_dict={
                att_ensemble_model.nn_inputs['fisher']: x_input['fisher'].T,
                att_ensemble_model.nn_inputs['corr']: x_input['corr'].T,
                att_ensemble_model.nn_inputs['ttest']: x_input['ttest'].T,
                att_ensemble_model.nn_inputs['random']: x_input['random'].T,
                att_ensemble_model.y: y_actual})
            print(acc)
            # ax = sns.heatmap(my_att.T)
            # plt.show()
