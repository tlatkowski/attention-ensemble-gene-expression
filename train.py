import os

import numpy as np
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

from data.data_loader import load_data
from models.attention_ensemble import AttentionBasedEnsemble
from selection import features_utils
from utils.batch_creator import BatchCreator
from utils.hyperparams import Hyperparams as hp
from utils.preprocessing import create_feed_dict


sns.set()
logs_path = 'logs/'

X = load_data()
y_actual = np.array(list(X['Labels']))
y_actual = np.reshape(y_actual, (-1, 1))
x_input = features_utils.execute_selection(hp.selection_methods, X, num_features=hp.num_features, force=False)

with tf.Session() as sess:
    model = AttentionBasedEnsemble()
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)
    if not os.path.isdir(logs_path):
        os.makedirs(logs_path)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    tqdm_iter = tqdm(range(hp.num_epochs))
    for epoch in tqdm_iter:
        batch_creator = BatchCreator(y_actual, x_input, hp.batch_size, hp.selection_methods)
        num_batches = len(X) // hp.batch_size
        for batch in range(num_batches):
            data_batch = batch_creator.next_batch(batch)
            feed_dict = create_feed_dict(model, data_batch, hp.selection_methods)

            my_opt, summary = sess.run([model.opt, model.merged_summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, epoch * num_batches + epoch)
        if epoch % hp.eval_every:
            full_data = batch_creator.full_data()
            feed_dict = create_feed_dict(model, full_data, hp.selection_methods)
            acc = sess.run([model.acc], feed_dict=feed_dict)
            tqdm_iter.set_postfix(acc=acc)
            # ax = sns.heatmap(my_att.T)
            # plt.show()



