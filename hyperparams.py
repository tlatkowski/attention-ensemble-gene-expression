import tensorflow as tf

class Hyperparams:
    num_features = 500
    num_methods = 4
    attention_size = 10
    activation = tf.nn.tanh
    activation_size = 64
    batch_size = 10
    num_epochs = 1000
    eval_every = 10
    selection_methods = ['fisher', 'ttest', 'corr', 'random']
    num_methods = len(selection_methods)
