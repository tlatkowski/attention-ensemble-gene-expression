import tensorflow as tf

class Hyperparams:
    num_features = 500
    num_methods = 4
    attention_size = 10
    activation = tf.nn.tanh
    activation_size = 64
    selection_methods = ['fisher', 'ttest', 'corr', 'random']
    num_methods = len(selection_methods)