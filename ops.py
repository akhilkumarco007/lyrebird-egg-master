import tensorflow as tf


# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initer = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initer)


def LSTM(x, weights, biases, num_hidden, seqLen, mode):
    """
    :param x: inputs of size [T, batch_size, input_size]
    :param weights: matrix of fully-connected output layer weights
    :param biases: vector of fully-connected output layer biases
    :param num_hidden: number of hidden units
    """
    if mode == 'train':
        # cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
        multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(num_hidden) for _ in range(2)])
        outputs, states = tf.nn.dynamic_rnn(multi_lstm_cell, x, sequence_length=seqLen, dtype=tf.float32)
        num_examples = tf.shape(x)[0]
        w_repeated = tf.tile(tf.expand_dims(weights, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, w_repeated) + biases
        out = tf.squeeze(out)
    else:
        out = 0
    return out

def lstm_cell(n_hidden):
    lstm = tf.nn.rnn_cell.LSTMCell(n_hidden)
    return lstm