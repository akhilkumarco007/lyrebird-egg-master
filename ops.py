import tensorflow as tf
import numpy as np


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
    elif mode == 'infer':
        initial_seed_input = np.array([[0, 0, 0]])
        multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(num_hidden) for _ in range(2)])
        state = tf.zeros((256, 256))
        out = []
        time_steps = 800
        for t in range(time_steps):
            if t == 0:
                output, state = multi_lstm_cell(initial_seed_input, state)
                out.append(output)
            else:
                output, state = multi_lstm_cell(out[t - 1], state)
                out.append(output)
    else:
        raise ValueError('Use train or infer for mode')
    return out


def lstm_cell(n_hidden):
    lstm = tf.nn.rnn_cell.LSTMCell(n_hidden)
    return lstm