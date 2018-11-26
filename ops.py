import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(shape):
    initer = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initer)


def LSTM(x, weights, biases, num_hidden, seqLen):
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(num_hidden) for _ in range(3)])
    outputs, states = tf.nn.dynamic_rnn(multi_lstm_cell, x, sequence_length=seqLen, dtype=tf.float32)
    num_examples = tf.shape(x)[0]
    w_repeated = tf.tile(tf.expand_dims(weights, 0), [num_examples, 1, 1])
    out = tf.matmul(outputs, w_repeated) + biases
    out = tf.squeeze(out)
    return out


def lstm_cell(n_hidden):
    lstm = tf.nn.rnn_cell.LSTMCell(n_hidden)
    return lstm


def rnn_sampling(num_hidden, initial_input, weights, biases):
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(num_hidden) for _ in range(3)])
    state = multi_lstm_cell.zero_state(1, dtype=tf.float32)
    out = []
    time_steps = 200
    for t in range(time_steps):
        if t == 0:
            output, state = multi_lstm_cell(initial_input, state)
            output = tf.matmul(output, weights) + biases
            output = tf.squeeze(output[0])
            out.append(output)
        else:
            output, state = multi_lstm_cell(tf.reshape(out[t - 1], [1, 3]), state)
            output = tf.matmul(output, weights) + biases
            output = tf.squeeze(output[0])
            out.append(output)
    return out
