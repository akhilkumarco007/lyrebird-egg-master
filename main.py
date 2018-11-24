import numpy as np
import tensorflow as tf

from util import generate_input, array_splitter, next_batch
from ops import weight_variable, bias_variable, LSTM
from utils import plot_stroke

# Parameters
learning_rate = 0.1    # The optimization initial learning rate
epochs = 1000  # Total number of training steps
batch_size = 25     # Size of each batch
display_freq = 1000     # Frequency of displaying the training results
mode = 'train'

# Processing input data
input_data = np.load('./data/strokes.npy')
strokes, max_seq_len, input_seq_len = generate_input(input_data)
x_train, x_test, y_train, y_test = array_splitter(strokes)
x_train_len, x_test_len, y_train_len, y_test_len = array_splitter(input_seq_len)

input_dim = 3
num_hidden_units = 256
out_dim = 3

x = tf.placeholder(tf.float32, shape=[None, max_seq_len, input_dim])
seq_len = tf.placeholder(tf.int32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None, max_seq_len, out_dim])

# create a weight and bias matrix
with tf.name_scope('weights'):
    w = weight_variable(shape=[num_hidden_units, out_dim])
    tf.summary.histogram('W', w)
    b = bias_variable(shape=[out_dim])
    tf.summary.histogram('b', b)

# LSTM construction
pred_out = LSTM(x, w, b, num_hidden_units, seq_len, mode)

with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.square(pred_out - y))
    tf.summary.scalar('loss', cost)
train_op = tf.train.AdamOptimizer().minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

# Model saver
saver = tf.train.Saver()
merged = tf.summary.merge_all()

with tf.Session() as session:
    session.run(init)
    summary = tf.summary.FileWriter(logdir="./logs/", graph=session.graph)
    for i in range(epochs):
        n_iterations = len(x_train) / batch_size
        print('Training Epoch: {0}'.format(i + 1))
        for j in range(n_iterations):
            start = j * batch_size
            end = (j + 1) * batch_size
            x_batch, y_batch, seq_len_batch = next_batch(x_train, y_train, x_train_len, start, end)
            _, mse, summary_tr = session.run([train_op, cost, merged], feed_dict={x: x_batch, seq_len: seq_len_batch, y: y_batch})
            if i % 100 == 0 and j == n_iterations - 1:
                print('Step {}, MSE={}'.format(i, mse))
                saver.save(session, "./saved_models/model" + str(i) + ".ckpt")
            summary.add_summary(summary_tr)
    summary.close()
    # test
    y_pred = session.run(pred_out, feed_dict={x: x_test, seq_len: x_test_len})
    print('--------------Test Results-------------')
    for i, x in enumerate(y_test):
        print("When the ground truth output is {}, the model thinks it is {}"
              .format(y_test[i], y_pred[i]))
    print()
