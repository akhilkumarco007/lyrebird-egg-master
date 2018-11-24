import tensorflow as tf
import numpy as np
from ops import *

input_dim = 3
max_time = 4
num_hidden_units = 10
out_dim = 3

x = tf.placeholder(tf.float32, [None, max_time, input_dim])
y = tf.placeholder(tf.float32, [None, max_time, out_dim])

# create weight matrix initialized randomely from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, out_dim])

# create bias vector initialized as zero
b = bias_variable(shape=[out_dim])

pred_out = LSTM(x, W, b, num_hidden_units)

cost = tf.reduce_mean(tf.square(pred_out - y))
train_op = tf.train.AdamOptimizer().minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

x_train = np.array([[[1, 2, 3], [2, 3, 4], [5, 6, 7], [6, 9, 1]],
                    [[5, 6, 9], [7, 0, 5], [7, 8, 4], [8, 9, 0]],
                    [[3, 2, 1], [4, 3, 5], [5, 6, 7], [9, 8, 4]],
                    [[1, 6, 0], [4, 3, 1], [6, 0, 3], [3, 5, 7]]])
y_train = np.array([[[2, 3, 4], [3, 4, 5], [6, 7, 8], [7, 0, 2]],
                    [[6, 7, 0], [8, 1, 6], [8, 9, 5], [9, 0, 1]],
                    [[4, 3, 2], [5, 4, 6], [6, 7, 8], [0, 9, 5]],
                    [[2, 7, 1], [5, 4, 2], [7, 1, 4], [4, 6, 8]]])
# y_train = np.array([[1, 3, 8, 14],
#                     [5, 12, 19, 27],
#                     [3, 7, 12, 21]])

x_test = np.array([[[1, 9, 2], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
                   [[4, 8, 0], [5, 3, 1], [6, 4, 2], [7, 9, 1]]])

# y_test = np.array([[1, 3, 6, 10],
#                    [4, 9, 15, 22]])
y_test = np.array([[[2, 0, 3], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                   [[5, 9, 1], [6, 4, 2], [7, 5, 3], [8, 0, 2]]])

with tf.Session() as sess:
    sess.run(init)
    for i in range(8000):
        _, mse = sess.run([train_op, cost], feed_dict={x: x_train, y: y_train})
        if i % 1000 == 0:
            print('Step {}, MSE={}'.format(i, mse))
    # Test
    y_pred = sess.run(pred_out, feed_dict={x: x_test})

    for i, x in enumerate(y_test):
        print("When the ground truth output is {}, the model thinks it is {}"
              .format(y_test[i], y_pred[i]))
