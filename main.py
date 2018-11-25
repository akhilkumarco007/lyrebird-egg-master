import numpy as np
import tensorflow as tf

from util import generate_input, array_splitter, next_batch
from ops import weight_variable, bias_variable, LSTM
from utils import plot_stroke

# Parameters
learning_rate = 0.1             # The optimization initial learning rate
epochs = 10000                  # Total number of training steps
batch_size = 32                 # Size of each batch
val_display_frequency = 100
train_disp_freq = 50
mode = 'train'

# Processing input data
input_data = np.load('./data/strokes.npy', encoding='latin1')
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
    mean_loss, mean_loss_op = tf.metrics.mean(cost)
    tf.summary.scalar('loss', mean_loss)
train_op = tf.train.AdamOptimizer().minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

# Model saver
saver = tf.train.Saver()
merged = tf.summary.merge_all()

with tf.Session() as session:
    session.run(init)
    session.run(tf.local_variables_initializer())
    train_writer = tf.summary.FileWriter(logdir="./logs/train/", graph=session.graph)
    val_writer = tf.summary.FileWriter(logdir="./logs/val/")
    step = 0
    best_val_loss = 100000

    # training
    for i in range(epochs):
        print('Epoch: {0}'.format(i))
        n_iterations = len(x_train) / batch_size
        for j in range(int(n_iterations)):
            start = j * batch_size
            end = (j + 1) * batch_size
            x_batch, y_batch, seq_len_batch = next_batch(x_train, y_train, x_train_len, start, end)
            if j % train_disp_freq == 0:
                _, _, summary_tr = session.run([train_op, mean_loss_op, merged],
                                               feed_dict={x: x_batch, seq_len: seq_len_batch, y: y_batch})
                loss = session.run(mean_loss)
                print('Mean loss for step {1}: {0}'.format(loss, step))
                train_writer.add_summary(summary_tr, step)
                session.run(tf.local_variables_initializer())

            else:
                session.run([train_op, mean_loss_op],
                            feed_dict={x: x_batch, seq_len: seq_len_batch, y: y_batch})
            step += 1

        # evaluation after each epoch
        session.run(tf.local_variables_initializer())
        val_iter = int(len(x_test) / batch_size)
        for k in range(int(val_iter)):
            st = k * batch_size
            en = (k + 1) * batch_size
            x_val, y_val, seq_len_val = next_batch(x_test, y_test, x_test_len, st, en)
            _, summary_val = session.run([mean_loss_op, merged], feed_dict={x: x_val, y: y_val, seq_len: seq_len_val})
        val_writer.add_summary(summary_val, step)
        val_loss = session.run(mean_loss)
        if val_loss < best_val_loss:
            saver.save(session, './saved_models/', global_step=step)
            best_val_loss = val_loss

    print()
