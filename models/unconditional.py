import tensorflow as tf
import numpy as np
from utils import plot_stroke
from ops import rnn_sampling, weight_variable, bias_variable


def generate_unconditionally(seed=0):
    checkpoint_file = "./saved_models/-15910"
    vars_to_rename = {
        "lstm/basic_lstm_cell/weights": "lstm/basic_lstm_cell/kernel",
        "lstm/basic_lstm_cell/biases": "lstm/basic_lstm_cell/bias",
    }
    new_checkpoint_vars = {}
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    for old_name in reader.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

    n_hidden = 512

    init_input = tf.placeholder(tf.float32, shape=[1, 3])
    w = weight_variable(shape=[n_hidden, 3])
    b = bias_variable(shape=[3])
    lstm = rnn_sampling(n_hidden, init_input, w, b)
    input_array = np.array([[0, 0.40, 1.05]], dtype=np.float32)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, checkpoint_file)
        output = sess.run([lstm], feed_dict={init_input: input_array})
        output = np.array(output)
        plot_stroke(output)


if __name__ == "__main__":
    generate_unconditionally()
