import numpy as np

from sklearn.model_selection import train_test_split


def generate_input(input_array):
    output_array = []
    seq_len = [len(seq) for seq in input_array]
    max_seq_len = max(seq_len)
    padding = np.array([[0, 0, 0]], dtype=np.float32)
    for i in range(len(input_array)):
        if len(input_array[i]) < max_seq_len:
            padding_difference = max_seq_len - len(input_array[i])
            for j in range(padding_difference):
                input_array[i] = np.concatenate((input_array[i], padding))
        output_array.append(input_array[i])
    return np.asarray(output_array), max_seq_len, np.asarray(seq_len)


def array_splitter(input_array):
    x = input_array
    y = np.roll(x, 1, axis=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    return x_train, x_test, y_train, y_test


def next_batch(x, y, seq_len, st, end):
    x_batch = x[st:end]
    y_batch = y[st:end]
    seq_len_batch = seq_len[st:end]
    return x_batch, y_batch, seq_len_batch

