import numpy as np


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
    data_size = len(input_array)
    split_data = np.split(input_array, [int(0.8 * data_size), data_size])
    x_train, x_test = split_data[0], split_data[1]
    y_train, y_test = np.roll(x_train, 1, axis=0), np.roll(x_test, 1, axis=0)
    return x_train, x_test, y_train, y_test


def next_batch(x, y, seq_len, st, end):
    x_batch = x[st:end]
    y_batch = y[st:end]
    seq_len_batch = seq_len[st:end]
    return x_batch, y_batch, seq_len_batch
    # x_batches = np.split(x, n_batches)
    # y_batches = np.split(y, n_batches)
    # seq_len_batches = np.split(seq_len, n_batches)
    # for x_b, y_b, sl_b in zip(x_batches, y_batches, seq_len_batches):
    #     yield x_b, y_b, sl_b
    # N = len(x_train)
    # batch_idxs = range(0, N, batch_size)
    # for i in range(len(batch_idx)):
    #     x_batch = x[batch_idx[i]:batch_idx[i + 1]]
    #     y_batch = y[batch_idx[i]:batch_idx[i + 1]]
    #     seq_len_batch = seq_len[batch_idx[i]:batch_idx[i + 1]]
    #     yield x_batch, y_batch, seq_len_batch
