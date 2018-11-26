import numpy as np
import string

from utils import plot_stroke

input_data = np.load('./data/strokes.npy', encoding='latin1')

# plot_stroke(input_data[2])

exclude = set(string.punctuation)
sentences = []
with open('./data/sentences.txt', 'r') as f:
    for line in f:
        for c in exclude:
            line = line.replace(c, " ")
        words = line.split()
        for word in words:
            if word.lower() not in sentences:
                sentences.append(word.lower())

print()
