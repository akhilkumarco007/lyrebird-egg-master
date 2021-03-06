## Hand Writing Generation and Recognition (WIP)
### Data description:

There are 2 data files that you need to consider: `data.npy` and `sentences.txt`. `data.npy`contains 6000 sequences of points that correspond to handwritten sentences. `sentences.txt` contains the corresponding text sentences. You can see an example on how to load and plot an example sentence in `example.ipynb`. Each handwritten sentence is represented as a 2D array with T rows and 3 columns. T is the number of timesteps. The first column represents whether to interrupt the current stroke (i.e. when the pen is lifted off the paper). The second and third columns represent the relative coordinates of the new point with respect to the last point. Please have a look at the plot_stroke if you want to understand how to plot this sequence.

### Task 1: Unconditional generation.
The idea here is to create a function that is able to create strokes unconditionally. Please have a look at Figure 11 in [1] to understand better how this should look like.
```
def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 1D-array
    return stroke
```

### Task 2: Conditional generation.
Things become more fun now. Let's get started with conditional generation. The idea is to create a function that is able to create the strokes for a corresponding text.
```
def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 1D-array
    return stroke
```

### Task 3: Handwriting recognition. (Optional)
Can you recognize the text given a stroke?
```
def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 1D-array

    # Output:
    #   text - str
    return stroke
```
