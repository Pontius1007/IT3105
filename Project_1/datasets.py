import numpy as np
from random import shuffle

# should be taken from variables.json
__mnist_path__ = "/Users/sebastian/Downloads/mnist-zip/"



# loads mnist
def load_flat_text_cases(filename, cfraction, dir=__mnist_path__,):
    f = open(dir + filename, "r")
    lines = [line.split(" ") for line in f.read().split("\n")]
    f.close()
    len_lines = float(len(lines))
    fraction = int(np.ceil(cfraction*len_lines))
    new_lines = lines[:fraction]
    x_l = list(map(int, new_lines[0]))[:(fraction-1)] # target
    x_t = [list(map(int, line)) for line in new_lines[1:]] # input
    x_l = [[i] for i in x_l]
    print([list(i) for i in zip(x_t, x_l)])
    return [list(i) for i in zip(x_t, x_l)]

    # x_l = [TFT.int_to_one_hot(int(fv), 10) for fv in new_lines[0]]
    # x_l = x_l[:(fraction-1)]
    # x_t = np.array([new_lines[i] for i in range(1, len(new_lines))]).astype(int)
    # x_t = x_t/255
    # #x_t = normalize_inputs(x_t.astype(int))

    # return [[l, t] for l, t in zip(x_t, x_l)]

    # [[[input], [target]], [[input], [target]]]


# loads yeast, wine, etc.
def load_generic_file(filename, cfraction, dir=__mnist_path__,):
    with open(dir+filename, 'r') as infile:
        output_list = []
        lines = infile.readlines()
        fraction = int(np.ceil(cfraction*len(lines)))
        for line in lines:
            line_output = []
            split_line = line.replace('; ', ', ')
            split_line = line.strip().split(',')
            input_vector = [i for i in split_line[:-1]]
            target_vector = split_line[-1]
            line_output.append(input_vector)
            line_output.append([target_vector])
            output_list.append(line_output)
        # have to shuffle to get whole range
        shuffle(output_list)
        return output_list[:fraction]