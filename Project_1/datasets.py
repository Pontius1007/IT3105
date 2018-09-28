import numpy as np
import tflowtools as TFT
from random import shuffle

# should be taken from variables.json


# loads mnist
def load_flat_text_cases(filename, cfraction,):
    f = open(filename, "r")
    lines = [line.split(" ") for line in f.read().split("\n")]
    f.close()
    len_lines = float(len(lines))
    fraction = int(np.ceil(cfraction*len_lines))
    new_lines = lines[:fraction]
    x_l = list(map(int, new_lines[0]))[:(fraction-1)] # target
    x_t = [list(map(int, line)) for line in new_lines[1:]] # input
    np.array(x_t)
    x_t = np.divide(x_t, 255)
    x_l = [TFT.int_to_one_hot(i, 10) for i in x_l]
    return [list(i) for i in zip(x_t, x_l)]


# loads yeast, wine, etc.
def load_generic_file(filename, cfraction,):
    with open(filename, 'r') as infile:
        output_list = []
        lines = infile.readlines()
        fraction = int(np.ceil(cfraction*len(lines)))
        for line in lines:
            line_output = []
            split_line = line.replace(';', ',')
            split_line = split_line.strip().split(',')
            input_vector = [float(i) for i in split_line[:-1]]
            target_vector = int(split_line[-1])
            hot_target = TFT.int_to_one_hot(target_vector, 11)
            line_output.append(input_vector)
            line_output.append(hot_target)
            output_list.append(line_output)
        # have to shuffle to get whole range
        shuffle(output_list)
        return output_list[:fraction]