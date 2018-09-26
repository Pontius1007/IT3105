from ann import *


class Parameters:
    def __init__(self):
        self.dims = []
        self.hidden_activation_function = "relu"
        self.optimizer = "RMSprop"
        self.weight_range_lower = -0.1
        self.weight_range_upper = 0.1
        self.learning_rate = 0.1
        self.show_int = 500
        self.mbs = 5000
        self.vfrac = 0.1
        self.tfrac = 0.1
        self.vint = 200
        self.sm = False

        # For training
        self.bestk = 1
        self.steps = 1000

        # For grabbed variables
        self.grabbed_weights = []
        self.grabbed_biases = []


class InputRunHandler:

    def __init__(self, ann):
        self.ann = ann
        self.params = Parameters()

    def evaluate_input(self, u_input):
        print("Du er en: ", u_input)
