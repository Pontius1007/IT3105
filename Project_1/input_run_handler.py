from ann import *


class Parameters:
    def __init__(self):
        self.dims = []
        self.hidden_activation_function = "relu"
        self.optimizer = "RMSprop"
        self.weight_range_lower = -0.1
        self.weight_range_upper = 0.1
        self.learning_rate = 0.1
        self.show_freq = 500
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

    def __str__(self):
        # TODO Create a decent toString method so we can print the different parameters.

        return ' ,  '.join(['{key} = {value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])


class InputRunHandler:

    def __init__(self, ann):
        self.ann = ann
        self.params = Parameters()

    def evaluate_input(self, u_input):
        if u_input == "load json" or u_input == "lj":
            filename = input("Enter the filepath to the JSON file. Leave blank for default: ")
            if filename == "":
                self.load_json("variables.json")
            else:
                self.load_json(filename)
            print("Parameters are now set to: ")
            print("\n")
            print(self.params)
            print("\n")

    def load_json(self, filename):
        with open(filename) as f:
            data = json.load(f)

        self.params.dims = [data["dimensions"][i] for i in data["dimensions"]]
        self.params.hidden_activation_function = data["hidden_activation_function"]["name"]
        self.params.output_activation_function = data["output_activation_function"]["softmax"]
        self.params.optimizer = data["optimizer"]["name"]
        self.params.weight_range_lower = float(data["ini_weight_range"]["lower_bound"])
        self.params.weight_range_upper = float(data["ini_weight_range"]["upper_bound"])
        self.params.learning_rate = float(data["learning_rate"]["value"])
        self.params.show_freq = int(data["grabbed_variables"]["show_freq"])
        self.params.mbs = int(data["minibatch_size"]["number_of_training_cases"])
        self.params.vfrac = float(data["validation_fraction"]["ratio"])
        self.params.tfrac = float(data["test_fraction"]["ratio"])
        self.params.vint = int(data["validation_interval"]["number"])
        self.params.cost_function = data["cost_function"]["name"]
        self.params.ncases = int(data["num_gen_training_case"]["amount"])
        self.params.steps = int(data["steps"]["number"])
        self.params.cfraction = float(data["case_fraction"]["ratio"])
        self.params.sm = True if (str(self.params.output_activation_function.lower()) == "true") else False
        self.params.bestk = 1 if (str(data["bestk"]["bool"].lower()) == "true") else 0

    def countex(self):
        print("")

