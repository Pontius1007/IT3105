from ann import *
import tflowtools as TFT
from datasets import *


class Parameters:
    def __init__(self):
        self.dims = [0, 0, 0]
        self.hidden_activation_function = "relu"
        self.optimizer = "RMSprop"
        self.weight_range_lower = -0.1
        self.weight_range_upper = 0.1
        self.learning_rate = 0.1
        self.show_freq = 500
        self.mbs = 500
        self.vfrac = 0.1
        self.tfrac = 0.1
        self.vint = 200
        self.sm = False
        self.cost_function = "MSE"
        self.ncases = 5000
        self.map_cases = 0
        self.dendrogram_cases = 0

        # For training
        self.bestk = 1
        self.steps = 1000
        self.run_more_steps = 500

        # For grabbed variables
        self.grab_module_index = []
        self.grab_type = []

    def __str__(self):
        return ' ,  '.join(['{key} = {value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])


class InputRunHandler:
    def __init__(self, ann):
        self.ann = ann
        self.params = Parameters()

    def evaluate_input(self, u_input):
        if u_input == "load json" or u_input == "lj":
            while True:
                filename = input("Enter the filepath to the JSON file. Leave blank for default: ")
                filepath = "./config/" + filename + ".json"
                try:
                    if filename == "":
                        self.load_json("./config/variables.json")
                    else:
                        self.load_json(filepath)
                    print("Parameters are now set to: \n")
                    print(self.params)
                    print("\n")
                except (OSError, IOError) as e:
                    print("Could not find file. Error: ", e)
                else:
                    break

        if u_input == "predict" or u_input == "p":
            ncases = int(input("Enter the number of cases you want to predict: "))
            self.ann.model.do_prediction(ncases)

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
        self.params.cost_function = str(data["cost_function"]["name"])
        self.params.ncases = int(data["num_gen_training_case"]["amount"])
        self.params.steps = int(data["steps"]["number"])
        self.params.cfraction = float(data["case_fraction"]["ratio"])
        self.params.sm = True if (str(self.params.output_activation_function.lower()) == "true") else False
        self.params.bestk = 1 if (str(data["bestk"]["bool"].lower()) == "true") else None
        self.params.run_more_steps = int(data["run_more"]["steps"])
        self.params.grab_module_index = [i for i in data["grab_module_index"]]
        self.params.grab_type = [i for i in data["grab_type"]]
        self.params.map_cases = data["do_mapping_cases"]
        self.params.dendrogram_cases = data["do_dendrogram_cases"]
        self.params.dendrogram_layers = data["dendrogram_layers"]
        self.params.map_layers = data["map_layers"]

    def build_ann(self):
        model = Gann(dims=self.params.dims, hidden_activation_function=self.params.hidden_activation_function,
                     optimizer=self.params.optimizer, lower=self.params.weight_range_lower,
                     upper=self.params.weight_range_upper, cman=self.ann.get_cman(), lrate=self.params.learning_rate,
                     showfreq=self.params.show_freq, mbs=self.params.mbs, vint=self.params.vint, softmax=self.params.sm,
                     cost_function=self.params.cost_function, grab_module_index=self.params.grab_module_index,
                     grab_type=self.params.grab_type)
        return model

    def check_mapping_and_dendro(self):
        if self.params.map_cases != 0:
            self.ann.model.do_mapping(self.params.map_layers, self.params.map_cases, self.params.bestk)
        if self.params.dendrogram_layers != 0:
            self.ann.model.create_dendrogram(self.params.dendrogram_layers, self.params.map_cases, bestk=self.params.bestk)
