from ann import *
import tflowtools as TFT
from datasets import *


class Parameters:
    def __init__(self):
        self.dims = []
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

        # For training
        self.bestk = 1
        self.steps = 1000
        self.run_more_steps = 500

        # For grabbed variables
        self.grabbed_weights = []
        self.grabbed_biases = []
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
            filename = input("Enter the filepath to the JSON file. Leave blank for default: ")
            if filename == "":
                self.load_json("variables.json")
            else:
                self.load_json(filename)
            print("Parameters are now set to: ")
            print("\n")
            print(self.params)
            print("\n")

        if u_input == "run" or u_input == "r":
            data_input = input("Please enter the dataset you want to run: ").lower()
            if data_input == "countex":
                self.countex()
            elif data_input == "autoex":
                self.autoex()
            elif data_input == "yeast":
                self.yeast()
            elif data_input == "glass":
                self.glass()
            elif data_input == "wine":
                self.wine()
            elif data_input == "mnist":
                self.mnist()

        if u_input == "mapping" or u_input == "dm":
            number_of_cases = int(input("Enter the number of cases you want to map: "))
            self.ann.model.do_mapping(number_of_cases)

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

    def build_ann(self):
        model = Gann(dims=self.params.dims, hidden_activation_function=self.params.hidden_activation_function,
                     optimizer=self.params.optimizer, lower=self.params.weight_range_lower,
                     upper=self.params.weight_range_upper, cman=self.ann.get_cman(), lrate=self.params.learning_rate,
                     showfreq=self.params.show_freq, mbs=self.params.mbs, vint=self.params.vint, softmax=self.params.sm,
                     cost_function=self.params.cost_function, grab_module_index=self.params.grab_module_index,
                     grab_type=self.params.grab_type)
        return model

    def countex(self):
        nbits = int(input("Enter the length of the vector in bits. Enter 0 to set it to the input layer size: "))
        nbits = nbits if (nbits != 0) else self.params.dims[0]
        case_generator = (lambda: TFT.gen_vector_count_cases(self.params.ncases, nbits))
        self.ann.set_cman(Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac))
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        # TFT.fireup_tensorboard('probeview')

    def autoex(self):
        nbits = int(input("Enter the length of the vector in bits. "
                          "Please be careful and not crash my shit with a number like 32: "))
        size = 2 ** nbits
        mbs = self.params.mbs if self.params.mbs else size
        case_generator = (lambda: TFT.gen_all_one_hot_cases(2 ** nbits))

        self.ann.set_cman(Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac))
        model = self.build_ann()
        self.ann.set_model(model)
        # model.gen_probe(0, 'wgt', ('hist', 'avg'))  # Plot a histogram and avg of the incoming weights to module 0.
        # model.gen_probe(1, 'out', ('avg', 'max'))  # Plot average and max value of module 1's output vector
        # model.add_grabvar(0, 'wgt')  # Add a grabvar (to be displayed in its own matplotlib window).
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        # model.runmore(self.params.run_more_steps, bestk=self.params.bestk)

    def yeast(self):
        case_generator = (lambda: load_generic_file('data/yeast.txt', self.params.cfraction))
        self.ann.set_cman(Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac))
        self.params.dims[0] = 8
        self.params.dims[2] = 11
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        # TFT.fireup_tensorboard('probeview')

    def wine(self):
        case_generator = (lambda: load_generic_file('data/winequality_red.txt', self.params.cfraction))
        self.ann.set_cman(Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac))
        self.params.dims[0] = 11
        self.params.dims[2] = 11
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        # TFT.fireup_tensorboard('probeview')

    def glass(self):
        case_generator = (lambda: load_generic_file('data/glass.txt', self.params.cfraction))
        self.ann.set_cman(Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac))
        self.params.dims[0] = 9
        self.params.dims[2] = 11
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        # TFT.fireup_tensorboard('probeview')

    def mnist(self):
        case_generator = (
            lambda: load_flat_text_cases('data/all_flat_mnist_training_cases_text.txt', self.params.cfraction))
        self.ann.set_cman(Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac))
        self.params.dims[0] = 784
        self.params.dims[2] = 10
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        # TFT.fireup_tensorboard('probeview')
