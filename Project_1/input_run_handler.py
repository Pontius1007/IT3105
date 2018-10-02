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

        if u_input == "run" or u_input == "r":
            while True:
                data_input = input("Please enter the dataset you want to run: ").lower()
                try:
                    if data_input == "bitcounter":
                        self.bitcounter()
                    elif data_input == "autoencoder":
                        self.autoencoder()
                    elif data_input == "parity":
                        self.parity()
                    elif data_input == "symmetry":
                        self.symmetry()
                    elif data_input == "segmentcounter" or data_input == "sc":
                        self.segmentcounter()
                    elif data_input == "yeast":
                        self.yeast()
                    elif data_input == "glass":
                        self.glass()
                    elif data_input == "wine":
                        self.wine()
                    elif data_input == "iris":
                        self.iris()
                    elif data_input == "mnist":
                        self.mnist()
                    elif data_input == "q":
                        break
                except Exception as e:
                    print("Not a supported dataset or bit: ", e)
                else:
                    break

        # Only reliable way to get interactive mode for matplot in a standard cmd shell. Use ipython to avoid
        # Or, use MP
        if u_input == "show" or u_input == "plt":
            print("\n You will need to ctrl-z to run this program again. \n")
            PLT.show()
        # TODO Add predict

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
            self.ann.model.do_mapping(self.params.map_cases)
        if self.params.dendrogram_cases != 0:
            self.ann.model.create_dendrogram(self.params.dendrogram_cases)

    def parity(self):
        nbits = input("Enter the length of the vectors. Default to 10: ")
        nbits = nbits if nbits else 10
        case_generator = (lambda: TFT.gen_all_parity_cases(nbits))
        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        self.ann.set_cman(case_man)
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()

    def symmetry(self):
        length = int(input("Enter the length of the vectors: "))
        count = int(input("Enter the number of vectors: "))
        case_generator = (lambda: TFT.gen_symvect_dataset(length, count))
        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        self.ann.set_cman(case_man)
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()

    #  You will not be asked to run a performance test on an autoencoder at the demo
    #  session, but you may choose an autoencoder as the network that you explain in detail.
    def autoencoder(self):
        nbits = int(input("Enter the length of the vector in bits. "
                          "Please be careful and not crash my shit with a number like 32: "))
        case_generator = (lambda: TFT.gen_all_one_hot_cases(2 ** nbits))

        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.ann.set_cman(case_man)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        model = self.build_ann()
        self.ann.set_model(model)
        # model.gen_probe(0, 'wgt', ('hist', 'avg'))  # Plot a histogram and avg of the incoming weights to module 0.
        # model.gen_probe(1, 'out', ('avg', 'max'))  # Plot average and max value of module 1's output vector
        # model.add_grabvar(0, 'wgt')  # Add a grabvar (to be displayed in its own matplotlib window).
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()

    def bitcounter(self):
        nbits = input("Enter the length of the vector in bits. 15 is default: ")
        nbits = nbits if nbits else 15
        case_generator = (lambda: TFT.gen_vector_count_cases(self.params.ncases, nbits))
        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.ann.set_cman(case_man)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()

    def segmentcounter(self):
        size = input("Enter the size. 25 is default: ")
        size = size if size else 25
        count = input("Enter the number of cases. 1000 default: ")
        count = count if count else 1000
        minsegs = input("Enter the minimum number of segments in a vector. Default 0: ")
        minsegs = minsegs if minsegs else 0
        maxsegs = input("Enter the maximum number of segments in a vector. Default 5: ")
        maxsegs = maxsegs if maxsegs else 5
        print(size, count, minsegs, maxsegs)
        case_generator = (lambda: TFT.gen_segmented_vector_cases(size, count, minsegs, maxsegs))
        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.ann.set_cman(case_man)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()

    def yeast(self):
        case_generator = (lambda: load_generic_file('data/yeast.txt', self.params.cfraction))
        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.ann.set_cman(case_man)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()

    def wine(self):
        case_generator = (lambda: load_generic_file('data/winequality_red.txt', self.params.cfraction))
        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.ann.set_cman(case_man)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()

    def glass(self):
        case_generator = (lambda: load_generic_file('data/glass.txt', self.params.cfraction))
        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.ann.set_cman(case_man)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()

    def iris(self):
        case_generator = (lambda: load_iris_file('data/iris.txt', self.params.cfraction))
        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.ann.set_cman(case_man)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()

    def mnist(self):
        case_generator = (
            lambda: load_flat_text_cases('data/all_flat_mnist_training_cases_text.txt', self.params.cfraction))
        case_man = Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac)
        self.ann.set_cman(case_man)
        self.params.dims[0] = len(case_man.training_cases[0][0])
        self.params.dims[-1] = len(case_man.training_cases[0][1])
        print("\nNumber of bits taken from input layer: ", self.params.dims[0],
              "and output set to target vector length at: ", self.params.dims[-1])
        model = self.build_ann()
        self.ann.set_model(model)
        model.run(steps=self.params.steps, bestk=self.params.bestk)
        self.check_mapping_and_dendro()
