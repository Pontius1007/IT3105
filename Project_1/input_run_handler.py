from ann import *
import tflowtools as TFT


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

		# For grabbed variables
		self.grabbed_weights = []
		self.grabbed_biases = []

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
			if  data_input == "countex":
				self.countex()
			elif data_input == "yeast":
				self.yeast()
			elif data_input == "hey":
				print("heyy")

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
		self.params.bestk = 1 if (str(data["bestk"]["bool"].lower()) == "true") else 0

	def countex(self):
		nbits = int(input("Enter the length of the vector in bits. Enter 0 to set it to the input layer size: "))
		nbits = nbits if (nbits != 0) else self.params.dims[0]
		case_generator = (lambda: TFT.gen_vector_count_cases(self.params.ncases, nbits))
		self.ann.set_cman(Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac))
		model = Gann(dims=self.params.dims, hidden_activation_function=self.params.hidden_activation_function,
					 optimizer=self.params.optimizer, lower=self.params.weight_range_lower,
					 upper=self.params.weight_range_upper, cman=self.ann.get_cman(), lrate=self.params.learning_rate,
					 showfreq=self.params.show_freq, mbs=self.params.mbs, vint=self.params.vint, softmax=self.params.sm,
					 cost_function=self.params.cost_function)
		self.ann.set_model(model)
		model.run(steps=self.params.steps, bestk=self.params.bestk)
		# TFT.fireup_tensorboard('probeview')

	def yeast(self):
	# case_generator = load_flat_text_cases('all_flat_mnist_training_cases_text.txt', 0.01)
	# print(load_flat_text_cases('all_flat_mnist_training_cases_text.txt', 0.001))
	# case_generator = (lambda: load_flat_text_cases('all_flat_mnist_training_cases_text.txt'))
	# case_generator = (lambda: TFT.gen_all_one_hot_cases(2**4))
		case_generator = (lambda: load_generic_file('data/yeast.txt', self.params.cfraction))
		self.ann.set_cman(Caseman(cfunc=case_generator, vfrac=self.params.vfrac, tfrac=self.params.tfrac))

		model = Gann(dims=self.params.dims, hidden_activation_function=self.params.hidden_activation_function,
					 optimizer=self.params.optimizer, lower=self.params.weight_range_lower,
					 upper=self.params.weight_range_upper, cman=self.ann.get_cman(), lrate=self.params.learning_rate,
					 showfreq=self.params.show_freq, mbs=self.params.mbs, vint=self.params.vint, softmax=self.params.sm,
					 cost_function=self.params.cost_function)
		self.ann.set_model(model)
		model.run(steps=self.params.steps, bestk=self.params.bestk)
	# TFT.fireup_tensorboard('probeview')
