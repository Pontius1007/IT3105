import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
import json
import random
from random import shuffle
from itertools import cycle


# ******* A General Artificial Neural Network ********
# This is the original GANN, which has been improved in the file gann.py


class Gann:
    def __init__(self, dims, hidden_activation_function, optimizer, lower, upper, cman, grab_module_index, grab_type,
                 lrate=.1, showfreq=None, mbs=10, vint=None, softmax=False, cost_function="MSE"):
        self.learning_rate = lrate
        self.layer_sizes = dims  # Sizes of each layer of neurons
        self.show_interval = showfreq  # Frequency of showing grabbed variables
        self.global_training_step = 0  # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = []  # One matplotlib figure for each grabvar
        self.optimizer = optimizer
        self.hidden_activation_function = hidden_activation_function
        self.lower = lower
        self.upper = upper
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = cman
        self.softmax_outputs = softmax
        self.modules = []
        self.cost_function = cost_function
        self.build(grab_module_index, grab_type)

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type, spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self, module_index, type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        # if type != 'bias':
        #     self.grabvar_figures.append(PLT.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self, module):
        self.modules.append(module)

    def build(self, grab_module_index, grab_type):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input
        insize = num_inputs
        # Build all of the modules
        for i, outsize in enumerate(self.layer_sizes[1:]):
            gmod = Gannmodule(self, self.hidden_activation_function, i, invar, insize, outsize, self.upper, self.lower)
            invar = gmod.output
            insize = gmod.outsize
        self.output = gmod.output  # Output of last module is output of whole network
        if self.softmax_outputs: self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64, shape=(None, gmod.outsize), name='Target')
        self.configure_learning(self.cost_function)
        for i in range(len(grab_module_index)):
            self.add_grabvar(grab_module_index[i], grab_type[i])

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self, cost_function):
        if cost_function.upper() == "CE":
            self.error = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.target),
                name='Cross-Entropy')
        else:
            self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        # Basic gradient descent is the default
        if self.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            print("You have chosen BGD as optimizer")
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    def do_training(self, sess, cases, steps, continued=False):
        if not continued: self.error_history = []
        error = 0
        gvars = [self.error] + self.grabvars
        mbs = self.minibatch_size
        ncases = len(cases)
        nmb = math.ceil(ncases / mbs)
        start_index = 0
        end_index = mbs
        # The way we select minibatches might be subject to change.
        for cstart in range(0, steps):  # Loops through steps and sends one minibatch through per iteration
            step = self.global_training_step + cstart
            minibatch = cases[start_index:end_index]
            if end_index >= ncases:
                start_index = 0
                end_index = mbs
                np.random.shuffle(cases)
            else:
                start_index = end_index
                end_index += mbs
            inputs = [c[0] for c in minibatch]
            targets = [c[1] for c in minibatch]
            feeder = {self.input: inputs, self.target: targets}
            _, grabvals, _ = self.run_one_step([self.trainer], gvars, self.probes, session=sess,
                                               feed_dict=feeder, step=step, show_interval=self.show_interval)
            error += grabvals[0]
            self.error_history.append((step, grabvals[0]))
            self.consider_validation_testing(step, sess)
        self.global_training_step += steps
        TFT.plot_training_history(self.error_history, self.validation_history, xtitle="Steps", ytitle="Error",
                                  title="TRAINING HISTORY", fig=not continued)

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.

    def do_mapping(self, map_layers, number_of_cases, bestk):
        self.reopen_current_session()
        test_cases = self.caseman.get_training_cases()
        cases = test_cases[:number_of_cases]
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        grabvar_names = [grabvar for grabvar in self.grabvars]
        grabvar_layers = []
        for layer in map_layers:
            for grabvar in grabvar_names:
                if ('-' + str(layer) + '-out') in str(grabvar.name):
                    grabvar_layers.append(grabvar)
        for grabvar in grabvar_layers:
            features = []
            for index, case in enumerate(cases):
                self.test_func = self.predictor
                if bestk is not None:
                    self.test_func = self.gen_match_counter(self.predictor,
                                                            [TFT.one_hot_to_int(list(v)) for v in targets], k=bestk)
                testres, grabvals, _ = self.run_one_step(self.test_func, grabvar, self.probes,
                                                         session=self.current_session,
                                                         feed_dict=feeder, show_interval=self.show_interval,
                                                         step=self.global_training_step)
                features.append(grabvals[index])

            TFT.hinton_plot(np.array(features), fig=PLT.figure(), title=grabvar.name + " Activation Levels")

        new_target = np.array(targets)
        TFT.hinton_plot(new_target, fig=PLT.figure(), title="Input Targets")

    def setupSession(self, sess=None, dir="probeview"):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.roundup_probes()
        self.current_session = session

    def do_prediction(self, case,):
        self.setupSession()
        r_input = case
        feeder = {self.input: [r_input]}
        print("The input is: \n", r_input)
        print("The ANN guessed this: \n")
        print(self.current_session.run(self.output, feed_dict=feeder))
        self.close_current_session(view=False)

    def create_dendrogram(self, dendrogram_layers, number_of_cases, bestk):
        self.reopen_current_session()
        test_cases = self.caseman.get_testing_cases()
        cases = test_cases[:number_of_cases]
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        grabvar_names = [grabvar for grabvar in self.grabvars]
        grabvar_layers = []
        for layer in dendrogram_layers:
            for grabvar in grabvar_names:
                if ('-' + str(layer) + '-out') in str(grabvar.name):
                    grabvar_layers.append(grabvar)
        for grabvar in grabvar_layers:
            features = []
            labels = []
            for index, case in enumerate(cases):
                self.test_func = self.predictor
                if bestk is not None:
                    self.test_func = self.gen_match_counter(self.predictor,
                                                            [TFT.one_hot_to_int(list(v)) for v in targets], k=bestk)
                testres, grabvals, _ = self.run_one_step(self.test_func, grabvar, self.probes,
                                                         session=self.current_session,
                                                         feed_dict=feeder, show_interval=self.show_interval,
                                                         step=self.global_training_step)
                labels.append(TFT.bits_to_str(case[1]))
                features.append(grabvals[index])
            TFT.dendrogram(features, labels, title=grabvar.name)

    def do_testing(self, sess, cases, msg='Testing', bestk=None):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, [TFT.one_hot_to_int(list(v)) for v in targets],
                                                    k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                 feed_dict=feeder, show_interval=None)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100 * (testres / len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns an OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k)  # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self, steps, sess=None, dir="probeview", continued=False):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.roundup_probes()  # this call must come AFTER the session is created, else graph is not in tensorboard.
        self.do_training(session, self.caseman.get_training_cases(), steps, continued=continued)

    def testing_session(self, sess, bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg='Final Testing', bestk=bestk)

    def consider_validation_testing(self, epoch, sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess, cases, msg='Validation Testing')
                self.validation_history.append((epoch, error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self, sess, bestk=None):
        self.do_testing(sess, self.caseman.get_training_cases(), msg='Total Training', bestk=bestk)

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                     session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir="probeview")
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars]
        msg = "Grabbed Variables at Step " + str(step)
        # print("\n" + msg, end="\n")
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1 and (
                    'out' not in str(names[i])):  # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v, fig=PLT.figure(), title=names[i] + ' at step ' + str(step))
            elif 'bias' in str(names[i]):
                fig = PLT.figure()
                v_list = v.tolist()
                v_length = len(v_list)
                x_axis = list(range(1, v_length + 1))
                PLT.plot(x_axis, v_list, 'ro')
                for a, b in zip(x_axis, v_list):
                    PLT.text(a, b, str(round(b, 6)))
                PLT.title("Bias at step: " + str(step) + " for layer " + names[i])
                PLT.xlabel("Node")
                PLT.ylabel("Value")
                PLT.draw()
                PLT.pause(0.1)
            else:
                # print(v, end="\n\n")
                pass

    def run(self, steps=100, sess=None, continued=False, bestk=None):
        PLT.ion()
        self.training_session(steps, sess=sess, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk)
        self.testing_session(sess=self.current_session, bestk=bestk)
        self.close_current_session(view=False)
        PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self, epochs=100, bestk=None):
        self.reopen_current_session()
        self.run(epochs, sess=self.current_session, continued=True, bestk=bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self, view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule:

    def __init__(self, ann, h_activation_function, index, invariable, insize, outsize, lower, upper):
        self.ann = ann
        self.insize = insize  # Number of neurons feeding into this module
        self.outsize = outsize  # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-" + str(self.index)
        self.build(h_activation_function, lower, upper)

    def build(self, h_activation_function, lower, upper):
        mona = self.name
        n = self.outsize
        self.weights = tf.Variable(np.random.uniform(lower, upper, size=(self.insize, n)),
                                   name=mona + '-wgt', trainable=True)  # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(lower, upper, size=n),
                                  name=mona + '-bias', trainable=True)  # First bias vector
        if h_activation_function == "relu6":
            self.output = tf.nn.relu6(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        elif h_activation_function == "crelu":
            self.output = tf.nn.crelu(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        elif h_activation_function == "elu":
            self.output = tf.nn.elu(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        elif h_activation_function == "selu":
            self.output = tf.nn.selu(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        elif h_activation_function == "softplus":
            self.output = tf.nn.softplus(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        elif h_activation_function == "softsign":
            self.output = tf.nn.softsign(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        elif h_activation_function == "dropout":
            self.output = tf.nn.dropout(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        elif h_activation_function == "bias_add":
            self.output = tf.nn.bias_add(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        elif h_activation_function == "sigmoid":
            self.output = tf.nn.sigmoid(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        elif h_activation_function == "tanh":
            self.output = tf.nn.tanh(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        else:
            self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        self.ann.add_module(self)

    def getvar(self, type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self, type, spec):
        var = self.getvar(type)
        base = self.name + '_' + type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/', var)


# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

    def __init__(self, cfunc, vfrac=0, tfrac=0):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca)  # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases
