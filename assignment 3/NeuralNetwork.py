import math
from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
import numpy as np


class NeuralNetwork(object):

    def __init__(self, input_neuron_count,
                 hidden_neuron_count,
                 output_neuron_count):

        self.epsilon = 0.00001

        self.input_layer = InputLayer(input_neuron_count)
        self.hidden_layer = HiddenLayer(hidden_neuron_count)
        self.output_layer = OutputLayer(output_neuron_count)

    def skub(self, dataset):
        """
        Push the given dataset through the network
        :param dataset:
        """
        targets = dataset.unpack_targets()

        final_results = {}

        for i in range(0, 100):

            error = {}

            yss = {}
            input_zss = {}
            hidden_zss = {}

            for j, data_point in enumerate(dataset):
                if j not in error.keys():
                    error[j] = 0
                    yss[j] = []
                    input_zss[j] = []
                    hidden_zss[j] = []

                ys, input_zs, hidden_zs = self.push_through(data_point.params)
                delta_js, delta_ks = self.backpropagate(ys, [data_point.target], input_zs, hidden_zs)

                yss[j] = ys
                input_zss[j] = input_zs
                hidden_zss[j] = hidden_zs

                error_new = data_point.target - ys[0]

                print data_point.target, ys[0], error_new

                if abs(error[j]) - abs(error_new) < 0.1:
                    final_results[j] = ys[0]
                    continue

                error[j] = error_new

                if math.isnan(ys[0]):
                    final_results[j] = ys[0]
                    continue

            # backpropagate
            #for j, data_point in enumerate(dataset):
                #delta_js, delta_ks = self.backpropagate(yss[j], [data_point.target], input_zss[j], hidden_zss[j])

        print ""

        print self.hidden_layer.get_weights()
        print self.output_layer.get_weights()

        print targets
        print final_results.values()

    def push_through(self, params):

        input_zs = {}
        hidden_zs = {}

        # for each data point
        for i, param in enumerate(params):
            # for each input neuron, get z's
            input_zs[i] = self.input_layer.neurons[i].activation_function(param)

        #print "Input z's"
        #print input_zs

        # for each hidden neuron, get next z's
        for j, hidden_neuron in enumerate(self.hidden_layer.neurons):
            hidden_zs[j] = hidden_neuron.activation_function(input_zs.values() + self.hidden_layer.bias)

        hidden_weights = self.hidden_layer.get_weights()

        #print "\nHidden z's"
        #print hidden_zs

        #print "\nHidden weights: "
        #print hidden_weights

        ys = {}

        # for each output neuron
        for k, output_neuron in enumerate(self.output_layer.neurons):
            ys[k] = output_neuron.activation_function(hidden_zs.values() + self.output_layer.bias)

        output_weights = self.output_layer.get_weights()

        #print "\nOutput weights: "
        #print output_weights

        #print "\nY's:"
        #print ys

        return ys, input_zs, hidden_zs

    def backpropagate(self, ys, targets, input_zs, hidden_zs):

        E_errors, delta_ks = self.error(ys, targets)

        #print "\nE and Delta k:"
        #print E_errors
        #print delta_ks

        delta_js = {}

        # for each hidden neuron
        for j, hidden_neuron in enumerate(self.hidden_layer.neurons):

            delta_js[j] = 0
            zj = hidden_neuron.activation_function_derivative(hidden_zs[j])

            # for each output neuron
            for k, output_neuron in enumerate(self.output_layer.neurons):
                delta_js[j] += output_neuron.ws[j] * delta_ks[k]

            delta_js[j] *= zj

        #print "\nDelta j:"
        #print delta_js

        #print "\nDerive j & k:"

        derive_j = {}
        for j, hidden_neuron in enumerate(self.hidden_layer.neurons):
            if j not in derive_j.keys():
                derive_j[j] = []

            for i, z in enumerate(input_zs.values() + self.hidden_layer.bias):
                derive_j[j] += [delta_js[j] * z]

        #print derive_j

        derive_k = {}
        for k, output_neuron in enumerate(self.output_layer.neurons):
            if k not in derive_k.keys():
                derive_k[k] = []

            for j, z in enumerate(hidden_zs.values() + self.output_layer.bias):
                derive_k[k] += [delta_ks[k] * z]

        #print derive_k

        # here we modify the weights directly
        for j, neuron in enumerate(self.hidden_layer.neurons):

            for i, w in enumerate(neuron.ws):
                neuron.ws[i] -= w*derive_j[j][i] + self.epsilon

        for k, neuron in enumerate(self.output_layer.neurons):

            for i, w in enumerate(neuron.ws):
                neuron.ws[i] -= w*derive_k[k][i] + self.epsilon

        return delta_js, delta_ks

    def error(self, predictions, targets):
        """
        Get the errors squared
        :param predictions:
        :param targets:
        :return:
        """
        delta_errors = {}
        E_errors = {}

        for k, prediction in predictions.iteritems():

            delta_errors[k] = prediction - targets[k]
            E_errors[k] = (prediction - targets[k]) ** 2

        #map(lambda x: x/2, E_errors)

        return E_errors, delta_errors