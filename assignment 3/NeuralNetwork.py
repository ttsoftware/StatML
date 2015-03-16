from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
import numpy as np


class NeuralNetwork(object):

    def __init__(self, input_neuron_count,
                 hidden_neuron_count,
                 output_neuron_count):

        self.input_layer = InputLayer(input_neuron_count)
        self.hidden_layer = HiddenLayer(hidden_neuron_count)
        self.output_layer = OutputLayer(output_neuron_count)

    def skub(self, dataset):
        """
        Push the given dataset through the network
        :param dataset:
        """
        values = dataset.unpack_params()
        targets = dataset.unpack_targets()

        input_zs = {}
        hidden_zs = {}

        # for each data point
        for j, params in enumerate(values):
            if j not in input_zs.keys():
                input_zs[j] = []

            # for each input neuron, get z's
            for k, output_neuron in enumerate(self.input_layer.neurons):
                input_zs[j] += [output_neuron.activation_function(params[k])]

        print "Input z's"
        print input_zs

        # for each input z
        for j, z in input_zs.iteritems():
            if j not in hidden_zs.keys():
                hidden_zs[j] = []

            # for each hidden neuron, get next z's
            for k, output_neuron in enumerate(self.hidden_layer.neurons):
                hidden_zs[j] += [output_neuron.activation_function(z)]

        hidden_weights = self.hidden_layer.get_weights()

        print "\nHidden z's"
        print hidden_zs

        print "\nHidden weights: "
        print hidden_weights

        ys = {}
        # for each hidden z
        for j, z in hidden_zs.iteritems():
            if j not in ys.keys():
                ys[j] = []

            for k, output_neuron in enumerate(self.output_layer.neurons):
                ys[j] += [output_neuron.activation_function(z)]

        output_weights = self.output_layer.get_weights()

        print "\nOutput weights: "
        print output_weights

        print "\nY's:"
        print ys

        E_errors, delta_ks = self.error(ys, targets)

        print "\nE and Delta k:"
        print E_errors
        print delta_ks

        delta_js = {}

        # for each hidden z
        for j, hidden_neuron in enumerate(self.hidden_layer.neurons):
            if j not in delta_js.keys():
                delta_js[j] = []

            zj = hidden_neuron.activation_function_derivative(hidden_zs[j])

            # for each output neuron
            for k, output_neuron in enumerate(self.output_layer.neurons):

                delta_js[j] += [zj * output_neuron.ws[j] * delta_ks[j][k]]

        print "\nDelta j:"
        print delta_js

        derive_j = {}
        for j, hidden_neuron in enumerate(self.hidden_layer.neurons):
            if j not in derive_j.keys():
                derive_j[j] = []
            for i, z in enumerate(input_zs[j]):
                for k, delta_j in delta_js.iteritems():
                    derive_j[j] += [delta_j[j] * z]

        derive_k = {}
        for k, output_neuron in enumerate(self.output_layer.neurons):
            if k not in derive_k.keys():
                derive_k[k] = []
            for j, z in enumerate(hidden_zs[k]):
                for i, delta_k in delta_ks.iteritems():
                    derive_k[k] += [delta_k[k] * z]

        print "\nDerive j & k:"
        print derive_j
        print derive_k

        for j, neuron in enumerate(self.hidden_layer.neurons):

            print neuron.ws

            for i, w in enumerate(neuron.ws):
                neuron.ws[i] += w*derive_j[j][i]

            print neuron.ws

    def error(self, predictions, targets):
        """
        Get the errors squared
        :param predictions:
        :param targets:
        :return:
        """
        delta_errors = {}
        E_errors = {}

        for i, target in enumerate(targets):
            if i not in E_errors.keys():
                E_errors[i] = []
                delta_errors[i] = []

            # try all preditions for given input
            for j, prediction in enumerate(predictions[i]):
                delta_errors[i] += [prediction - target]
                E_errors[i] += [(prediction - target) ** 2]

            map(lambda x: x/2, E_errors)

        return E_errors, delta_errors