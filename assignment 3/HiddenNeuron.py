from __future__ import division
from Neuron import Neuron


class HiddenNeuron(Neuron):

    def __init__(self, weights=None):
        super(HiddenNeuron, self).__init__(weights)

    def activation_function(self, xs):
        super(HiddenNeuron, self).activation_function(xs)
        a = 0
        for j in range(len(xs)):
            a += self.ws[j] * xs[j]

        return a / (1 + abs(a))

    def activation_function_derivative(self, a):
        return 1 / (1 + abs(a))**2