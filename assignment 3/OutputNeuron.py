from Neuron import Neuron


class OutputNeuron(Neuron):

    def __init__(self, weights=None):
        super(OutputNeuron, self).__init__(weights)

    def activation_function(self, xs):
        super(OutputNeuron, self).activation_function(xs)
        a = 0
        for j in range(len(xs)):
            a += self.ws[j] * xs[j]

        return a / (1 + abs(a))