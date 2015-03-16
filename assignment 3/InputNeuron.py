from Neuron import Neuron

class InputNeuron(Neuron):

    def __init__(self, weights=None):
        super(InputNeuron, self).__init__(weights)

    def activation_function(self, xs):
        return xs