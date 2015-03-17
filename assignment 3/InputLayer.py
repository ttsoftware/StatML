from InputNeuron import InputNeuron
from NeuronLayer import NeuronLayer


class InputLayer(NeuronLayer):

    def __init__(self, neuron_count):
        super(InputLayer, self).__init__()

        for d in range(neuron_count):
            self.neurons += [InputNeuron()]