from HiddenNeuron import HiddenNeuron
from NeuronLayer import NeuronLayer


class HiddenLayer(NeuronLayer):

    def __init__(self, neuron_count):
        super(HiddenLayer, self).__init__()

        for d in range(neuron_count):
            self.neurons += [HiddenNeuron()]