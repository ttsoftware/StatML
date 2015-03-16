from OutputNeuron import OutputNeuron
from NeuronLayer import NeuronLayer


class OutputLayer(NeuronLayer):

    def __init__(self, neuron_count):
        super(OutputLayer, self).__init__()

        for d in range(neuron_count):
            self.neurons += [OutputNeuron()]