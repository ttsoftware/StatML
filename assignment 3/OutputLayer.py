from OutputNeuron import OutputNeuron
from NeuronLayer import NeuronLayer


class OutputLayer(NeuronLayer):

    def __init__(self, neuron_count):
        super(OutputLayer, self).__init__()

        #self.neurons += [OutputNeuron([0.9102906873054797, 1.2018715189445017, 0.15499876265390913])]

        for d in range(neuron_count):
            self.neurons += [OutputNeuron()]