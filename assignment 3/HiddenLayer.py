from HiddenNeuron import HiddenNeuron
from NeuronLayer import NeuronLayer


class HiddenLayer(NeuronLayer):

    def __init__(self, neuron_count):
        super(HiddenLayer, self).__init__()

        #self.neurons += [HiddenNeuron([-1.3630613033632497, -0.8022794198756549])]
        #self.neurons += [HiddenNeuron([0.0973375695344754, 0.24311309588048313])]

        for d in range(neuron_count):
            self.neurons += [HiddenNeuron()]