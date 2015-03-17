

class NeuronLayer(object):

    def __init__(self):
        self.neurons = []
        self.bias = [1]

    def get_weights(self):
        """
        Get all weights indexed by neuron
        :return:
        """
        weights = {}
        for i, neuron in enumerate(self.neurons):
            if i not in weights.keys():
                weights[i] = []
            weights[i] += neuron.ws

        return weights