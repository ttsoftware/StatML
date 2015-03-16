import numpy as np


class Neuron(object):

    def __init__(self, ws=None):
        # starting weights for this neuron
        self.ws = ws
        if type(ws) == list:
            self.ws = ws[:]

    def activation_function(self, xs):
        if xs is None:
            raise Exception('Not yet implemented')

        if self.ws is None:
            # +1 for bias
            #xs = xs[:]
            #xs += [1]
            # weights from 0 to N.
            self.ws = np.random.normal(size=(1, len(xs))).tolist()[0]
