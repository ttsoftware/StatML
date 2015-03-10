import numpy as np


class Neuron(object):

    def __init__(self, xs):
        # starting weights for this neuron
        ws = np.random.normal(size=(len(xs), 1))
        ws.append(1)
        xs.append(1)
        self.ws = ws
        self.xs = xs

    def a(self):
        accum = 0
        for j in range(len(self.xs)):
            accum += self.ws[j] * self.xs[j]
        return accum
