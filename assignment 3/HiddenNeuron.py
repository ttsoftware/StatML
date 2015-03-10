from __future__ import division


class HiddenNeuron(object):

    def __init__(self):
        pass

    def h(self, a):
        return a / (1+abs(a))

    def h_(self, a):
        return 1 / (1+abs(a))**2