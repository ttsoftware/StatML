import numpy as np


class DataPoint(object):

    def __init__(self, params, label=None):
        if label != None:
            self.params = map(lambda x: float(x), params)
            self.label = float(label)
        else:
            self.label = float(params.pop())
            self.params = map(lambda x: float(x), params)

    def get_vector(self):
        return np.array(map(lambda x: [x], self.params))