import numpy as np


class DataPoint(object):

    def __init__(self, params, target=None):
        if target != None:
            self.params = map(lambda x: float(x), params)
            self.target = float(target)
        else:
            self.target = float(params.pop())
            self.params = map(lambda x: float(x), params)

    def get_vector(self):
        return np.array(map(lambda x: [x], self.params))