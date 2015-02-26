import numpy as np


class DataPoint(object):

    def __init__(self, params, label=None):
        self.params = params
        self.label = label

    def get_vector(self):
        return np.array(map(lambda x: [x], self.params))