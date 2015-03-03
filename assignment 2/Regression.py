from __future__ import division
import numpy as np
from DataPoint import DataPoint
from DataSet import DataSet


class Regression(object):
    def __init__(self, d_mat, t_vec):

        self.d_mat = map(lambda x: [1] + x, d_mat)
        self.t_vec = t_vec

        self.w_ml = self.regression()


    def regression(self):
        return np.dot(np.linalg.pinv(self.d_mat), self.t_vec)

    def reguession(self, guess):
        return np.dot(self.w_ml.T, self.d_mat[guess])[0]