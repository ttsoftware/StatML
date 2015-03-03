from __future__ import division
import numpy as np
from DataPoint import DataPoint
from DataSet import DataSet


class Regression(object):
    def __init__(self, d_mat, t_vec):
        """
        prepends 1's onto each row in the design matrix, and runs regressions,
        to find w_ml.

        :param d_mat (Design matric):
        :param t_vec (Target vector:
        """

        self.d_mat = map(lambda x: [1] + x, d_mat)
        self.t_vec = t_vec
        self.w_ml = self.regression()


    def regression(self):
        """
        Returns the dot product of the pseudo-inverse of the design matrix,
        and the target vector.
        """
        return np.dot(np.linalg.pinv(self.d_mat), self.t_vec)


    def reguession(self, guess):
        """
        Returns the dot product of the transposed w_ml, and the design matrix
        row corresponding to the given guess.
        """
        return sum(np.dot(self.w_ml.T, self.d_mat[guess]))

