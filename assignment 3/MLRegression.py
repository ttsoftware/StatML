from __future__ import division
import numpy as np
from DataPoint import DataPoint
from DataSet import DataSet
import Regression as Regression


class MLRegression(Regression.Regression):

    def __init__(self, d_mat, t_vec):
        super(MLRegression, self).__init__(d_mat, t_vec)

        self.w = self.regression()

    def regression(self):
        """
        Returns the dot product of the pseudo-inverse of the design matrix,
        and the target vector.
        """
        return np.dot(np.linalg.pinv(self.d_mat), self.t_vec)

    def predict(self, x):
        """
        Returns the dot product of the transposed w, and the design matrix
        row corresponding to the given guess.
        """
        return sum(np.dot(self.w.T, self.d_mat[x]))