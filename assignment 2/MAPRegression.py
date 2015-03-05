from __future__ import division
import numpy as np
import Regression as Regression


class MAPRegression(Regression.Regression):

    def __init__(self, alpha, beta, d_mat, t_vec):
        super(MAPRegression, self).__init__(d_mat, t_vec)

        S_N = np.linalg.inv(alpha * np.identity(self.d_mat.shape[1]) + beta * np.dot(self.d_mat.T, self.d_mat))

        M_N = beta * np.dot(np.dot(S_N, self.d_mat.T), self.t_vec)
        self.w = M_N

    def predict(self, x):
        return np.dot(self.w.T, self.d_mat[x])[0]