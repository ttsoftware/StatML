from __future__ import division
import numpy as np


class MAPRegression(object):

    def __init__(self, alpha, beta, design_matrix, target_matrix):

        design_matrix = np.array(map(lambda x: [1] + x, design_matrix))
        target_matrix = np.array(target_matrix)

        self.design_matrix = design_matrix

        S_N = np.linalg.inv(alpha*np.identity(design_matrix.shape[1]) + beta * np.dot(design_matrix.T, design_matrix))

        M_N = beta * np.dot(np.dot(S_N, design_matrix.T), target_matrix)
        self.w_map = M_N

    def guess(self, x):
        return np.dot(self.w_map.T, self.design_matrix[x])[0]