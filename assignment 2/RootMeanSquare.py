import math
from MLRegression import MLRegression
import numpy as np

class RootMeanSquare(object):
    def __init__(self, regression):
        self.reg = regression

    def root_mean_square(self):
        rms_val = sum((self.reg.t_vec[i] - np.dot(self.reg.d_mat[i], self.reg.w_ml))**2 for i in range(len(self.reg.w_ml)))
        return math.sqrt(rms_val/len(self.reg.w_ml))
