from __future__ import division
import numpy as np
from DataPoint import DataPoint
from DataSet import DataSet

class Regression(object):

	def __init__(self, dataset):

		self.dataset = dataset


	@staticmethod
	def regression(xset, yset):
		xset = np.array(xset)
		yset = np.array(yset)

		return np.dot(np.linalg.pinv(xset), yset)

	def reguession(self, w_ml, guess):
		return sum([[1, self.dataset[(guess - (2**i)) - 1716]] for i in range(5)]) / 5
