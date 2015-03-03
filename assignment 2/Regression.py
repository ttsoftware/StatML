from __future__ import division
import numpy as np
from DataPoint import DataPoint
from DataSet import DataSet

class Regression(object):

	def __init__(self, dataset, keys):

		self.dataset = {}

		for i in range(len(keys)):
			self.dataset[keys[i]] = dataset[i]

		self.phiset = np.array(map(lambda x: [1, x], keys))

		self.w_ml = self.regression()


	def regression(self):
		T = np.array(self.dataset.values())

		return np.dot(np.linalg.pinv(self.phiset), T)

	#def reguession(self, w_ml, guess):
	#	return sum([[1, self.dataset[(guess - (2**i)) - 1716]] for i in range(5)]) / 5
