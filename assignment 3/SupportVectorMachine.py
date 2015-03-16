from __future__ import division
import math

class SupportVectorMachine(object):

    def __init__(self, dataset, gamma=0.1, C=10, kernel='rbf'):
        self.dataset = dataset

    gamma = 0.1
C = 10

clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
clf.fit(parkinson_training.unpack_params(), parkinson_training.unpack_targets())

predictions = clf.predict(parkinson_test.unpack_params())

print predictions == parkinson_test.unpack_targets()