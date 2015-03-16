from DataReader import DataReader
from NeuralNetwork import NeuralNetwork
from Normalizer import Normalizer
import numpy as np
from sklearn import svm
import math

############### 1.1 ################

training_dataset = DataReader.read_data('data/sincTrain25.dt')

network = NeuralNetwork(1, 1, 1)
network.skub(training_dataset)

exit()

############## 2.1 ################

# Normalization
parkinson_training = DataReader.read_data("data/parkinsonsTrainStatML.dt")
parkinson_test = DataReader.read_data("data/parkinsonsTestStatML.dt")

normalizer = Normalizer(parkinson_training)

print normalizer.dimensions_means
print normalizer.variance()

parkinson_train_normalized = normalizer.normalize_means(parkinson_training)
parkinson_test_normalized = normalizer.normalize_means(parkinson_test)

test_normalizer = Normalizer(parkinson_test_normalized)
train_normalizer = Normalizer(parkinson_train_normalized)
print test_normalizer.dimensions_means
print test_normalizer.variance()

print "\n", train_normalizer.variance()

# SVM

