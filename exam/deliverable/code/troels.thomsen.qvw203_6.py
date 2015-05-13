from DataReader import DataReader
from matplotlib import pyplot as plt
import numpy as np
from LDAClassifier import LDAClassifier
from NNClassifier import NNClassifier
from Normalizer import Normalizer

training_dataset = DataReader.read_data('data/keystrokesTrainMulti.csv', ',')
test_dataset = DataReader.read_data('data/keystrokesTestMulti.csv', ',')

training_lda = LDAClassifier(training_dataset)

training_lda_classified = training_lda.classify_dataset(training_dataset)
test_lda_classified = training_lda.classify_dataset(test_dataset)

training_lda_accuracy = LDAClassifier.find_accuracy(training_dataset, training_lda_classified)
test_lda_accuracy = LDAClassifier.find_accuracy(test_dataset, test_lda_classified)

print "\nTraining and test accuracy for LDA:"
print training_lda_accuracy
print test_lda_accuracy

training_nn = NNClassifier(training_dataset)

#best_k = training_nn.cross_validate()
# we found best k to be 1.
best_k = 1

training_nn_classified = training_nn.classify_dataset(best_k, training_dataset)
test_nn_classified = training_nn.classify_dataset(best_k, test_dataset)

training_nn_accuracy = NNClassifier.find_accuracy(training_dataset, training_nn_classified)
test_nn_accuracy = NNClassifier.find_accuracy(test_dataset, test_nn_classified)

print "\nKNN classifier best k:" + str(best_k)
print "KNN traing and test accuracy"
print training_nn_accuracy
print test_nn_accuracy
