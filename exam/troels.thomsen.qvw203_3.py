from DataReader import DataReader
from matplotlib import pyplot as plt
import numpy as np
from LDAClassifier import LDAClassifier
from NNClassifier import NNClassifier
from Normalizer import Normalizer

training_dataset = DataReader.read_data('data/keystrokesTrainTwoClass.csv', ',')
test_dataset = DataReader.read_data('data/keystrokesTestTwoClass.csv', ',')

normalizer = Normalizer(training_dataset)

# We dont use normalized dataset currently. It does not improve performance.
#normalized_training_dataset = normalizer.normalize_means(training_dataset)
#normalized_test_dataset = normalizer.normalize_means(test_dataset)

training_lda = LDAClassifier(training_dataset)

training_lda_classified = training_lda.classify_dataset(training_dataset)
test_lda_classified = training_lda.classify_dataset(test_dataset)

training_lda_accuracy = LDAClassifier.find_accuracy(training_dataset, training_lda_classified)
test_lda_accuracy = LDAClassifier.find_accuracy(test_dataset, test_lda_classified)

training_lda_sensitivity, training_lda_specificity = LDAClassifier.find_sensispevity(training_dataset, training_lda_classified)
test_lda_sensitivity, test_lda_specificity = LDAClassifier.find_sensispevity(test_dataset, test_lda_classified)

print "\nTraining and test accuracy for LDA:"
print training_lda_accuracy
print test_lda_accuracy

print "\nLDA training sensitivity and specificity:"
print training_lda_sensitivity
print training_lda_specificity

print "\nLDA test sensitivity and specificity:"
print test_lda_sensitivity
print test_lda_specificity

training_nn = NNClassifier(training_dataset)

best_k = training_nn.cross_validate()

training_nn_classified = training_nn.classify_dataset(best_k, training_dataset)
test_nn_classified = training_nn.classify_dataset(best_k, test_dataset)

training_nn_accuracy = NNClassifier.find_accuracy(training_dataset, training_nn_classified)
test_nn_accuracy = NNClassifier.find_accuracy(test_dataset, test_nn_classified)

training_nn_sensitivity, training_nn_specificity = NNClassifier.find_sensispevity(training_dataset, training_nn_classified)
test_nn_sensitivity, test_nn_specificity = NNClassifier.find_sensispevity(test_dataset, test_nn_classified)

print "\nKNN classifier best k:" + str(best_k)
print "KNN traing and test accuracy"
print training_nn_accuracy
print test_nn_accuracy

print "\nNN training sensitivity and specificity:"
print training_nn_sensitivity
print training_nn_specificity

print "\nNN test sensitivity and specificity:"
print test_nn_sensitivity
print test_nn_specificity