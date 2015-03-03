from DataReader import DataReader
from Gauss import Gauss
from LDAClassifier import LDAClassifier
from Normalizer import Normalizer
from MLRegression import MLRegression
from MAPRegression import MAPRegression
import numpy as np
import matplotlib.pyplot as plt

training_dataset = DataReader.read_data('IrisTrain2014.dt')
test_dataset = DataReader.read_data('IrisTest2014.dt')

lda = LDAClassifier(training_dataset)

training_classified = lda.classify_dataset(training_dataset)
test_classified = lda.classify_dataset(test_dataset)

training_accuracy = LDAClassifier.find_accuracy(training_dataset, training_classified)
test_accuracy = LDAClassifier.find_accuracy(test_dataset, test_classified)

print 'Standard training set accuracy: ' + str(training_accuracy)
print 'Standard test set accuracy: ' + str(test_accuracy)

############################# Normalized ###############################

normalizer = Normalizer(training_dataset)

normalized_training_dataset = normalizer.normalize_means(training_dataset)
normalized_test_dataset = normalizer.normalize_means(test_dataset)

normalized_lda = LDAClassifier(normalized_training_dataset)

normalized_training_classified = normalized_lda.classify_dataset(normalized_training_dataset)
normalized_test_classified = normalized_lda.classify_dataset(normalized_test_dataset)

normalized_training_accuracy = LDAClassifier.find_accuracy(training_dataset, normalized_training_classified)
normalized_test_accuracy = LDAClassifier.find_accuracy(test_dataset, normalized_test_classified)

print 'Normalized training set accuracy: ' + str(normalized_training_accuracy)
print 'Normalized test set accuracy: ' + str(normalized_test_accuracy)

############################# Sunspot prediction ###############################

sunspot_training_dataset = DataReader.read_data("sunspotsTrainStatML.dt")
sunspot_test_dataset = DataReader.read_data("sunspotsTestStatML.dt")

# Training
selection1_training = map(lambda x: [x.params[2], x.params[3]], sunspot_training_dataset)
selection2_training = map(lambda x: [x.params[4]], sunspot_training_dataset)
selection3_training = map(lambda x: x.params, sunspot_training_dataset)
target_training = map(lambda x: [x.target], sunspot_training_dataset)

# Test
selection1_test = map(lambda x: [x.params[2], x.params[3]], sunspot_test_dataset)
selection2_test = map(lambda x: [x.params[4]], sunspot_test_dataset)
selection3_test = map(lambda x: x.params, sunspot_test_dataset)
target_test = map(lambda x: [x.target], sunspot_test_dataset)

# ML
ml_regression_train1 = MLRegression(selection1_training, target_training)
ml_regression_train2 = MLRegression(selection2_training, target_training)
ml_regression_train3 = MLRegression(selection3_training, target_training)

ml_regression_test1 = MLRegression(selection1_test, target_test)
ml_regression_test2 = MLRegression(selection2_test, target_test)
ml_regression_test3 = MLRegression(selection3_test, target_test)

ml_guess_training1 = [ml_regression_train1.predict(x) for x in range(200)]
ml_guess_training2 = [ml_regression_train2.predict(x) for x in range(200)]
ml_guess_training3 = [ml_regression_train3.predict(x) for x in range(200)]

ml_guess_test1 = [ml_regression_test1.predict(x) for x in range(96)]
ml_guess_test2 = [ml_regression_test2.predict(x) for x in range(96)]
ml_guess_test3 = [ml_regression_test3.predict(x) for x in range(96)]

"""
plt.figure("ML Selection 1 (test)")
plt.plot(range(1916, 2012), target_test, color='b', label="Actual test data")
plt.plot(range(1916, 2012), ml_guess_test1, color='r', label="Predicted data")
plt.legend(loc="upper left")
plt.show()

plt.figure("ML Selection 2 (training and test)")
plt.plot(range(1716, 1916), target_training, color='b', label="Actual training data")
plt.plot(range(1716, 1916), ml_guess_training2, color='r', label="Predicted data")
plt.plot(range(1916, 2012), target_test, color='black', label="Actual test data")
plt.plot(range(1916, 2012), ml_guess_test2, color='y')
plt.legend(loc="upper left")
plt.show()

plt.figure("ML Selection 3 (test)")
plt.plot(range(1916, 2012), target_test, color='b', label="Actual test data")
plt.plot(range(1916, 2012), ml_guess_test3, color='r', label="Predicted data")
plt.legend(loc="upper left")
plt.show()
"""

print "RMS for ml selection1: " + str(ml_regression_test1.root_mean_square())
print "RMS for ml selection2: " + str(ml_regression_test2.root_mean_square())
print "RMS for ml selection3: " + str(ml_regression_test3.root_mean_square())

# MAP
alpha = 0.5
beta = 1

map_regression_train1 = MAPRegression(alpha, beta, selection1_training, target_training)
map_regression_train2 = MAPRegression(alpha, beta, selection2_training, target_training)
map_regression_train3 = MAPRegression(alpha, beta, selection3_training, target_training)

map_regression_test1 = MAPRegression(alpha, beta, selection1_test, target_test)
map_regression_test2 = MAPRegression(alpha, beta, selection2_test, target_test)
map_regression_test3 = MAPRegression(alpha, beta, selection3_test, target_test)

map_guess_training1 = [map_regression_train1.predict(x) for x in range(200)]
map_guess_training2 = [map_regression_train2.predict(x) for x in range(200)]
map_guess_training3 = [map_regression_train3.predict(x) for x in range(200)]

map_guess_test1 = [map_regression_test1.predict(x) for x in range(96)]
map_guess_test2 = [map_regression_test2.predict(x) for x in range(96)]
map_guess_test3 = [map_regression_test3.predict(x) for x in range(96)]

"""
plt.figure("MAP Selection 1 (test)")
plt.plot(range(1916, 2012), target_test, color='b', label="Actual test data")
plt.plot(range(1916, 2012), map_guess_test1, color='r', label="Predicted data")
plt.legend(loc="upper left")
plt.show()

plt.figure("MAP Selection 2 (training and test)")
plt.plot(range(1716, 1916), target_training, color='b', label="Actual training data")
plt.plot(range(1716, 1916), map_guess_training2, color='r', label="Predicted data")
plt.plot(range(1916, 2012), target_test, color='b', label="Actual test data")
plt.plot(range(1916, 2012), map_guess_test2, color='r')
plt.legend(loc="upper left")
plt.show()

plt.figure("MAP Selection 3 (test)")
plt.plot(range(1916, 2012), target_test, color='b', label="Actual test data")
plt.plot(range(1916, 2012), map_guess_test3, color='r', label="Predicted data")
plt.legend(loc="upper left")
plt.show()
"""

print "RMS for map selection1: " + str(map_regression_test1.root_mean_square())
print "RMS for map selection2: " + str(map_regression_test2.root_mean_square())
print "RMS for map selection3: " + str(map_regression_test3.root_mean_square())