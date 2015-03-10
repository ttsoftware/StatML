from DataReader import DataReader
from LDAClassifier import LDAClassifier
from Normalizer import Normalizer
from MLRegression import MLRegression
from MAPRegression import MAPRegression
import matplotlib.pyplot as plt
import numpy as np

training_dataset = DataReader.read_data('IrisTrain2014.dt')
test_dataset = DataReader.read_data('IrisTest2014.dt')

lda = LDAClassifier(training_dataset)

training_classified = lda.classify_dataset(training_dataset)
test_classified = lda.classify_dataset(test_dataset)

training_error = LDAClassifier.find_error(training_dataset, training_classified)
test_error = LDAClassifier.find_error(test_dataset, test_classified)

print 'Standard training set error: ' + str(training_error)
print 'Standard test set error: ' + str(test_error)

############################# Normalized ###############################

normalizer = Normalizer(training_dataset)

normalized_training_dataset = normalizer.normalize_means(training_dataset)
normalized_test_dataset = normalizer.normalize_means(test_dataset)

normalized_lda = LDAClassifier(normalized_training_dataset)

normalized_training_classified = normalized_lda.classify_dataset(normalized_training_dataset)
normalized_test_classified = normalized_lda.classify_dataset(normalized_test_dataset)

normalized_training_error = LDAClassifier.find_error(training_dataset, normalized_training_classified)
normalized_test_error = LDAClassifier.find_error(test_dataset, normalized_test_classified)

print 'Normalized training set error: ' + str(normalized_training_error)
print 'Normalized test set error: ' + str(normalized_test_error)

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

fig = plt.figure("ML prediction")
ax = fig.add_subplot(111)
ax.set_title('ML selection 1, 2, 3 - Test')
ax.set_xlabel("Years")
ax.set_ylabel("Sunspots")
plt.plot(range(1916, 2012), target_test, color='black', label="Actual test data")
plt.plot(range(1916, 2012), ml_guess_test1, color='r', label="Selection 1 prediction")
plt.plot(range(1916, 2012), ml_guess_test2, color='g', label="Selection 2 prediction")
plt.plot(range(1916, 2012), ml_guess_test3, color='b', label="Selection 3 prediction")
plt.legend(loc="upper left")
plt.show()

fig = plt.figure("ML Regression")
ax = fig.add_subplot(111)
ax.set_xlabel("Number of sunspots for year t")
ax.set_ylabel("Number of sunspots for year t-1")
plt.plot(selection2_training, target_training, '.b', label="Training data")
plt.plot(selection2_test, target_test, '.y', label="Test data")
plt.plot(selection2_test, ml_guess_test2, color='r', label="Test prediction")
plt.legend(loc="upper left")
plt.show()

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

alphas = np.arange(0, 500, 1)

RMSs_test1 = []
for a in alphas:
    RMSs_test1.append(MAPRegression(a, beta, selection1_test, target_test).root_mean_square())

RMSs_test2 = []
for a in alphas:
    RMSs_test2.append(MAPRegression(a, beta, selection2_test, target_test).root_mean_square())

RMSs_test3 = []
for a in alphas:
    RMSs_test3.append(MAPRegression(a, beta, selection3_test, target_test).root_mean_square())

ml_RMS1 = ml_regression_test1.root_mean_square()
ml_RMS2 = ml_regression_test2.root_mean_square()
ml_RMS3 = ml_regression_test3.root_mean_square()

fig = plt.figure("RMS values")
ax1 = fig.add_subplot(311)
fig.tight_layout()
ax1.set_title('Selection 1')
ax1.set_xlabel("Alpha values")
ax1.set_ylabel("Root mean square")
plt.plot(alphas, RMSs_test1, color='b', label="MAP RMS values (selection 1)")
plt.plot(alphas, [ml_RMS1 for a in alphas], color='r', label="ML RMS (selection 1)")
plt.legend(loc="right")

ax2 = fig.add_subplot(312)
ax2.set_title('Selection 2')
ax2.set_xlabel("Alpha values")
ax2.set_ylabel("Root mean square")
plt.plot(alphas, RMSs_test2, color='b', label="MAP RMS values (selection 2)")
plt.plot(alphas, [ml_RMS2 for a in alphas], color='r', label="ML RMS (selection 2)")
plt.legend(loc="right")

ax3 = fig.add_subplot(313)
ax3.set_title('Selection 3')
ax3.set_xlabel("Alpha values")
ax3.set_ylabel("Root mean square")
plt.plot(alphas, RMSs_test3, color='b', label="MAP RMS values (selection 3)")
plt.plot(alphas, [ml_RMS3 for a in alphas], color='r', label="ML RMS (selection 3)")
plt.legend(loc="right")
plt.show()

"""
fig = plt.figure("MAP Selection 1 (test)")
ax = fig.add_subplot(111)
ax.set_xlabel("Years")
ax.set_ylabel("Sunspots")
plt.plot(range(1916, 2012), target_test, color='b', label="Actual test data")
plt.plot(range(1916, 2012), map_guess_test1, color='r', label="Predicted data")
plt.legend(loc="upper left")
plt.show()

fig = plt.figure("MAP Selection 2 (training and test)")
ax = fig.add_subplot(111)
ax.set_xlabel("Years")
ax.set_ylabel("Sunspots")
plt.plot(range(1716, 1916), target_training, color='b', label="Actual training data")
plt.plot(range(1716, 1916), map_guess_training2, color='r', label="Predicted data")
plt.plot(range(1916, 2012), target_test, color='b', label="Actual test data")
plt.plot(range(1916, 2012), map_guess_test2, color='r')
plt.legend(loc="upper left")
plt.show()

fig = plt.figure("MAP Selection 3 (test)")
ax = fig.add_subplot(111)
ax.set_xlabel("Years")
ax.set_ylabel("Sunspots")
plt.plot(range(1916, 2012), target_test, color='b', label="Actual test data")
plt.plot(range(1916, 2012), map_guess_test3, color='r', label="Predicted data")
plt.legend(loc="upper left")
plt.show()
"""

print "RMS for map selection1: " + str(map_regression_test1.root_mean_square())
print "RMS for map selection2: " + str(map_regression_test2.root_mean_square())
print "RMS for map selection3: " + str(map_regression_test3.root_mean_square())