from DataReader import DataReader
from Gauss import Gauss
from LDAClassifier import LDAClassifier
from Normalizer import Normalizer
from Regression import Regression
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
sunspot_test_dataset     = DataReader.read_data("sunspotsTestStatML.dt")

    # Training
selection1_training = map(lambda x: [x.params[2], x.params[3]], sunspot_training_dataset)
selection2_training = map(lambda x: [x.params[4]], sunspot_training_dataset)
selection3_training = map(lambda x: x.params, sunspot_training_dataset)
target_training     = map(lambda x: [x.target], sunspot_training_dataset)

regression_train1 = Regression(selection1_training, target_training)
regression_train2 = Regression(selection2_training, target_training)
regression_train3 = Regression(selection3_training, target_training)

reguessions_training1 = [regression_train1.reguession(x) for x in range(200)]
reguessions_training2 = [regression_train2.reguession(x) for x in range(200)]
reguessions_training3 = [regression_train3.reguession(x) for x in range(200)]

    # Test
selection1_test = map(lambda x: [x.params[2], x.params[3]], sunspot_test_dataset)
selection2_test = map(lambda x: [x.params[4]], sunspot_test_dataset)
selection3_test = map(lambda x: x.params, sunspot_test_dataset)
target_test     = map(lambda x: [x.target], sunspot_test_dataset)

regression_test1 = Regression(selection1_test, target_test)
regression_test2 = Regression(selection2_test, target_test)
regression_test3 = Regression(selection3_test, target_test)

reguessions_test1 = [regression_test1.reguession(x) for x in range(96)]
reguessions_test2 = [regression_test2.reguession(x) for x in range(96)]
reguessions_test3 = [regression_test3.reguession(x) for x in range(96)]

plt.figure("Selection 1 (test)")
plt.plot(range(1916, 2012), map(lambda x: sum(x), selection1_test), color='g', label="Actual test data")
plt.plot(range(1916, 2012), reguessions_test1, color='r', label="Predicted data")
plt.legend(loc="upper left")
plt.show()

plt.figure("Selection 2 (training and test)")
plt.plot(range(1716, 1916), map(lambda x: sum(x), selection2_training), color='b', label="Actual training data")
plt.plot(range(1716, 1916), reguessions_training2, color='r', label="Predicted data")
plt.plot(range(1916, 2012), map(lambda x: sum(x), selection2_test), color='g', label="Actual test data")
plt.plot(range(1916, 2012), reguessions_test2, color='r')
plt.legend(loc="upper left")
plt.show()

plt.figure("Selection 3 (test)")
plt.plot(range(1916, 2012), map(lambda x: sum(x), selection3_test), color='g', label="Actual test data")
plt.plot(range(1916, 2012), reguessions_test3, color='r', label="Predicted data")
plt.legend(loc="upper left")
plt.show()

