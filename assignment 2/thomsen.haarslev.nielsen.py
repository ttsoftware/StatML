from DataReader import DataReader
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

normalized_training_classified = normalized_lda.classify_dataset(training_dataset)
normalized_test_classified = normalized_lda.classify_dataset(test_dataset)

normalized_training_accuracy = LDAClassifier.find_accuracy(normalized_training_dataset, normalized_training_classified)
normalized_test_accuracy = LDAClassifier.find_accuracy(normalized_test_dataset, normalized_test_classified)

print 'Normalized training set accuracy: ' + str(normalized_training_accuracy)
print 'Normalized test set accuracy: ' + str(normalized_test_accuracy)


############################# Sunspot prediction ###############################

sunspot_training_dataset = DataReader.read_data("sunspotsTrainStatML.dt")
sunspot_test_dataset 	 = DataReader.read_data("sunspotsTestStatML.dt")

selection1_training = map(lambda x: [x.params[2], x.params[3]], sunspot_training_dataset)
selection2_training = map(lambda x: [x.params[4]], sunspot_training_dataset)
selection3_training = map(lambda x: x.params, sunspot_training_dataset)

selection1_test = map(lambda x: [x.params[2], x.params[3]], sunspot_test_dataset)
selection2_test = map(lambda x: [x.params[4]], sunspot_test_dataset)
selection3_test = map(lambda x: x.params, sunspot_test_dataset)

regression_training = Regression(sunspot_training_dataset, 1716)

w1_training = Regression.regression(map(lambda x: [1, x], range(1716, 1916)), selection1_training, )
w2_training = Regression.regression(map(lambda x: [1, x], range(1716, 1916)), selection2_training)
w3_training = Regression.regression(map(lambda x: [1, x], range(1716, 1916)), selection3_training)

reguessions_training = [Regression.reguession(w2_training, x) for x in range(1716, 1916)]
reguessions1_test = [Regression.reguession(w1_training, x) for x in range(1916, 2012)]
reguessions2_test = [Regression.reguession(w2_training, x) for x in range(1916, 2012)]
reguessions3_test = [Regression.reguession(w3_training, x) for x in range(1916, 2012)]

plt.figure("Regression")
plt.plot(range(1716, 1916), map(lambda x: sum(x), selection2_training), '.b')
plt.plot(range(1716, 1916), reguessions_training, 'r')
plt.show()

plt.figure("Regression")
plt.plot(range(1916, 2012), map(lambda x: sum(x), selection1_test), '.b')
plt.plot(range(1916, 2012), reguessions1_test, 'r')
plt.show()

plt.figure("Regression")
plt.plot(range(1916, 2012), map(lambda x: sum(x), selection2_test), '.b')
plt.plot(range(1916, 2012), reguessions2_test, 'r')
plt.show()

plt.figure("Regression")
plt.plot(range(1916, 2012), map(lambda x: sum(x), selection3_test), '.b')
plt.plot(range(1916, 2012), reguessions3_test, 'r')
plt.show()