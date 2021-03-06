from Classifier import Classifier
from Gauss import Gauss
import numpy as np
import matplotlib.pyplot as plt
from LearningDataReader import LearningDataReader
from Normalizer import Normalizer

# I.2

Gauss.draw_gauss_one(-1, 1)
Gauss.draw_gauss_one(0, 2)
Gauss.draw_gauss_one(2, 3)
plt.show()

mean = np.array([[1], [2]])
covariance = np.array([[0.3, 0.2], [0.2, 0.2]])

Gauss.draw_gauss_multi(mean, covariance, label='Sample')
plt.show()

Gauss.draw_likelihood(mean, covariance)
plt.show()

Gauss.draw_eigenvectors(mean, covariance)
plt.show()

Gauss.draw_rotated_covariance(mean, covariance)
plt.show()

# I.3.1

trainingset = LearningDataReader.read_iris('IrisTrain2014.dt')

classifier = Classifier(trainingset)

testset = LearningDataReader.read_iris('IrisTest2014.dt')

accuracy_test1 = classifier.find_accuracy(testset, 1)
accuracy_test3 = classifier.find_accuracy(testset, 3)
accuracy_test5 = classifier.find_accuracy(testset, 5)

print "Test accuracy k=1: " + str(accuracy_test1)
print "Test accuracy k=3: " + str(accuracy_test3)
print "Test accuracy k=5: " + str(accuracy_test5)

accuracy_train1 = classifier.find_accuracy(trainingset, 1)
accuracy_train3 = classifier.find_accuracy(trainingset, 3)
accuracy_train5 = classifier.find_accuracy(trainingset, 5)

print "Train accuracy k=1: " + str(accuracy_train1)
print "Train accuracy k=3: " + str(accuracy_train3)
print "Train accuracy k=5: " + str(accuracy_train5)

# I.3.2

classifier = Classifier(trainingset)
best_k = classifier.cross_validator(5)
raw_accuracy_test = classifier.find_accuracy(testset, best_k)
raw_accuracy_training = classifier.find_accuracy(trainingset, best_k)

# I.3.3

normalizer = Normalizer(trainingset)
normalized_trainingset = normalizer.normalize_means(trainingset)
normalized_testset = normalizer.normalize_means(testset)

training_params = trainingset.unpack_numpy_array()
normalized_training_params = normalized_trainingset.unpack_numpy_array()

testset_params = testset.unpack_numpy_array()
normalized_testset_params = normalized_testset.unpack_numpy_array()

fig = Gauss.draw_gauss_multi(sample=training_params, label='Training set')
Gauss.draw_gauss_multi(sample=normalized_training_params, label='Training set transform', fig=fig)
plt.show()

Gauss.draw_gauss_multi(sample=testset_params, label='Test set', fig=fig)
Gauss.draw_gauss_multi(sample=normalized_testset_params, label='Test set transform', fig=fig)
plt.show()

classifier = Classifier(normalized_trainingset)
best_k = classifier.cross_validator(5)
accuracy_normalized_test = classifier.find_accuracy(normalized_testset, best_k)
accuracy_normalized_training = classifier.find_accuracy(normalized_trainingset, best_k)

# Result from I.3.2
print "Raw accuracy test: " + str(raw_accuracy_test)
print "Raw accuracy training: " + str(raw_accuracy_training)
# Result from I.3.3
print "Normalized accuracy test: " + str(accuracy_normalized_test)
print "Normalized accuracy training: " + str(accuracy_normalized_training)