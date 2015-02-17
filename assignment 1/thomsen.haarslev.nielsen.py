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

accuracy_1 = classifier.find_accuracy(testset, 1)
accuracy_3 = classifier.find_accuracy(testset, 3)
accuracy_5 = classifier.find_accuracy(testset, 5)

print "Accuracy k=1: " + str(accuracy_1)
print "Accuracy k=3: " + str(accuracy_3)
print "Accuracy k=5: " + str(accuracy_5)

# I.3.2

classifier = Classifier(trainingset)
best_k = classifier.cross_validator(5)
raw_accuracy = classifier.find_accuracy(testset, best_k)

# I.3.3

normalizer = Normalizer(trainingset)
normalized_trainingset = normalizer.normalize_means()

normalizer = Normalizer(testset)
normalized_testset = normalizer.normalize_means()

training_params = LearningDataReader.unpack_numpy_array(trainingset)
normalized_training_params = LearningDataReader.unpack_numpy_array(normalized_trainingset)

testset_params = LearningDataReader.unpack_numpy_array(testset)
normalized_testset_params = LearningDataReader.unpack_numpy_array(normalized_testset)

fig = Gauss.draw_gauss_multi(sample=training_params, label='Training set')
Gauss.draw_gauss_multi(sample=normalized_training_params, label='Training set transform', fig=fig)
plt.show()

Gauss.draw_gauss_multi(sample=testset_params, label='Test set', fig=fig)
Gauss.draw_gauss_multi(sample=normalized_testset_params, label='Test set transform', fig=fig)
plt.show()

classifier = Classifier(normalized_trainingset)
best_k = classifier.cross_validator(5)
accuracy_normalized = classifier.find_accuracy(normalized_testset, best_k)

# Result from I.3.2
print "Raw accuracy: " + str(raw_accuracy)
# Result from I.3.3
print "Normalized accuracy: " + str(accuracy_normalized)