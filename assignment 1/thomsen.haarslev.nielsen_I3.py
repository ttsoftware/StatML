from __future__ import division
import matplotlib.pyplot as plt
from Classifier import Classifier
from Gauss import Gauss
from LearningDataReader import LearningDataReader
from Normalizer import Normalizer

trainingset = LearningDataReader.read_iris('IrisTrain2014.dt')

classifier = Classifier(trainingset)

testset = LearningDataReader.read_iris('IrisTest2014.dt')

accuracy_1 = classifier.find_accuracy(testset, 1)
accuracy_3 = classifier.find_accuracy(testset, 3)
accuracy_5 = classifier.find_accuracy(testset, 5)

#print accuracy_1
#print accuracy_3
#print accuracy_5

classifier = Classifier(trainingset)
best_k = classifier.cross_validator(5)
accuracy_raw = classifier.find_accuracy(testset, best_k)

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
#Gauss.draw_gauss_multi(sample=testset_params, label='Test set', fig=fig)
#Gauss.draw_gauss_multi(sample=normalized_testset_params, label='Test set transform', fig=fig)
plt.show()

classifier = Classifier(normalized_trainingset)
best_k = classifier.cross_validator(5)
accuracy_normalized = classifier.find_accuracy(normalized_testset, best_k)

print accuracy_raw, accuracy_normalized