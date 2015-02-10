from __future__ import division
from Classifier import Classifier
from LearningDataReader import LearningDataReader

trainingset = LearningDataReader.read_iris('IrisTrain2014.dt')

classifier = Classifier(trainingset)

testset = LearningDataReader.read_iris('IrisTest2014.dt')

accuracy_1 = []
for i, data in enumerate(testset):
    label = classifier.nearest_neighbour(1, data['params'])
    accuracy_1 += [label == data['label']]

accuracy_3 = []
for i, data in enumerate(testset):
    label = classifier.nearest_neighbour(3, data['params'])
    accuracy_3 += [label == data['label']]
 
accuracy_5 = []
for i, data in enumerate(testset):
    label = classifier.nearest_neighbour(5, data['params'])
    accuracy_5 += [label == data['label']]

print accuracy_1.count(True) / len(accuracy_1)
print accuracy_3.count(True) / len(accuracy_3)
print accuracy_5.count(True) / len(accuracy_5)