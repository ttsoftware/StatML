from __future__ import division
import numpy as np
import scipy.spatial.distance as spatial


class Classifier(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def nearest_neighbour(self, k, coordinate, train_data=None):

        if train_data is None:
            train_data = self.dataset

        neighbours = []

        # we expect dataset to be sorted, and we expect the first parameter to be the x-axis value.
        for i, data in enumerate(train_data):
            neighbours += [
                (data,
                 spatial.euclidean(coordinate, data['params']))
            ]

        # get labels for k-nearest neighbours
        labels = map(
            lambda x: x[0]['label'],
            sorted(neighbours, key=lambda x: x[1])[0:k]
        )

        # return label with most occurences
        return reduce(lambda x, y: x if labels.count(x) > y else y, labels)

    def cross_validator(self, s_fold=5, max_k=25):
        s_partitions = int(len(self.dataset)/s_fold)

        test_partitions = []
        train_partitions = []

        for i in xrange(s_fold):
            start_current = i * s_partitions
            end_current = (i + 1) * s_partitions

            test_partitions += [self.dataset[start_current:end_current]]
            train_partitions += [self.dataset[:start_current] + self.dataset[end_current:]]

        best_k = (-1, -1)
        for i in range(1, max_k + 1, 2):
            accuracy = []

            for h in xrange(0, len(train_partitions)):
                for j, data in enumerate(test_partitions[h]):
                    label = self.nearest_neighbour(i, data['params'], train_partitions[h])
                    accuracy += [label == data['label']]

            if accuracy.count(True) / len(accuracy) > best_k[1]:
                best_k = (i, accuracy.count(True) / len(accuracy))

        # Return the amount of neighbors that yields the best accuracy
        return best_k[0]

    def find_accuracy(self, testset, k):
        """
        Find the accuracy if the self.dataset on the given testset, for k = k
        :param testset:
        :param k:
        :return:
        """
        accuracy = []
        for i, data in enumerate(testset):
            label = self.nearest_neighbour(k, data['params'])
            accuracy += [label == data['label']]

        return accuracy.count(True) / len(accuracy)
