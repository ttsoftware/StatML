import numpy as np
import scipy.spatial.distance as spatial


class Classifier(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def nearest_neighbour(self, k, coordinate):

        neighbours = []

        # we expect dataset to be sorted, and we expect the first parameter to be the x-axis value.
        for i, data in enumerate(self.dataset):
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