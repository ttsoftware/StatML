from __future__ import division
import numpy as np
from LearningDataReader import LearningDataReader


class Normalizer(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.values = LearningDataReader.unpack_params(self.dataset)

    def normalize(self, normalize_function):
        normalized_dataset = []
        for i, data_point in enumerate(self.dataset):
            normalized_dataset += [{
                'params': [
                    map(
                        lambda (dimension, dimension_value): normalize_function(dimension, dimension_value),
                        enumerate(self.values[i])
                    )
                ],
                'label': data_point['label']
            }]

        return normalized_dataset

    def normalize_means(self):
        """
        This normalization asserts data is already gaussian distributed
        """
        flat_dimensions = []
        for i in range(len(self.values[0])):
            flat_dimensions += [map(lambda x: x[i], self.values)]

        dimensions_means = []
        dimensions_std = []
        for dim, dim_values in enumerate(flat_dimensions):
            dimensions_means += [np.mean(dim_values)]
            dimensions_std += [np.std(dim_values)]

        return self.normalize(lambda d, x: (x - dimensions_means[d]) / dimensions_std[d])