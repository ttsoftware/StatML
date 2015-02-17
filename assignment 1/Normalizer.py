from __future__ import division
import numpy as np
from LearningDataReader import LearningDataReader


class Normalizer(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.values = LearningDataReader.unpack_params(self.dataset)

    def normalize(self, normalize_function):
        """
        Return the normalized self.dataset using the given {normalize_function} in each dimension
        :param normalize_function:
        :return:
        """
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
        Return normalized version of self.dataset, normalized to each dimension in mean 0 and variance 1.
        Since the variance is standard deviation^2, we can simply subtract the dimension mean and divide by dimension standard deviation in each data point in each dimension.
        This normalization asserts data is gaussian distributed.
        """
        flat_dimensions = []
        for i in range(len(self.values[0])):
            flat_dimensions += [map(lambda x: x[i], self.values)]

        dimensions_means = []  # mean for each dimension
        dimensions_std = []  # standard deviation for each dimension
        for dim, dim_values in enumerate(flat_dimensions):
            dimensions_means += [np.mean(dim_values)]
            dimensions_std += [np.std(dim_values)]

        return self.normalize(lambda d, x: (x - dimensions_means[d]) / dimensions_std[d])