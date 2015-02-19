from __future__ import division
import numpy as np
from LearningDataReader import LearningDataReader


class Normalizer(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.values = LearningDataReader.unpack_params(self.dataset)

        # We do this in the constructor in order to save time when we normalize input in the future
        flat_dimensions = []
        for i in range(len(self.values[0])):
            flat_dimensions += [map(lambda x: x[i], self.values)]

        self.dimensions_means = []  # mean for each dimension
        self.dimensions_std = []  # standard deviation for each dimension
        for dim, dim_values in enumerate(flat_dimensions):
            self.dimensions_means += [np.mean(dim_values)]
            self.dimensions_std += [np.std(dim_values)]

    def normalize(self, normalize_function, inputset):
        """
        Return the normalized self.dataset using the given {normalize_function} in each dimension
        :param normalize_function:
        :param input: must be of type {DataPoint}
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

    def normalize_means(self, testset):
        """
        Return normalized version of testset, normalized to each dimension in mean 0 and variance 1.
        Since the variance is standard deviation^2, we can simply subtract the dimension mean and divide by dimension standard deviation in each data point in each dimension.
        This normalization asserts data is gaussian distributed.
        """

        return self.normalize(lambda d, x: (x - self.dimensions_means[d]) / self.dimensions_std[d])