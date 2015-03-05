from __future__ import division
import numpy as np
from DataPoint import DataPoint
from DataSet import DataSet
from LearningDataReader import LearningDataReader


class Normalizer(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.values = dataset.unpack_params()

        # We do this in the constructor in order to save time when we normalize input in the future
        flat_dimensions = []
        for i in range(len(self.values[0])):
            flat_dimensions += [map(lambda x: x[i], self.values)]

        self.dimensions_means = []  # dimension_means for each dimension
        self.dimensions_std = []  # standard deviation for each dimension
        for dim, dim_values in enumerate(flat_dimensions):
            self.dimensions_means += [np.mean(dim_values)]
            self.dimensions_std += [np.std(dim_values)]

    def normalize(self, normalize_function, inputset):
        """
        Return the normalized self.dataset using the given {normalize_function} in each dimension
        :param lambda normalize_function:
        :param DataSet inputset:
        :return DataSet:
        """
        inputset_values = inputset.unpack_params()
        normalized_dataset = DataSet()
        for i, data_point in enumerate(inputset):
            normalized_dataset += [DataPoint(
                params=[
                    map(
                        lambda (dimension, dimension_value):
                            normalize_function(dimension, dimension_value),
                        enumerate(inputset_values[i])
                    )
                ],
                target=data_point.label
            )]

        return normalized_dataset

    def normalize_means(self, inputset):
        """
        Return normalized version of inputset, normalized to each dimension in dimension_means 0 and variance 1 in the self.dataset.
        Since the variance is standard deviation^2, we can simply subtract the dimension dimension_means and divide by dimension standard deviation in each data point in each dimension.
        This normalization asserts data is gaussian distributed.

        :param DataSet inputset:
        :return DataSet:
        """
        return self.normalize(
            lambda d, x: (x - self.dimensions_means[d]) / self.dimensions_std[d],
            inputset
        )