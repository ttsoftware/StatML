import numpy as np
from DataPoint import DataPoint
from DataSet import DataSet


class DataReader(object):

    @staticmethod
    def read_iris(filename):
        """
        Read our iris data, and return it in our specified data format.
        :param String filename:
        :return DataSet:
        """
        dataset = DataSet()
        with open(filename) as f:
            for line in f:
                c1, c2, c3 = line.split(' ')
                dataset += [DataPoint([float(c1), float(c2)], int(c3))]

        # we sort by x-axis so we can more easily discover nearest neighbours
        dataset.sort()
        return dataset