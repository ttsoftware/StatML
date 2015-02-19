from DataPoint import DataPoint
import numpy as np


class DataSet(list):

    def __init__(self, *args, **kwargs):
        """
        :param args: List of DataPoints
        :param kwargs:
        """
        if len(args) > 0:
            for i, x in enumerate(args[0]):
                if not type(x) == DataPoint:
                    raise TypeError('Objects must be of type DataPoint')
            super(DataSet, self).__init__(args[0])

    def unpack_params(self):
        """
        Get the parameters from our dataset
        """
        values = []
        for i, data_point in enumerate(self):
            values += [data_point.params]

        return values

    def unpack_numpy_array(self):
        """
        Get the parameters from our dataset as numpy arrays
        """
        values = []
        for i, data_point in enumerate(self):
            values += [np.array(
                map(lambda x: [x], data_point.params)
            )]

        return values

    def sort(self, cmp=None, key=None, reverse=False):
        super(DataSet, self).sort(cmp=cmp, key=lambda x: x.params[0], reverse=reverse)

    def __iadd__(self, other):
        if not type(other[0]) == DataPoint:
            raise TypeError('Objects must be of type DataPoint')
        return super(DataSet, self).__iadd__(other)