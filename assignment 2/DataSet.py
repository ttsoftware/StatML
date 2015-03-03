from DataPoint import DataPoint
import numpy as np


class DataSet(list):
    def __init__(self, *args, **kwargs):
        """
        :param args: List of DataPoints
        :param kwargs:
        """

        self.class_sets = {}

        if len(args) > 0:
            for i, x in enumerate(args[0]):
                if not type(x) == DataPoint:
                    raise TypeError('Objects must be of type DataPoint')
                self.add_class_point(x)
            super(DataSet, self).__init__(args[0])

    def add_class_point(self, data_point):
        """
        Add datapoint to set of classes
        :param data_point:
        """
        if data_point.target not in self.class_sets.keys():
            self.class_sets[data_point.target] = DataSet()
        self.class_sets[data_point.target].__iadd__([data_point], True)

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
            values += [data_point.get_vector()]
        return values

    def get_by_class(self, class_name):
        """
        Returns all DataPoints which belongs to given class, as numpy arrays
        :param class_name:
        :return:
        """
        return self.class_sets[class_name]

    def sort(self, cmp=None, key=None, reverse=False):
        super(DataSet, self).sort(cmp=cmp, key=lambda x: x.params[0], reverse=reverse)

    def __iadd__(self, other, avoid_recursion=False):
        if not type(other[0]) == DataPoint:
            raise TypeError('Objects must be of type DataPoint')

        if not avoid_recursion:
            for i, data_point in enumerate(other):
                self.add_class_point(data_point)

        return super(DataSet, self).__iadd__(other)