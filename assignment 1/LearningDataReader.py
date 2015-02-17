import numpy as np

class LearningDataReader(object):

    @staticmethod
    def read_iris(filename):
        """
        Read our iris data, and return it in our specified data format.
        :param filename:
        :return:
        """
        dataset = []
        with open(filename) as f:
            for line in f:
                c1, c2, c3 = line.split(' ')
                dataset += [{
                    'params': [float(c1), float(c2)],
                    'label': int(c3)
                }]

        # we sort by x-axis so we can more easily discover nearest neighbours
        return sorted(dataset, key=lambda x: x['params'][0])

    @staticmethod
    def unpack_params(dataset):
        """
        Get the parameters from our dataset
        :param dataset:
        """
        values = []
        for i, data_point in enumerate(dataset):
            values += [data_point['params']]

        return values

    @staticmethod
    def unpack_numpy_array(dataset):
        """
        Get the parameters from our dataset as numpy arrays
        :param dataset:
        """
        values = []
        for i, data_point in enumerate(dataset):
            values += [np.array(
                map(lambda x: [x], data_point['params'])
            )]

        return values