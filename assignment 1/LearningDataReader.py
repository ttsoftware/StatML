
class LearningDataReader(object):

    @staticmethod
    def read_iris(filename):

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