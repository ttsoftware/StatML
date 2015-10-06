from __future__ import division


class LDA(object):

    def __init__(self, dataset):
        self.dataset = dataset

        self.parameter_class_count_histogram = {}
        self.parameter_count = {}
        self.class_count = {}

        self.parameter_class_propability = {}
        self.parameter_propability = {}
        self.class_propability = {}

        self.calc_propabilities()

    def calc_propabilities(self):
        """
        Calculate the class conditional densities for p(X = x|Y = C_k), p(X = x) and p(Y = C_k)
        """

        # count number of types of given parameters in each class
        for i, point in enumerate(self.dataset):
            if point.label not in self.parameter_class_count_histogram.keys():
                self.parameter_class_count_histogram[point.label] = {}

            for j, param in enumerate(point.params):
                if param in self.parameter_count.keys():
                    self.parameter_count[param] += 1
                else:
                    self.parameter_count[param] = 1

                if param in self.parameter_class_count_histogram[point.label].keys():
                    self.parameter_class_count_histogram[point.label][param] += 1
                else:
                    self.parameter_class_count_histogram[point.label][param] = 1

        for param, count in self.parameter_count.iteritems():
            # calculate the propability of a given parameter occuring in any class
            self.parameter_propability[param] = count / sum(self.parameter_count.values())

        for class_name, params in self.parameter_class_count_histogram.iteritems():
            # calculate the propability of a given parameter being in this class
            self.parameter_class_propability[class_name] = map(lambda x: x / len(params), params.values())

            # calculate the propability of any parameter being in this class
            self.class_count[class_name] = sum(params.values())
            self.class_propability[class_name] = self.class_count[class_name] / len(self.dataset)

    def class_mean(self, class_name):
        (1/self.class_count[class_name]) * sum(self.parameter_count.values())

    def find_covariance(self):
        pass