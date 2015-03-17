from __future__ import division
from sklearn import svm


class SupportVectorMachine(object):

    def __init__(self, dataset, gamma=0.1, C=10, kernel='rbf'):
        self.dataset = dataset
        self.features = dataset.unpack_params()
        self.targets = dataset.unpack_targets()
        self.gamma = gamma
        self.C = C
        self.kernel = kernel

        self.clf = svm.SVC(kernel=kernel, gamma=gamma, C=C)
        self.clf.fit(self.features, self.targets)

    def predict(self, inputset):
        """
        Predicts the label for all features in the inputset
        """
        return self.clf.predict(inputset)

    def loss(self, inputset):
        features = inputset.unpack_params()
        targets = inputset.unpack_targets()
        prediction = self.predict(features)

        losses = []
        for i in range(len(prediction)):
            losses.append(prediction[i] == targets[i])
        return losses.count(False) / len(losses)

    def get_params(self):
        params = self.clf.get_params()
        return {'C': params['C'], 'gamma': params['gamma']}

    def cross_validator(self, gammas, Cs, s_fold=5):

        s_partitions = int(len(self.dataset)/s_fold)

        test_partitions = []
        test_targets = []
        train_partitions = []
        train_targets = []

        for k in xrange(s_fold):
            start_current = k * s_partitions
            end_current = (k + 1) * s_partitions

            test_partitions += [map(lambda x: x.get_vector(), self.dataset[start_current:end_current])]
            test_targets += [map(lambda x: x.target, self.dataset[start_current:end_current])]
            train_partitions += [map(lambda x: x.get_vector(), (self.dataset[:start_current] + self.dataset[end_current:]))]
            train_targets += [map(lambda x: x.target, (self.dataset[:start_current] + self.dataset[end_current:]))]

        best_set = (-1, -1, 2)
        for j in range(len(Cs)):
            for i in range(len(gammas)):
                loss = []

                for h in xrange(len(train_partitions)):

                    clf = svm.SVC(kernel='rbf', gamma=gammas[i], C=Cs[j])
                    clf.fit(train_partitions[h], train_targets[h])

                    predictions = clf.predict(test_partitions[h])
                    for k in range(len(predictions)):
                        loss.append(predictions[k] == test_targets[h][k])

                avg_loss = loss.count(False) / len(loss)
                if avg_loss < best_set[2]:
                    best_set = (gammas[i], Cs[j], avg_loss)

        # Return the amount of neighbors that yields the best loss
        return best_set