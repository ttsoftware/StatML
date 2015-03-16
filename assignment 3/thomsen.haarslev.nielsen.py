from __future__ import division
from DataReader import DataReader
from NeuralNetwork import NeuralNetwork
from Normalizer import Normalizer
from SupportVectorMachine import SupportVectorMachine

############### 1.1 ################

training_dataset = DataReader.read_data('data/sincTrain25.dt')

network = NeuralNetwork(1, 1, 1)
network.skub(training_dataset)

exit()

############## 2.1 ################

# Normalization
parkinson_training = DataReader.read_data("data/parkinsonsTrainStatML.dt")
parkinson_test = DataReader.read_data("data/parkinsonsTestStatML.dt")

normalizer = Normalizer(parkinson_training)

print "Training mean:", normalizer.dimensions_means
print "Training variance:", normalizer.variance()

parkinson_train_normalized = normalizer.normalize_means(parkinson_training)
parkinson_test_normalized = normalizer.normalize_means(parkinson_test)

test_normalizer = Normalizer(parkinson_test_normalized)
train_normalizer = Normalizer(parkinson_train_normalized)
print "Normalized test mean:", test_normalizer.dimensions_means
print "Normalized test variance:", test_normalizer.variance()

# SVM

SVM = SupportVectorMachine(parkinson_training)
SVM_normalized = SupportVectorMachine(parkinson_train_normalized)

appr_params = SVM.get_params()

gammas = [appr_params['gamma'] * 10**x for x in range(-3, 4)]
Cs = [appr_params['C'] * 10**x for x in range(-3, 4)]


best_gamma, best_C, best_loss = SVM.cross_validator(gammas, Cs)
best_gamma_normalized, best_C_normalized, best_loss_normalized = SVM_normalized.cross_validator(gammas, Cs)

# print best_gamma, best_C, best_loss
# print best_gamma_normalized, best_C_normalized, best_loss_normalized

SVM_best = SupportVectorMachine(parkinson_training, gamma=best_gamma, C=best_C)
SVM_normalized_best = SupportVectorMachine(parkinson_train_normalized, gamma=best_gamma, C=best_C)

print "Loss for raw training set:", SVM_best.loss(parkinson_training)
print "Loss for raw test set:", SVM_best.loss(parkinson_test)

print "Loss for normalized training set:", SVM_normalized_best.loss(parkinson_train_normalized)
print "Loss for normalized test set:", SVM_normalized_best.loss(parkinson_test_normalized)


bounded = 0
free = 0
for coef in SVM_best.clf.dual_coef_[0]:
    if abs(coef) == best_C:
        bounded += 1
    else:
        free += 1

print "Bounded SV:", bounded
print "Free SV:", free
