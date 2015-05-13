from DataReader import DataReader
from MLRegression import MLRegression
from matplotlib import rc
from matplotlib import pyplot as plt

from MAPRegression import MAPRegression
from Normalizer import Normalizer
from SupportVectorRegression import SupportVectorRegression
from DataSet import DataSet
import numpy as np

training_dataset = DataReader.read_data('data/redshiftTrain.csv')
test_dataset = DataReader.read_data('data/redshiftTest.csv')

#normalizer = Normalizer(training_dataset)

#normalized_training_dataset = normalizer.normalize_means(training_dataset)
#normalized_test_dataset = normalizer.normalize_means(test_dataset)

ml_regression = MLRegression(training_dataset)
map_regression = MAPRegression(0, 1, training_dataset)

ml_train_MSE = ml_regression.mean_square()
ml_test_MSE = ml_regression.mean_square(test_dataset)
map_train_MSE = map_regression.mean_square()
map_test_MSE = map_regression.mean_square(test_dataset)

print "\nML Regressions MSE training and test:"
print ml_train_MSE
print ml_test_MSE

print "\nMAP Regressions MSE training and test:"
print map_train_MSE
print map_test_MSE

# SVR

gammas = [0.1*10**i for i in range(-3, 4)]
cs = [10*10**i for i in range(-3, 4)]

"""
best_gamma, best_c, avg_MSE = SupportVectorRegression.cross_validator(
    training_dataset,
    gammas,
    cs,
    kernel='rbf'
)
"""

# we found the following values from cross validation
best_gamma = 0.01
best_c = 10
avg_MSE = 0.0520741985314

print "\nCross validation results for rbf:"
print "best gamma: " + str(best_gamma)
print "best C: " + str(best_c)
print "avg MSE: " + str(avg_MSE)

svr_rbf = SupportVectorRegression(training_dataset, gamma=best_gamma, C=best_c)
svr_rbf_train_MSE = svr_rbf.mean_square(training_dataset)
svr_rbf_test_MSE = svr_rbf.mean_square(test_dataset)

print "\nSVR rbf MSE training and test:"
print svr_rbf_train_MSE
print svr_rbf_test_MSE

svr_poly = SupportVectorRegression(training_dataset, gamma=best_gamma, C=best_c, kernel='poly')
svr_poly_train_MSE = svr_poly.mean_square(training_dataset)
svr_poly_test_MSE = svr_poly.mean_square(test_dataset)

print "\nSVR poly MSE training and test:"
print svr_poly_train_MSE
print svr_poly_test_MSE

"""
best_gamma, best_c, avg_MSE = SupportVectorRegression.cross_validator(
    training_dataset,
    gammas,
    cs,
    kernel='linear'
)
"""

best_gamma = 0.0001
best_c = 100
avg_MSE = 0.0554482311156

print "\nCross validation results for linear:"
print "best gamma: " + str(best_gamma)
print "best C: " + str(best_c)
print "avg MSE: " + str(avg_MSE)

svr_linear = SupportVectorRegression(training_dataset, gamma=best_gamma, C=best_c, kernel='linear')
svr_lin_train_MSE = svr_linear.mean_square(training_dataset)
svr_lin_test_MSE = svr_linear.mean_square(test_dataset)

print "\nSVR linear MSE training and test:"
print svr_lin_train_MSE
print svr_lin_test_MSE

# plot

ml_guess_test = [ml_regression.predict([1] + test_dataset[x].params) for x in range(100)]
map_guess_test = [map_regression.predict([1] + test_dataset[x].params) for x in range(100)]

svr_rbf_guess_test = svr_rbf.predict(DataSet(test_dataset[:100]))
svr_poly_guess_test = svr_poly.predict(DataSet(test_dataset[:100]))
svr_lin_guess_test = svr_linear.predict(DataSet(test_dataset[:100]))

fig = plt.figure('Linear Model', dpi=120)

"""
ax = fig.add_subplot(511)
ax.set_title('ML')
ax.set_xlabel('Line in test data file')
ax.set_ylabel('Redshift')

#plt.plot(ml_guess_training, color='b', label="Training fit")
plt.plot(test_dataset.unpack_targets()[0:100], color='b', label="Test actual")
plt.plot(ml_guess_test, color='r', label="Test guess with MSE: " + str(ml_test_MSE))
plt.legend(loc="upper left")

ax1 = fig.add_subplot(512)
ax1.set_title('MAP')
ax1.set_xlabel('Line in test data file')
ax1.set_ylabel('Redshift')
plt.plot(test_dataset.unpack_targets()[:100], color='b', label="Test actual")
plt.plot(map_guess_test, color='r', label="Test guess with MSE: " + str(map_test_MSE))
plt.legend(loc="upper left")

ax4 = fig.add_subplot(513)
ax4.set_title('SVR linear')
ax4.set_xlabel('Line in test data file')
ax4.set_ylabel('Redshift')
plt.plot(test_dataset.unpack_targets()[:100], color='b', label="Test actual")
plt.plot(svr_lin_guess_test, color='r', label="Test guess with MSE: " + str(svr_lin_test_MSE))
plt.legend(loc="upper left")
"""

ax2 = fig.add_subplot(211)
ax2.set_title('SVR rbf')
ax2.set_xlabel('Line in test data file')
ax2.set_ylabel('Redshift')
plt.plot(test_dataset.unpack_targets()[:100], color='b', label="Test actual")
plt.plot(svr_rbf_guess_test, color='r', label="Test guess with MSE: " + str(svr_rbf_test_MSE))
plt.legend(loc="upper left")

ax3 = fig.add_subplot(212)
ax3.set_title('SVR poly')
ax3.set_xlabel('Line in test data file')
ax3.set_ylabel('Redshift')
plt.plot(test_dataset.unpack_targets()[:100], color='b', label="Test actual")
plt.plot(svr_poly_guess_test, color='r', label="Test guess with MSE: " + str(svr_poly_test_MSE))
plt.legend(loc="upper left")

fig.tight_layout()

#plt.show()