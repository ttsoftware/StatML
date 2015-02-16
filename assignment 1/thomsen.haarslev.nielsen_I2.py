from Gauss import Gauss
import numpy as np
import matplotlib.pyplot as plt

#Gauss.draw_gauss_one(-1, 1)
#Gauss.draw_gauss_one(0, 2)
#Gauss.draw_gauss_one(2, 3)\

mean = np.array([[1], [2]])
covariance = np.array([[0.3, 0.2], [0.2, 0.2]])

#fig = Gauss.draw_gauss_multi(mean, covariance)
#plt.show()
#Gauss.draw_likelihood(mean, covariance)
#Gauss.draw_eigenvectors(mean, covariance)
#Gauss.draw_eigenvectors(mean, covariance)
Gauss.draw_rotated_covariance(mean, covariance)
plt.show()