from DataReader import DataReader
from matplotlib import pyplot as plt
import numpy as np

training_dataset = DataReader.read_data('data/keystrokesTrainTwoClass.csv', ',')
test_dataset = DataReader.read_data('data/keystrokesTestTwoClass.csv', ',')

training_numpy_array = np.array(training_dataset.unpack_params()).T
covariance = np.cov(training_numpy_array)

# eigenvectors and eigenvalues for the from the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance)

transformed_training_dataset = training_dataset.principal_component(2)

class1_x = map(lambda x: x.params[0], transformed_training_dataset.get_by_class(0.0))
class1_y = map(lambda x: x.params[1], transformed_training_dataset.get_by_class(0.0))
class2_x = map(lambda x: x.params[0], transformed_training_dataset.get_by_class(1.0))
class2_y = map(lambda x: x.params[1], transformed_training_dataset.get_by_class(1.0))

fig = plt.figure('Principal component analysis', dpi=150)
ax = fig.add_subplot(211)

fig.tight_layout()

ax.set_title('Eigenspectrum')
ax.set_xlabel('Attribute')
ax.set_ylabel('Eigenvalues')

plt.plot(range(21), sorted(eigenvalues.tolist(), reverse=True), color='b', label="Eigenspectrum")
plt.legend(loc="upper left")

ax1 = fig.add_subplot(212)
ax1.set_title('Scatter plot')
ax1.set_xlabel('Principal component 1')
ax1.set_ylabel('Principal component 2')

plt.plot(class1_x, class1_y, '.', color=[0, 0, 0], label="Class 1")
plt.plot(class2_x, class2_y, '.', color='red', label="Class 2")
plt.legend(loc="upper left")

plt.show()