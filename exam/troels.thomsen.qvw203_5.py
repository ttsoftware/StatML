from DataReader import DataReader
from matplotlib import pyplot as plt
import numpy as np
from sklearn import cluster as cl
from DataPoint import DataPoint
from DataSet import DataSet

training_dataset = DataReader.read_data('data/keystrokesTrainTwoClass.csv', ',')
test_dataset = DataReader.read_data('data/keystrokesTestTwoClass.csv', ',')

centroids = DataSet(
    map(
        lambda x: DataPoint(x, target=-1),
        cl.k_means(training_dataset.unpack_params(), 2)[0].tolist()
    )
)

reduced_centroids = training_dataset.principal_component(2, centroids)

centroid_1 = reduced_centroids[0].params
centroid_2 = reduced_centroids[1].params

reduced_training_dataset = training_dataset.principal_component(2)

class1_x = map(lambda x: x.params[0], reduced_training_dataset.get_by_class(0.0))
class1_y = map(lambda x: x.params[1], reduced_training_dataset.get_by_class(0.0))
class2_x = map(lambda x: x.params[0], reduced_training_dataset.get_by_class(1.0))
class2_y = map(lambda x: x.params[1], reduced_training_dataset.get_by_class(1.0))

fig = plt.figure('Principal component analysis', dpi=160)
ax = fig.add_subplot(111)

fig.tight_layout()

ax.set_title('Scatter plot')
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')

plt.plot(class1_x, class1_y, '.', color=[0, 0, 0], label="Class 1")
plt.plot(class2_x, class2_y, '.', color='red', label="Class 2")

plt.plot(centroid_1[0], centroid_1[1], 'D', color='#000AFF', label="Centroid 1: (" + str(centroid_1[0]) + ", " + str(centroid_1[1]) + ")")
plt.plot(centroid_2[0], centroid_2[1], 'D', color='#FF00F7', label="Centroid 2: (" + str(centroid_2[0]) + ", " + str(centroid_2[1]) + ")")

plt.legend(loc="upper left")

plt.show()