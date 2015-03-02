from DataReader import DataReader
from Gauss import Gauss
from LDAClassifier import LDAClassifier
from Normalizer import Normalizer
import matplotlib.pyplot as plt
import numpy as np

training_dataset = DataReader.read_data('IrisTrain2014.dt')
test_dataset = DataReader.read_data('IrisTest2014.dt')

lda = LDAClassifier(training_dataset)

training_classified = lda.classify_dataset(training_dataset)
test_classified = lda.classify_dataset(test_dataset)

training_accuracy = LDAClassifier.find_accuracy(training_dataset, training_classified)
test_accuracy = LDAClassifier.find_accuracy(test_dataset, test_classified)

print 'Standard training set accuracy: ' + str(training_accuracy)
print 'Standard test set accuracy: ' + str(test_accuracy)

############################# Normalized ###############################

normalizer = Normalizer(training_dataset)

normalized_training_dataset = normalizer.normalize_means(training_dataset)
normalized_test_dataset = normalizer.normalize_means(test_dataset)

normalized_lda = LDAClassifier(normalized_training_dataset)

normalized_training_classified = normalized_lda.classify_dataset(normalized_training_dataset)
normalized_test_classified = normalized_lda.classify_dataset(normalized_test_dataset)

normalized_training_accuracy = LDAClassifier.find_accuracy(training_dataset, normalized_training_classified)
normalized_test_accuracy = LDAClassifier.find_accuracy(test_dataset, normalized_test_classified)

print 'Normalized training set accuracy: ' + str(normalized_training_accuracy)
print 'Normalized test set accuracy: ' + str(normalized_test_accuracy)

class_sets = training_dataset.class_sets
normalized_class_sets = normalized_training_dataset.class_sets

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

a1 = fig.add_subplot(2, 1, 1)

class_colors = {
    0: np.random.random(3),
    1: np.random.random(3),
    2: np.random.random(3)
}

for class_name, data_set in class_sets.iteritems():
    fig = Gauss.draw_gauss_multi(
        sample=data_set.unpack_numpy_array(),
        label='training class: ' + str(class_name),
        fig=fig,
        color=class_colors[class_name]
    )

ax2 = fig.add_subplot(2, 1, 2)

for class_name, data_set in normalized_class_sets.iteritems():
    fig = Gauss.draw_gauss_multi(
        sample=data_set.unpack_numpy_array(),
        label='training normalized class: ' + str(class_name),
        fig=fig,
        color=class_colors[class_name]
    )

plt.show()

############################# Sunspot prediction ###############################

sunspot_training_dataset = DataReader.read_data("sunspotsTrainStatML.dt")
sunspot_test_dataset 	 = DataReader.read_data("sunspotsTestStatML.dt")

selection1 = map(lambda x: [x.params[2], x.params[3]], sunspot_training_dataset)
selection2 = map(lambda x: [x.params[4]], sunspot_training_dataset)
selection3 = map(lambda x: [x.params], sunspot_training_dataset)