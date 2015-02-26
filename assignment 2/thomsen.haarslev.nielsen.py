from DataReader import DataReader
from LDAClassifier import LDAClassifier
from Normalizer import Normalizer

training_dataset = DataReader.read_iris('IrisTrain2014.dt')
test_dataset = DataReader.read_iris('IrisTest2014.dt')

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

normalized_training_classified = normalized_lda.classify_dataset(training_dataset)
normalized_test_classified = normalized_lda.classify_dataset(test_dataset)

normalized_training_accuracy = LDAClassifier.find_accuracy(normalized_training_dataset, normalized_training_classified)
normalized_test_accuracy = LDAClassifier.find_accuracy(normalized_test_dataset, normalized_test_classified)

print 'Normalized training set accuracy: ' + str(normalized_training_accuracy)
print 'Normalized test set accuracy: ' + str(normalized_test_accuracy)