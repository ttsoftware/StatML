from DataReader import DataReader
from LDA import LDA

training_dataset = DataReader.read_iris('IrisTrain2014.dt')

lda = LDA(training_dataset)