

class InputLayer(object):

    def __init__(self, dataset):
        self.dataset = dataset

        neurons = []
        for d in dataset.dimensions:
