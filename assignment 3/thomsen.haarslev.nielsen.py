from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer
from pybrain.structure import FullConnection
import NonLinearLayer

network = FeedForwardNetwork()

inputLayer = # SOMETHING
hiddenLayer = NonLinearLayer.CustomLayer(1)
outputLayer = LinearLayer(1)

network.addInputModule(inputLayer)
network.addModule(hiddenLayer)
network.addOutputModule(outputLayer)

in_to_hidden = FullConnection(inputLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outputLayer)

network.addConnection(in_to_hidden)
network.addConnection(hidden_to_out)

network.sortModules()