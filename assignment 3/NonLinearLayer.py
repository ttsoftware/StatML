from pybrain.structure.modules.neuronlayer import NeuronLayer

class CustomLayer(NeuronLayer):
"""Layer implementing the custom function."""

def _forwardImplementation(self, inbuf, outbuf):
    outbuf[:] = custom_func_fwd(inbuf)

def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
    inerr[:] = custom_func_bkwd(outbuf,outerr)