from pybrain.structure.modules.neuronlayer import NeuronLayer


class CustomLayer(NeuronLayer):
    """Layer implementing the custom function."""

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf / (1 + abs(inbuf))

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        raise Exception("NOT YET IMPLEMENTED")
        #inerr[:] = custom_func_bkwd(outbuf, outerr)