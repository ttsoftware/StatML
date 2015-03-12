from __future__ import division
import math

class SupportVectorMachine(object):


    def GaussKernel(self, x, z, gamma):
        delta = math.sqrt((1/(2*gamma)))
        return math.exp(-delta * len(x-z)**2)