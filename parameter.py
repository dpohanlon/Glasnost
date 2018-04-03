import numpy as np

class Parameter(object):
    """docstring for [object Object]."""

    def __init__(self, name, initialValue, minVal = None, maxVal = None, fixed = False, priorDistribution = None):

        self.name = name
        self.initialValue = initialValue
        self.value_ = self.initialValue

        self.min = minVal if maxVal else value - 3. * value
        self.max = maxVal if maxVal else value + 3. * value

        self.priorDistribution = priorDistribution

        self.fixed = False

        # Can envision blinding, errors, etc

    @property
    def value(self):
        if self.priorDistribution = False:
            return self.value_
        else:
            return self.priorDistribution.randomSample()

    def updateValue(self, value):
        self.value_ = value
