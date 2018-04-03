import numpy as np

class Parameter(object):

    """

    Parameter class for storing parameter information. This includes the name, value, ranges, whether the
    parameter is fixed, and the prior probability distribution of the parameter. Contains member functions
    to access and modify the value of the parameter.

    """

    def __init__(self, name, initialValue, minVal = None, maxVal = None, fixed = False, priorDistribution = None):

        self.name = name
        self.initialValue = initialValue
        self.value_ = self.initialValue

        self.min = minVal if maxVal else self.initialValue - 3. * self.initialValue
        self.max = maxVal if maxVal else self.initialValue + 3. * self.initialValue

        self.priorDistribution = priorDistribution

        self.fixed = False

        # Can envision blinding, errors, etc

    @property
    def value(self):
        if self.priorDistribution = False:
            return self.value_
        else:
            return self.priorDistribution.randomSample() # TODO: Implement me

    def updateValue(self, value):
        self.value_ = value
