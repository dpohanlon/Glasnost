import numpy as np

from glasnost.model import Model

class SimultaneousModel(Model):
    """

    Class derived from the Model class to provide functionality for simultaneous models, that are treated
    as the sum over all log-likelihoods constructed from (model, data) pairs.

    """

    def __init__(self, initialFitComponents, data = None, name = ''):

        self.fitComponentsSimultaneous = initialFitComponents # List of components

        # Don't pass fitComponents - not a dict
        # Init after setting fitComponentsSimultaneous, as init calls this method
        # These should really derive from a single Model class
        super(SimultaneousModel, self).__init__(name = name, data = data)

        self.parameters = {}

        for component in self.fitComponentsSimultaneous:
            for parameter in component.getParameters().values():
                self.parameters[parameter.name] = parameter

    def getTotalYield(self):

        return None # No yields - simultaneous

    def getComponentFloatingParameterNames(self):

        names = []

        for c in self.fitComponentsSimultaneous:

            names += list(map(lambda x : x, c.getFloatingParameterNames()))

        return names

    def getFloatingParameterNames(self):

        names = set(self.getComponentFloatingParameterNames())

        return sorted(list(names))

    def getFloatingParameterValues(self):

        values = {}

        for c in self.fitComponentsSimultaneous:
            for v in c.getParameters().values():
                values[v.name] = v.value

        return values

    def parameterRangeLnPriors(self):

        return 0 # Simultaneous, handled by the sub-models

    def lnprob(self, data):

        # Already lnProb - just sum these to get the total likelihood

        z = np.vstack([ self.fitComponentsSimultaneous[i].lnprobVal(data[i]) for i in range(len(self.fitComponentsSimultaneous)) ])
        return np.sum(z)

    def lnprobVal(self, data):

        # No EML
        return self.lnprob(data)
