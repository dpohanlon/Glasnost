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

        values = {v.name : v.value for v in self.getFloatingParameters().values()}

        return values

    def getFloatingParameters(self):

        params = {}

        for c in self.fitComponentsSimultaneous:
            for v in c.getParameters().values():
                params[v.name] = v

        return params

    def parameterRangeLnPriors(self):

        return 0 # Simultaneous, handled by the sub-models

    def sample(self, sentinel = None, nEvents = None, minVal = None, maxVal = None):

        if nEvents != None:

            print('Simultaneous model not configured for generating with yields')
            exit(1)

        z = [c.sample(minVal = minVal, maxVal = maxVal) for c in self.fitComponentsSimultaneous]

        return z

    def lnprob(self, data):

        # Already lnProb - just sum these to get the total likelihood

        z = np.vstack([ self.fitComponentsSimultaneous[i].lnprobVal(data[i]) for i in range(len(self.fitComponentsSimultaneous)) ])
        return np.sum(z)

    def lnprobVal(self, data):

        # No EML
        return self.lnprob(data)
