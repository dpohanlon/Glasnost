from Glasnost.distribution import Distribution

import numpy as np

class Model(Distribution):

    """

    Class corresponding to a composite likelihood model. Inherits from Distribution (implements prob
    and log-prob functions). Initialised with yields (dictionary of names to Parameters) and fit
    components (dictionary of names to Distributions). By default, assume extended maximum likelihood
    fit, where all input distributions are summed.

    """

    def __init__(self, name, initialFitYields = None, initialFitComponents = None):

        super(Model, self).__init__(name)

        # TODO: Fix me

        self.fitYields = initialFitYields

        # dictionary of (model name, distribution)
        self.fitComponents = initialFitComponents

    def getParameterNames(self):
        names = []
        for c in self.fitComponents.values():
            names += list(map(lambda x : self.name + '-' + x, c.getParameterNames()))

        return names

    def prob(self, data):

        return np.exp(self.lnprob(data))

    def lnprob(self, data):

        # This assumes that the total likelihood is a sum over components

        nObs = len(data)
        totalYield = np.sum(list(self.fitYields.values()))

        # COPIES of dictionary values
        # In future: https://docs.python.org/2/library/stdtypes.html#dictionary-view-objects
        components = list(self.fitComponents.values())
        yields = list(self.fitYields.values())

        # Matrix of (nComponents, nData) -> uses lots of memory, rewrite using einsum?
        p = np.vstack([ yields[i] * components[i].prob(data) for i in range(len(components)) ])

        # Sum across component axis, vector of length nData
        p = np.sum(p, 0)

        # Take log of each component, (sum over data axis to get total log-likelihood)
        p = np.log(p)

        return p

    def probVal(self, data):

        return np.exp(self.lnprobVal(data))

    def lnprobVal(self, data):

        # With EML criteria

        nObs = len(data)
        totalYield = np.sum(list(self.fitYields.values()))

        return np.sum(self.lnprob(data)) + nObs * np.log(totalYield) - totalYield
