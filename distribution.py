from abc import ABCMeta, abstractmethod

import numpy as np

class Distribution(object):

    """

    Abstract base class for distributions. Member functions that are abstract are those that
    provide the likelihood, log-likelihood, and parameter names. Provides methods to update
    distribution parameters, and operators to get parameters by name, and obtain the
    log-posterior (log likelihood + log prior).

    """

    __metaclass__ = ABCMeta

    def __init__(self, name, parameters = None):

        self.name = name
        self.parameters = parameters

    def updateParameters(self, parameters):

        for p in parameters.items():
            self.updateParameter(p, parameters[p])

    def updateParameter(self, paramName, paramValue):

        if not hasattr(self, paramName):
            return

        setattr(self, paramName, paramValue)

    @abstractmethod
    def getParameterNames(self):

        pass

    @abstractmethod
    def prob(self, data):

        pass

    @abstractmethod
    def lnprob(self, data):

        pass

    def hasDefaultPrior(self):

        return False

    def prior(self, data):

        return np.ones(data.shape)

    def lnprior(self, data):

        return np.zeros(data.shape)

    def __getitem__(self, name):

        if hasattr(self, name):
            return getattr(self, name)
        else:
            # throw
            return None

    def __call__(self, data):

        lnprior = self.lnprior(data)

        return lnprior if any(lnprior == -np.inf) else lnprior + self.lnprob(data)

class Gaussian(Distribution):

    """

    One dimensional Gaussian (normal) distribution. Inherits from Distribution. Parameterised
    with mean and width (sigma).

    """

    # Takes dictionary of Parameters with name mean and sigma
    def __init__(self, name, parameters = None):

        super(Gaussian, self).__init__(name, parameters)

        # Names correspond to input parameter dictionary
        self.mean = self.parameters['mean']
        self.sigma = self.parameters['sigma']

        self.meanParamName = 'mean'
        self.sigmaParamName = 'sigma'

        # Names of actual parameter objects
        self.paramNames = [p.name for p in parameters]

    # mean, sigma are functions that always return the mean, sigma parameter from the dictionary,
    # which is updatable , without knowing the exact name of the sigma parameter in this model

    @property
    def sigma(self):

        return self.parameters[self.sigmaParamName]

    @property
    def mean(self):

        return self.parameters[self.meanParamName]

    def prob(self, data):

        m = self.mean
        s = self.sigma

        g = 1. / np.sqrt( 2. * np.pi * s ** 2 )
        e = - ((data - m) ** 2) / (2. * s ** 2)

        return g * np.exp(e)

    def getParameterNames(self):

        return self.paramNames

    def lnprob(self, data):

        m = self.mean
        s = self.sigma

        g = 1. / np.sqrt( 2. * np.pi * s ** 2 )
        e = - ((data - m) ** 2) / (2. * s ** 2)

        return np.log(g) * e

    def hasDefaultPrior(self):

        return True

    def prior(self, data):

        p = 1.0 if self.sigma > 0.0 else 0.0

        return p * np.ones(data.shape)

    def lnprior(self, data):

        p = 0.0 if self.sigma > 0.0 else -np.inf

        return p * np.ones(data.shape)

if __name__ == '__main__':

    gaus = Gaussian('gaus', {'mean' : 0., 'sigma' : 1.})
