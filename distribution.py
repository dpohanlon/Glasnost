from abc import ABCMeta, abstractmethod

import numpy as np

class Distribution(object):
    """docstring for [object Object]."""

    __metaclass__ = ABCMeta

    def __init__(self, name, initialParams = None):

        self.name = name
        self.initialParams = initialParams

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
    """docstring for [object Object]."""

    def __init__(self, name, initialParams = None):

        super(Gaussian, self).__init__(name, initialParams)

        self.mean = self.initialParams['mean'] if 'mean' in self.initialParams else 0.0
        self.sigma = self.initialParams['sigma'] if 'sigma' in self.initialParams else 1.0

    def prob(self, data):

        m = self.mean
        s = self.sigma

        g = 1. / np.sqrt( 2. * np.pi * s ** 2 )
        e = - ((data - m) ** 2) / (2. * s ** 2)

        return g * np.exp(e)

    def getParameterNames(self):

        return [self.name + '-mean', self.name + '-sigma']

    def lnprob(self, data):

        m = self.mean
        s = self.sigma

        g = 1. / np.sqrt( 2. * np.pi * s ** 2 )
        e = - ((data - m) ** 2) / (2. * s ** 2)

        return np.log(g) * e

    def hasDefaultPrior(self):

        return True

    def prior(self, data):

        p = 1.0 if self.sigma > 0 else 0.0

        return p * np.ones(data.shape)

    def lnprior(self, data):

        p = 0.0 if self.sigma > 0 else -np.inf

        return p * np.ones(data.shape)

if __name__ == '__main__':

    gaus = Gaussian('gaus', {'mean' : 0., 'sigma' : 1.})
