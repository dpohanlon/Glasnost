import numpy as np

np.random.seed(42)

from iminuit import Minuit

import emcee

import sys

class Fitter(object):
    """

    Class that wraps the fitter backends and initialises the model in each case.

    """

    def __init__(self, model, backend = 'minuit'):

        self.validBackends = ['minuit',
                              'minos',
                              'minuit-migrad-hesse',
                              'minuit-migrad-minos-hesse',
                              'emcee']

        self.model = model

        if backend not in self.validBackends:
            backend = 'minuit'
            print('WARNING: Backend %s not a valid backend; defaulting to Minuit.' % (backend) )

        self.backend = backend

    def fit(self, data, verbose = False, **kwargs):

        self.model.setData(data)

        minimiser = None

        if self.backend in ['minuit', 'minuit-migrad-hesse', 'minos', 'minuit-minos']:

            # Initialise Minuit class from iminuit
            minuit = Minuit(self.model, errordef = 1.0, **self.model.getInitialParameterValuesAndStepSizes(), **kwargs)

            stdout = sys.stdout

            if not verbose:

                # Forcibly gag the minuit output to stdout
                sys.stdout = None

            # Run migrad and hesse (for covariance matrix)
            minuit.migrad()
            minuit.hesse()

            if self.backend in ['minos', 'minuit-minos']:

                # Run minos (for asymmstric per-parameter uncertainties)
                minuit.minos()

            # reset stdout
            sys.stdout = stdout

            minimiser = minuit

        if self.backend in ['emcee']:

            nwalkers = 50

            params = self.model.floatingParameterNames
            initParams = np.array([self.model.parameters[p].value_ for p in params])
            ndim = len(initParams)

            ipos = [initParams + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

            minimiser = emcee.EnsembleSampler(nwalkers, ndim, self.model.logL, threads = 1)

            minimiser.run_mcmc(ipos, 10000)

        return minimiser
