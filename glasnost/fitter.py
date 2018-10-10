import numpy as np

np.random.seed(42)

from scipy.stats import gaussian_kde

from iminuit import Minuit

from tqdm import tqdm

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

    def getMode(self, samples):

        # Subsample so that this doesn't take forever
        if (len(samples) > 10000):
            samples = np.random.choice(samples, 10000)

        kde = gaussian_kde(samples)
        xs = np.linspace(np.min(samples), np.max(samples), 10000) # Make this settable

        interp = kde(xs)
        maxVal = xs[np.argmax(interp)]

        return maxVal

    def postProcessMCMC(self, minimiser):

        # Set parameters to values accord to the maximum of the posterior probability
        # for plotting

        chain = minimiser.chain

        # Burn in first 25%
        samples = chain[:, int(chain.shape[1] * 0.25):, :].reshape((-1, self.model.getNFloatingParameters()))

        # Names are in the same order these were given to the fitter
        paramNames = self.model.getFloatingParameterNames()
        params = self.model.getFloatingParameters()

        for i, name in enumerate(paramNames):
            vals = samples[:,i]

            map = self.getMode(vals)
            params[name].updateValue(map)

    def fit(self, data, verbose = False,
            nIterations = 1000, # For emcee
            nWalkers = 100, # For emcee
            ):

        self.model.setData(data)

        minimiser = None

        if self.backend in ['minuit', 'minuit-migrad-hesse', 'minos', 'minuit-minos']:

            # Initialise Minuit class from iminuit
            minuit = Minuit(self.model, errordef = 1.0, **self.model.getInitialParameterValuesAndStepSizes())

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

            params = self.model.floatingParameterNames

            initParams = np.array([self.model.parameters[p].value_ for p in params])
            ndim = len(initParams)

            ipos = [initParams + 1e-4 * np.random.randn(ndim) for i in range(nWalkers)]

            minimiser = emcee.EnsembleSampler(nWalkers, ndim, self.model.logL, threads = 1)

            # minimiser.run_mcmc(ipos, 1000)

            # A hack for a tqdm progress bar
            for pos, lnp, rstate in tqdm(minimiser.sample(ipos, iterations = nIterations),
                                         desc = 'Running Emcee',
                                         total = nIterations):
                pass

            self.postProcessMCMC(minimiser)

        return minimiser
