import numpy as np

np.random.seed(42)

from multiprocessing import Pool

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
            params[name].updateError(np.std(vals)) # Do this for now, but in future do something better!

    def postProcessMinuit(self, minimiser):
        # Make sure that values and errors are set

        paramNames = self.model.getFloatingParameterNames()
        params = self.model.getFloatingParameters()

        for name in paramNames:

            params[name].updateValue(minimiser.values[name])
            params[name].updateError(minimiser.errors[name])

    def fit(self, data, verbose = False,
            nIterations = 1000, # For emcee
            nWalkers = 100, # For emcee
            nCPU = 1 # For emcee
            ):

        self.model.setData(data)

        if self.backend in ['minuit', 'minuit-migrad-hesse', 'minos', 'minuit-minos']:

            # Initialise Minuit class from iminuit
            minuit = Minuit(self.model, errordef = 1.0, **self.model.getInitialParameterValuesAndStepSizes())

            stdout = sys.stdout

            if not verbose:

                # Forcibly gag the minuit output to stdout
                sys.stdout = None

            # Run migrad and hesse (for covariance matrix)
            minuitRet = minuit.migrad()
            hesseRet = minuit.hesse()

            if self.backend in ['minos', 'minuit-minos']:

                # Run minos (for asymmstric per-parameter uncertainties)
                minuit.minos()

            # reset stdout
            sys.stdout = stdout

            self.postProcessMinuit(minuit)

            return minuit, minuitRet, hesseRet

        if self.backend in ['emcee']:

            params = self.model.floatingParameterNames

            initParams = np.array([self.model.parameters[p].value_ for p in params])
            ndim = len(initParams)

            ipos = [initParams + 1e-4 * np.random.randn(ndim) for i in range(nWalkers)]

            with Pool(nCPU) as pool:

                minimiser = emcee.EnsembleSampler(nWalkers, ndim, self.model.logL, pool=pool)

                # A hack for a tqdm progress bar
                for pos, lnp, rstate in tqdm(minimiser.sample(ipos, iterations = nIterations),
                                             desc = 'Running Emcee',
                                             total = nIterations):
                    pass

                self.postProcessMCMC(minimiser)

            return minimiser
