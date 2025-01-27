from abc import ABCMeta, abstractmethod

import numpy as np

from scipy.special import erf, gamma, gammaincc, beta, betainc

from scipy.signal import convolve, gaussian

from scipy.integrate import quad, fixed_quad

import glasnost as gl

# Adaptive vectorised quadrature
from quadpy.line_segment import integrate_adaptive

# Caching of integrals
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import operator

class Distribution(object):

    """

    Abstract base class for distributions. Member functions that are abstract are those that
    provide the likelihood, log-likelihood, and parameter names. Provides methods to update
    distribution parameters, and operators to get parameters by name, and obtain the
    log-posterior (log likelihood + log prior).

    """

    __metaclass__ = ABCMeta

    def __init__(self, parameters = None, name = ''):

        self.name = gl.utils.nameScope.rstrip('/') if not name else gl.utils.nameScope + name

        if parameters:
            for k, p in parameters.items():
                if type(p) != gl.Parameter:
                    # Assume this is int, float, ...
                    parameters[k] = gl.Parameter(p, name = self.name + '/' + k)

        self.parameters = parameters

        self.cache = LRUCache(maxsize = 128)

    def updateParameters(self, parameters):

        for p in parameters.items():
            self.updateParameter(p, parameters[p])

    def updateParameter(self, paramName, paramValue):

        if not hasattr(self, paramName):
            return

        setattr(self, paramName, paramValue)

    def getParameters(self):
        return self.parameters

    @property
    def paramsValueTuple(self):
        return tuple(map(lambda x : x.value, self.parameters.values()))

    @abstractmethod
    def getParameterNames(self):

        pass

    @abstractmethod
    def prob(self, data):

        pass

    def getParameterNames(self):

        return self.paramNames

    def getFloatingParameterNames(self):

        return [p.name for p in filter(lambda p : not p.isFixed, self.parameters.values())]

    def lnprob(self, data):

        return np.log(self.prob(data))

    def sample(self, nEvents = None, minVal = None, maxVal = None):

        print('Sample not implemented for %s!' %(self.name))

    def integral(self, minVal, maxVal):

        return self.integral_(minVal, maxVal, self.paramsValueTuple)

    @cachedmethod(cache = operator.attrgetter('cache'), key = hashkey)
    def integral_(self, minVal, maxVal, valTuple):

        # Might need to fiddle with the tolerance sometimes
        int, err = integrate_adaptive(self.prob, [minVal, maxVal], 1E-5)

        return int

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

    def __repr__(self):
        return "%s: %s" % (self.name, self.parameters.items())

class Gaussian(Distribution):

    """

    One dimensional Gaussian (normal) distribution. Inherits from Distribution. Parameterised
    with mean and width (sigma).

    """

    # Takes dictionary of Parameters with name mean and sigma
    def __init__(self, parameters = None, name = 'gaussian'):
        super(Gaussian, self).__init__(parameters, name)

        # Names correspond to input parameter dictionary
        # self.mean = self.parameters['mean']
        # self.sigma = self.parameters['sigma']

        self.meanParamName = 'mean'
        self.sigmaParamName = 'sigma'

        # Names of actual parameter objects
        self.paramNames = [p.name for p in self.parameters.values()]

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

    def lnprob(self, data):

        m = self.mean
        s = self.sigma

        g = 1. / np.sqrt( 2. * np.pi * s ** 2 )
        e = - ((data - m) ** 2) / (2. * s ** 2)

        return np.log(g) * e

    def hasDefaultPrior(self):

        return True

    def sample(self, sentinel = None, nEvents = None, minVal = None, maxVal = None):

        if sentinel != None:
            print('Sentinel in sample %s' %(self.name))
            exit(0)

        integral = self.integral(minVal, maxVal)

        # Oversample and then truncate
        genEvents = nEvents * int(1./integral)

        # ...which is a good idea, unless we're generating in the tail
        # in which case, just do accept/reject

        if genEvents < 50 * nEvents:

            samples = np.random.normal(self.mean, self.sigma, size = int(genEvents))
            return  samples[(samples > minVal) & (samples < maxVal)]

        else:

            sampler = gl.sampler.RejectionSampler(self.prob, minVal, maxVal,
                                                  ceiling = max(self.prob(minVal), self.prob(maxVal)))
            return sampler.sample(nEvents)

    def cdf(self, x):

        erfArg = (x - self.mean) / (self.sigma * np.sqrt(2.))

        return 0.5 * (1 + erf(erfArg))

    @cachedmethod(cache = operator.attrgetter('cache'), key = hashkey)
    def integral_(self, minVal, maxVal, valTuple):

        cdfMin = self.cdf(minVal)
        cdfMax = self.cdf(maxVal)

        return cdfMax - cdfMin

    def integral(self, minVal, maxVal):

        return self.integral_(minVal, maxVal, self.paramsValueTuple)

    def prior(self, data):

        p = 1.0 if self.sigma > 0.0 else 0.0

        return p * np.ones(data.shape)

    def lnprior(self, data):

        p = 0.0 if self.sigma > 0.0 else -np.inf

        return p * np.ones(data.shape)

class Uniform(Distribution):

    """

    Uniform distribution defined in the range [min, max]. No floating parameters.

    """

    # Takes dictionary of Parameters with name mean and sigma
    def __init__(self, parameters = None, name = 'uniform'):

        super(Uniform, self).__init__(parameters, name)

        # Names correspond to input parameter dictionary

        self.minParamName = 'min'
        self.maxParamName = 'max'

        # Names of actual parameter objects
        self.paramNames = [p.name for p in self.parameters.values()]

        # Check that params are fixed

        for p in self.parameters.values():
            if not p.isFixed:
                self.fixed_ = True

    @property
    def min(self):

        return self.parameters[self.minParamName]

    @property
    def max(self):

        return self.parameters[self.maxParamName]

    def prob(self, data):

        min = self.min
        max = self.max

        p = np.full_like(data, 1. / (max - min))
        p[(data < min) | (data > max)] = 0.

        return p

    def hasDefaultPrior(self):

        return True

    def sample(self, nEvents = None, minVal = None, maxVal = None):

        if not (minVal and maxVal) : return np.random.uniform(self.min, self.max, size = int(nEvents))
        else : return np.random.uniform(minVal, maxVal, size = int(nEvents))

    def integral(self, minVal, maxVal):

        return self.integral_(minVal, maxVal, self.paramsValueTuple)

    @cachedmethod(cache = operator.attrgetter('cache'), key = hashkey)
    def integral_(self, minVal, maxVal, valTuple):

        if minVal <= self.min and maxVal >= self.max:
            return 1.0
        elif maxVal <= self.min or minVal >= self.max:
            return 0.0
        elif minVal > self.min and maxVal > self.max:
            return (self.max - minVal) / (self.max - self.min)
        elif minVal < self.min and maxVal < self.max:
            return (maxVal - self.min) / (self.max - self.min)
        else: # range is a subrange of (self.min, self.max)
            return (maxVal - minVal) / (self.max - self.min)

    def prior(self, data):

        return np.full_like(data, 1.0)

    def lnprior(self, data):

        return np.full_like(data, 0.0)

class CrystalBall(Distribution):

    """

    Crystal Ball distribution.

    """

    # Takes dictionary of Parameters with name mean and sigma
    def __init__(self, parameters = None, name = 'crystalBall'):

        super(CrystalBall, self).__init__(parameters, name)

        # Names correspond to input parameter dictionary

        self.meanParamName = 'mean'
        self.sigmaParamName = 'sigma'

        self.aParamName = 'a'
        self.nParamName = 'n'

        # Names of actual parameter objects
        self.paramNames = [p.name for p in self.parameters.values()]

    @property
    def a(self):

        return self.parameters[self.aParamName]

    @property
    def n(self):

        return self.parameters[self.nParamName]

    @property
    def sigma(self):

        return self.parameters[self.sigmaParamName]

    @property
    def mean(self):

        return self.parameters[self.meanParamName]

    def prob(self, data):

        a = self.a.value_
        n = self.n.value_
        m = self.mean.value_
        s = self.sigma.value_

        nOverA = n / np.abs(a)
        expA = np.exp(-0.5 * np.abs(a) ** 2)

        A = ( nOverA ) ** n * expA
        B = ( nOverA ) - np.abs(a)
        C = ( nOverA ) * (1./(n - 1.)) * expA
        D = np.sqrt(0.5 * np.pi) * (1. + erf(np.abs(a) / np.sqrt(2)))

        N = 1./( s * (C + D) )

        z = (data - m) / s

        v1 = N * np.exp( - 0.5 * z ** 2 )

        # This can result in a complex number if this path isn't taken
        # Make complex and then just take the real part
        # (Check whether this is faster than just branching)

        v2 = (N * A * (B - z).astype(np.complex) ** (-n))

        return np.where(z > -a, v1, np.real(v2))

    def hasDefaultPrior(self):

        return True

    def sample(self, nEvents = None, minVal = None, maxVal = None):
        sampler = gl.sampler.RejectionSampler(self.prob, minVal, maxVal, ceiling = self.prob(self.mean))

        return sampler.sample(nEvents)

    def prior(self, data):

        p = 1.0 if self.sigma > 0.0 else 0.0

        return p * np.ones(data.shape)

    def lnprior(self, data):

        p = 0.0 if self.sigma > 0.0 else -np.inf

        return p * np.ones(data.shape)

class Exponential(Distribution):

    """

    Exponential distribution, with shape parameter 'a', and min and max ranges.

    """

    # Takes dictionary of Parameters with name mean and sigma
    def __init__(self, parameters = None, name = 'exponential'):

        super(Exponential, self).__init__(parameters, name)

        # Names correspond to input parameter dictionary

        self.aParamName = 'a'

        self.minParamName = 'min'
        self.maxParamName = 'max'

        # Names of actual parameter objects
        self.paramNames = [p.name for p in self.parameters.values()]

        # Make sure the min and max range are fixed
        for p in [self.max, self.max]:
            if not p.isFixed:
                self.fixed_ = True

    @property
    def a(self):

        return self.parameters[self.aParamName]

    @property
    def min(self):

        return self.parameters[self.minParamName]

    @property
    def max(self):

        return self.parameters[self.maxParamName]

    def norm(self):
        if self.a == 0:
            return 1. / (self.max - self.min)
        else:
            return self.a / ( np.exp(self.a * self.max) - np.exp(self.a * self.min) )

    def prob(self, data):

        return self.norm() * np.exp(self.a * data)

    def sample(self, nEvents = None, minVal = None, maxVal = None):

        # Exponential is monotonic
        ceiling = np.max(self.prob(np.array([minVal, maxVal])))

        sampler = gl.sampler.RejectionSampler(self.prob, minVal, maxVal, ceiling = ceiling)

        return sampler.sample(nEvents)

    def integral(self, minVal, maxVal):

        return self.integral_(minVal, maxVal, self.paramsValueTuple)

    @cachedmethod(cache = operator.attrgetter('cache'), key = hashkey)
    def integral_(self, minVal, maxVal, valTuple):

        if minVal <= self.min and maxVal >= self.max:
            return 1.0
        elif maxVal <= self.min or minVal >= self.max:
            return 0.0
        elif minVal > self.min and maxVal > self.max:
            return (self.max - minVal) / (self.max - self.min)
        elif minVal < self.min and maxVal < self.max:
            return (maxVal - self.min) / (self.max - self.min)
        else: # range is a subrange of (self.min, self.max)
            return (maxVal - minVal) / (self.max - self.min)

class StudentsT(Distribution):

    """

    Generalised Student's-t distribution in terms of a mean, width (sigma - not the standard deviation),
    and normality parameter, nu.

    """

    # Takes dictionary of Parameters with name mean and sigma
    def __init__(self, parameters = None, name = 'studentsT'):

        super(StudentsT, self).__init__(parameters, name)

        # Names correspond to input parameter dictionary

        self.nuParamName = 'nu'
        self.meanParamName = 'mean'
        self.sigmaParamName = 'sigma'

        # Names of actual parameter objects
        self.paramNames = [p.name for p in self.parameters.values()]

    @property
    def nu(self):

        return self.parameters[self.nuParamName]

    @property
    def mean(self):

        return self.parameters[self.meanParamName]

    @property
    def sigma(self):

        return self.parameters[self.sigmaParamName]

    def lnprior(self, data):

        p = 0.0 if self.nu > 1.0 else -np.inf

        if np.isfinite(p):
            p = 0.0 if self.sigma > 0.0 else -np.inf

        return p * np.ones(data.shape)

    def sample(self, nEvents = None, minVal = None, maxVal = None):
        sampler = gl.sampler.RejectionSampler(self.prob, minVal, maxVal, ceiling = self.prob(self.mean))

        return sampler.sample(nEvents)

    def prob(self, data):

        l = 1. / ( np.abs(self.sigma) * np.sqrt(self.nu * np.pi) * gamma(0.5 * self.nu) / gamma(0.5 * (self.nu + 1)))
        r = (1. + ((data - self.mean) / self.sigma) ** 2 / self.nu) ** (-0.5 * (self.nu + 1.))

        return l * r

    def cdf(self, x):
        # Following the TF-Prob implementation

        y = (x - self.mean) / self.sigma
        x_t = self.nu / (y**2. + self.nu)

        neg_cdf = 0.5 * betainc(0.5 * self.nu, 0.5, x_t)

        return neg_cdf if y < 0. else 1. - neg_cdf

    def integral(self, minVal, maxVal):

        return self.integral_(minVal, maxVal, self.paramsValueTuple)

    @cachedmethod(cache = operator.attrgetter('cache'), key = hashkey)
    def integral_(self, minVal, maxVal, valTuple):

        cdfMin = self.cdf(minVal)
        cdfMax = self.cdf(maxVal)

        return cdfMax - cdfMin

class Beta(Distribution):

    """

    Beta distribution defined by mean and standard-deviation. For use as a
    prior distribution.

    """

    # Takes dictionary of Parameters with name mean and sigma
    def __init__(self, parameters = None, name = 'beta'):

        super(Beta, self).__init__(parameters, name)

        # Names correspond to input parameter dictionary

        self.meanParamName = 'mean'
        self.sigmaParamName = 'sigma'

        # Names of actual parameter objects
        self.paramNames = [p.name for p in self.parameters.values()]

    @property
    def mean(self):

        return self.parameters[self.meanParamName]

    @property
    def sigma(self):

        return self.parameters[self.sigmaParamName]

    @property
    def kappa(self):

        return ( (self.mean * (1. - self.mean)) / self.sigma ** 2 ) - 1.

    @property
    def alpha(self):

        return self.mean * self.kappa

    @property
    def beta(self):

        return (1. - self.mean) * self.kappa

    def sample(self, nEvents = None, minVal = None, maxVal = None):

        mode = (self.alpha - 1.) / (self.alpha + self.beta - 2.) # MPV of beta distribution

        sampler = gl.sampler.RejectionSampler(self.prob, minVal, maxVal, ceiling = self.prob(mode))

        return sampler.sample(nEvents)

    def prior(self, data):

        p = 1.0 if (self.alpha > 0.0 and self.beta > 0) else 0.0

        return p * np.ones(data.shape)

    def lnprior(self, data):

        p = 0.0 if (self.alpha > 0.0 and self.beta > 0) else -np.inf

        return p * np.ones(data.shape)

    def prob(self, data):

        n = np.power(data, self.alpha - 1.0) * np.power(1. - data, self.beta - 1.)

        return n / beta(self.alpha, self.beta)

class ARGaus(Distribution):

    """

    Generalised ARGUS distribution convoluted with a zero mean Gaussian resolution function.

    """

    # Takes dictionary of Parameters with name mean and sigma
    def __init__(self, parameters = None, name = 'argaus', minVal = 0, maxVal = 1E8, gridSize = 1000):

        super(ARGaus, self).__init__(parameters, name)

        # Names correspond to input parameter dictionary

        self.cParamName = 'c'
        self.pParamName = 'p'
        self.chiParamName = 'chi'

        self.sigmaParamName = 'sigma'

        self.minVal = minVal
        self.maxVal = maxVal
        self.gridSize = 1000

        # Names of actual parameter objects
        self.paramNames = [p.name for p in self.parameters.values()]

    @property
    def c(self):

        return self.parameters[self.cParamName]

    @property
    def p(self):

        return self.parameters[self.pParamName]

    @property
    def chi(self):

        return self.parameters[self.chiParamName]

    @property
    def sigma(self):

        return self.parameters[self.sigmaParamName]

    def prob(self, data):

        if len(data[data < self.minVal]) > 0 or len(data[data > self.maxVal]) > 0:
            print('Data exists outside of the grid range - exiting.')
            exit(1)

        grid = np.linspace(self.minVal, self.maxVal, self.gridSize)

        # For generalised ARGUS
        c = self.c.value_
        p = self.p.value_
        chi = self.chi.value_

        # For Gaussian resolution
        s = self.sigma.value_

        oneMinusChiOverCSq = (1. - (grid ** 2) / (c ** 2))

        t1n = np.power(2., -p) * np.power(chi, 2. * (p + 1.))
        t1d = gamma(p + 1.) - gammaincc(p + 1., 0.5 * chi ** 2) * gamma(p + 1.)

        t2 = (grid / (c ** 2)) * np.power(oneMinusChiOverCSq, p)
        t3 = np.exp( -0.5 * chi ** 2 * oneMinusChiOverCSq )

        argus = (t1n / t1d) * t2 * t3

        # ARGUS undefined above c, but we want to convolve, so replace nans with zero
        argus[np.isnan(argus)] = 0.

        conv = convolve(argus, gaussian(len(grid), s), mode = 'same', method = 'fft')

        pos = np.searchsorted(grid, data)

        lerp = conv[pos - 1] + ((conv[pos] - conv[pos - 1]) / (grid[pos] - grid[pos - 1])) * (data - grid[pos - 1])
        return lerp

    def hasDefaultPrior(self):

        return True

    def sample(self, nEvents = None, minVal = None, maxVal = None):
        sampler = gl.sampler.RejectionSampler(self.prob, minVal, maxVal, ceiling = 0.001)

        return sampler.sample(nEvents)

    def prior(self, data):

        p = 1.0 if self.sigma > 0.0 else 0.0

        return p * np.ones(data.shape)

    def lnprior(self, data):

        p = 0.0 if self.sigma > 0.0 else -np.inf

        return p * np.ones(data.shape)

    @cachedmethod(cache = operator.attrgetter('cache'), key = hashkey)
    def integral_(self, minVal, maxVal, valTuple):

        # Seprate fixed order integrator as the others are slow / don't converge

        int, err = fixed_quad(self.prob, minVal, maxVal, n = 30)

        return int

def simpleARGausModel(c, p, chi, sigma, nEvents):

    with gl.name_scope('simpleARGausTest'):

        cA = gl.Parameter(c, name = 'c', minVal = 4800., maxVal = 6000.)
        pA = gl.Parameter(p, name = 'p', minVal = 0., maxVal = 5.)
        chiA = gl.Parameter(chi, name = 'chi', minVal = 0., maxVal = 25.)
        sigmaA = gl.Parameter(sigma, name = 'sigma', minVal = 0., maxVal = 100.)

        argaus = gl.ARGaus({'c' : cA, 'chi' : chiA, 'p' :pA, 'sigma' : sigmaA}, minVal = 4800., maxVal = 6000., gridSize = 1000)

        argausYield = gl.Parameter(nEvents, name = 'argausYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

    fitYields = {argaus.name : argausYield}
    fitComponents = {argaus.name : argaus}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

    return model

def testSimpleARGaus():

    print('testSimpleARGaus')

    # Test generating and fitting back with the same model

    model = simpleARGausModel(5400., 1.1, 8., 30., 10000.)

    dataGen = model.sample(minVal = 4800., maxVal = 6000.)

    plt.hist(dataGen, bins = 150)
    plt.savefig('dataHist.pdf')
    plt.clf()

    print('Fitting')

    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    from pprint import pprint

    pprint(model.parameters)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleARGausTest.pdf')
    plt.clf()

    exit(0)

    fitterB = gl.Fitter(model, backend = 'emcee')
    res = fitterB.fit(dataGen, verbose = True, nIterations = 10000, nWalkers = 64) # * nparams

    import corner

    fig = plt.figure(figsize = (16, 12))

    samples = res.chain[:, 1000:, :].reshape((-1, 5))
    c = corner.corner(samples, lw = 1.0)
    c.savefig('argaus-corner.pdf')
    plt.clf()

    pprint(model.parameters)

if __name__ == '__main__':

    import matplotlib as mpl

    mpl.use('Agg')

    import matplotlib.pyplot as plt

    plt.style.use(['fivethirtyeight', 'seaborn-whitegrid', 'seaborn-ticks'])

    from matplotlib import rcParams
    from matplotlib import gridspec
    import matplotlib.ticker as plticker

    from matplotlib import cm

    rcParams['axes.facecolor'] = 'FFFFFF'
    rcParams['savefig.facecolor'] = 'FFFFFF'
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'

    rcParams.update({'figure.autolayout': True})

    testSimpleARGaus()

    exit(0)

    from parameter import Parameter

    # c = Parameter(5400., 'c')
    # p = Parameter(0.5, 'p')
    # chi = Parameter(10., 'chi')
    # s = Parameter(30., 's')

    c = Parameter(5400., minVal = 5300., maxVal = 5500.,  name = 'c')
    p = Parameter(0.5, minVal = -1.0, maxVal = 1.0, name = 'p')
    chi = Parameter(10., minVal = 0., maxVal = 30., name = 'chi')
    s = Parameter(30., minVal = 5., maxVal = 50., name = 'sigma')

    a = ARGaus(parameters = {'c' : c, 'p': p, 'chi' : chi, 'sigma' : s},
               minVal = 5000., maxVal = 5800., gridSize = 1000)

    data = a.sample(10000, 5000, 5800)
    plt.hist(data, bins = 200)
    plt.savefig('argaus.pdf')
    plt.clf()

    print('generated')

    from model import Model

    y = gl.Parameter(10000., name = 'yield', minVal = 5000., maxVal = 15000.,)

    m = Model(name = 'model', initialFitYields = {'yield' : y}, initialFitComponents = {'a' : a})

    from fitter import Fitter

    fitter = Fitter(m, backend = 'minuit')
    res = fitter.fit(data, verbose = True)

    from plotter import Plotter

    plotter = gl.Plotter(m, data)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('argaus_fit.pdf')
    plt.clf()

    # x = np.linspace(5000, 5500, 1000)
    # l = a.prob(x)
    # plt.plot(x, l, lw = 1.0)
    # plt.savefig('argaus_plot.pdf')
