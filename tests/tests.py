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

import numpy as np

from pprint import pprint

import glasnost as gl

import corner

np.random.seed(42)

def parameterPullsEval(trueParams, fittedParams):

    pulls = []

    for param in trueParams.keys():
        pull = (fittedParams[param].value - trueParams[param]) / fittedParams[param].error
        pulls.append(np.abs(pull))

    return np.array(pulls)

def parameterPullsOkay(trueParams, fittedParams):

    pulls = parameterPullsEval(trueParams, fittedParams)

    if any(pulls[pulls > 3.0]):
        return 1
    else:
        return 0

def simpleGaussianModel(mean, width, nEvents):

    with gl.name_scope('simpleGaussianTest'):

        m = gl.Parameter(mean, name = 'mean', minVal = 4200, maxVal = 5700)
        s = gl.Parameter(width, name = 'sigma', minVal = 0, maxVal = width * 5)

        gauss = gl.Gaussian({'mean' : m, 'sigma' : s})

        gaussYield = gl.Parameter(nEvents, name = 'gaussYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

    fitYields = {gauss.name : gaussYield}
    fitComponents = {gauss.name : gauss}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200, maxVal = 5700)

    return model

def testSimpleGaussian():

    # Test generating and fitting back with the same model

    model = simpleGaussianModel(5279., 20., 1000000.)

    dataGen = model.sample(minVal = 4200., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleGaussianTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianTest/mean' : 5279.,
                       'simpleGaussianTest/sigma' : 20,
                       'simpleGaussianTest/gaussYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleGaussianGen():

    # Test generating with NumPy and fitting back with a similar model

    model = simpleGaussianModel(5270., 23., 1050000.) # Similar enough to what is generated

    dataNumpy = np.random.normal(5279, 20, 1000000)
    dataNumpy = dataNumpy[(dataNumpy > 4200) & (dataNumpy < 5700)]

    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataNumpy, verbose = True)

    plotter = gl.Plotter(model, dataNumpy)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleGaussianGenTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianTest/mean' : 5279.,
                       'simpleGaussianTest/sigma' : 20,
                       'simpleGaussianTest/gaussYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleGaussianMCMC():

    # Test generating and fitting back with the same model

    model = simpleGaussianModel(5279., 20., 10000.)

    dataGen = model.sample(minVal = 4200., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'emcee')
    res = fitter.fit(dataGen, verbose = True, nIterations = 1000, nWalkers = 10)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 30)
    plt.savefig('simpleGaussianMCMCTest.pdf')
    plt.clf()

    fig = plt.figure(figsize = (16, 12))

    samples = res.chain[:, 200:, :].reshape((-1, model.getNFloatingParameters()))
    c = corner.corner(samples, lw = 1.0)
    c.savefig('simpleGaussianMCMCTestCorner.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianTest/mean' : 5279.,
                       'simpleGaussianTest/sigma' : 20,
                       'simpleGaussianTest/gaussYield' : 10000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

if __name__ == '__main__':

    print(testSimpleGaussian())
