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

def simpleGaussianWithExpModel(mean, width, a, nEventsGauss, nEventsExp):

    with gl.name_scope('simpleGaussianWithExpTest'):

        with gl.name_scope('gauss'):

            m = gl.Parameter(mean, name = 'mean', minVal = 4200, maxVal = 6000)
            s = gl.Parameter(width, name = 'sigma', minVal = 0, maxVal = width * 5)

            gauss = gl.Gaussian({'mean' : m, 'sigma' : s})

        with gl.name_scope('exp'):

            # Maybe have a global scope for these, if not otherwise specified
            min = gl.Parameter(4200., name = 'min', fixed = True)
            max = gl.Parameter(6000., name = 'max', fixed = True)

            aExp = gl.Parameter(a, name = 'a', minVal = -0.05, maxVal = -0.0001)

            exp = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

        gaussYield = gl.Parameter(nEventsGauss, name = 'gaussYield', minVal = 0.8 * nEventsGauss, maxVal = 1.2 * nEventsGauss)
        expYield = gl.Parameter(nEventsExp, name = 'expYield', minVal = 0.8 * nEventsExp, maxVal = 1.2 * nEventsExp)

    fitYields = {gauss.name : gaussYield, exp.name : expYield}
    fitComponents = {gauss.name : gauss, exp.name : exp}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200, maxVal = 6000)

    return model

def doubleGaussianYieldsModel(mean1, width1, nEvents1, mean2, width2, nEvents2):

    with gl.name_scope('doubleGaussianYieldsModel'):

        with gl.name_scope('gauss1'):

            m1 = gl.Parameter(mean1, name = 'mean', minVal = 4200, maxVal = 5700)
            s1 = gl.Parameter(width1, name = 'sigma', minVal = 0, maxVal = width1 * 5)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

            gauss1Yield = gl.Parameter(nEvents1, name = 'gauss1Yield', minVal = 0.8 * nEvents1, maxVal = 1.2 * nEvents1)

        with gl.name_scope('gauss2'):

            m2 = gl.Parameter(mean2, name = 'mean', minVal = 4200, maxVal = 5700)
            s2 = gl.Parameter(width2, name = 'sigma', minVal = 0, maxVal = width2 * 5)

            gauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

            gauss2Yield = gl.Parameter(nEvents2, name = 'gauss2Yield', minVal = 0.8 * nEvents2, maxVal = 1.2 * nEvents2)

    fitYields = {gauss1.name : gauss1Yield, gauss2.name : gauss2Yield}
    fitComponents = {gauss1.name : gauss1, gauss2.name : gauss2}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200, maxVal = 5700)

    return model

def doubleGaussianFracModel(mean1, width1, frac, mean2, width2, nEvents):

    with gl.name_scope('doubleGaussianFracModel'):

        with gl.name_scope('gauss1'):

            m1 = gl.Parameter(mean1, name = 'mean', minVal = 4200, maxVal = 5700)
            s1 = gl.Parameter(width1, name = 'sigma', minVal = 0, maxVal = width1 * 5)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope('gauss2'):

            m2 = gl.Parameter(mean2, name = 'mean', minVal = 4200, maxVal = 5700)
            s2 = gl.Parameter(width2, name = 'sigma', minVal = 0, maxVal = width2 * 5)

            gauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

        gaussFrac = gl.Parameter(frac, name = 'gaussFrac', minVal = 0.0, maxVal = 1.0)
        totalYield = gl.Parameter(nEvents, name = 'totalYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

    fitComponents = {gauss1.name : gauss1, gauss2.name : gauss2}
    doubleGaussian = gl.Model(initialFitFracs = {gauss1.name : gaussFrac}, initialFitComponents = fitComponents, minVal = 4200, maxVal = 5700)

    model = gl.Model(initialFitYields = {doubleGaussian.name : totalYield}, initialFitComponents = {doubleGaussian.name : doubleGaussian}, minVal = 4200, maxVal = 5700)

    return model

def simpleCBModel(mean, width, aVal, nVal, nEvents):

    with gl.name_scope('simpleCBTest'):

        m = gl.Parameter(mean, name = 'mean', minVal = -1., maxVal = 1.)
        s = gl.Parameter(width, name = 'sigma', minVal = 0.1 * width, maxVal = width * 2.0)
        a = gl.Parameter(aVal, name = 'a', minVal = 0 * aVal, maxVal = 1.5 * aVal)
        n = gl.Parameter(nVal, name = 'n', fixed = True)

        cb = gl.CrystalBall({'mean' : m, 'sigma' : s, 'a' : a, 'n' : n})

        cbYield = gl.Parameter(nEvents, name = 'cbYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

    fitYields = {cb.name : cbYield}
    fitComponents = {cb.name : cb}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = -10., maxVal = 10.)

    return model

def testSimpleGaussian():

    # Test generating and fitting back with the same model

    # model = simpleGaussianModel(5279., 20., 1000000.)
    model = simpleGaussianModel(4200., 20., 1000000.)

    dataGen = model.sample(minVal = 4200., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleGaussianTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianTest/mean' : 4200.,
                       'simpleGaussianTest/sigma' : 20,
                       'simpleGaussianTest/gaussYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleGaussianWithExp():

    # Test generating and fitting back with the same model

    model = simpleGaussianWithExpModel(5279., 150., -0.002, 1000000., 2000000.)

    dataGen = model.sample(minVal = 4200., maxVal = 6000.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 300)
    # plotter = gl.Plotter(data = dataGen)
    # plotter.plotData(nDataBins = 300)
    plt.savefig('simpleGaussianWithExpTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianWithExpTest/gauss/mean' : 5279.,
                       'simpleGaussianWithExpTest/gauss/sigma' : 150,
                       'simpleGaussianWithExpTest/exp/a' : -0.002,
                       'simpleGaussianWithExpTest/expYield' : 2000000.,
                       'simpleGaussianWithExpTest/gaussYield' : 1000000.}

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

def testSimpleCB():

    # Test generating and fitting back with the same model

    model = simpleCBModel(0., 1., 1.0, 1.1, 100000.)

    dataGen = model.sample(minVal = -10, maxVal = 10.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleCBTest.pdf')
    plt.clf()

    generatedParams = {'simpleCBTest/mean' : 0.,
                       'simpleCBTest/sigma' : 1.,
                       'simpleCBTest/a' : 1.0,
                       'simpleCBTest/cbYield' : 100000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testDoubleGaussianYields():

    model = doubleGaussianYieldsModel(5279., 20., 50000, 5379., 20., 30000)

    dataGen = model.sample(minVal = 4200., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('doubleGaussianYieldsTest.pdf')
    plt.clf()

    generatedParams = {'doubleGaussianYieldsModel/gauss1/mean' : 5279.,
                       'doubleGaussianYieldsModel/gauss2/mean' : 5379.,
                       'doubleGaussianYieldsModel/gauss1/sigma' : 20.,
                       'doubleGaussianYieldsModel/gauss2/sigma' : 20.,
                       'doubleGaussianYieldsModel/gauss1/gauss1Yield' : 50000.,
                       'doubleGaussianYieldsModel/gauss2/gauss2Yield' : 30000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testDoubleGaussianFrac():

    model = doubleGaussianFracModel(5279., 15., 0.75, 5379., 20., 10000.)

    dataGen = model.sample(minVal = 4200., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('doubleGaussianFracTest.pdf')
    plt.clf()

    generatedParams = {'doubleGaussianFracModel/gauss1/mean' : 5279.,
                       'doubleGaussianFracModel/gauss2/mean' : 5379.,
                       'doubleGaussianFracModel/gauss1/sigma' : 15.,
                       'doubleGaussianFracModel/gauss2/sigma' : 20.,
                       'doubleGaussianFracModel/gaussFrac' : 0.75,
                       'doubleGaussianFracModel/totalYield' : 10000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())


if __name__ == '__main__':

    print(np.sum([testSimpleCB(),
                  testDoubleGaussianYields(),
                  testDoubleGaussianFrac(),
                  testSimpleGaussian(),
                  ] ))

    # print(testSimpleGaussianWithExp())
