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

def studentsTModel(mean, width, nu, nEvents):

    with gl.name_scope('studentsTTest'):

        m = gl.Parameter(mean, name = 'mean', minVal = 5000, maxVal = 5600)
        s = gl.Parameter(width, name = 'sigma', minVal = 0, maxVal = width * 5)
        n = gl.Parameter(nu, name = 'nu', minVal = 0, maxVal = nu * 3)

        studentsT = gl.StudentsT({'mean' : m, 'sigma' : s, 'nu' : n})

        studentsTYield = gl.Parameter(nEvents, name = 'studentsTYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

    fitYields = {studentsT.name : studentsTYield}
    fitComponents = {studentsT.name : studentsT}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 5000, maxVal = 5600)

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

def simpleGaussianWithUniformModel(mean, width, nEventsGauss, nEventsUni):

    with gl.name_scope('simpleGaussianWithUniformTest'):

        with gl.name_scope('gauss'):

            m = gl.Parameter(mean, name = 'mean', minVal = 4200, maxVal = 6000)
            s = gl.Parameter(width, name = 'sigma', minVal = 0, maxVal = width * 5)

            gauss = gl.Gaussian({'mean' : m, 'sigma' : s})

        with gl.name_scope('uni'):

            # Maybe have a global scope for these, if not otherwise specified
            min = gl.Parameter(4200., name = 'min', fixed = True)
            max = gl.Parameter(6000., name = 'max', fixed = True)

            uni = gl.Uniform({'min' : min, 'max' : max})

        gaussYield = gl.Parameter(nEventsGauss, name = 'gaussYield', minVal = 0.8 * nEventsGauss, maxVal = 1.2 * nEventsGauss)
        uniYield = gl.Parameter(nEventsUni, name = 'uniYield', minVal = 0.8 * nEventsUni, maxVal = 1.2 * nEventsUni)

    fitYields = {gauss.name : gaussYield, uni.name : uniYield}
    fitComponents = {gauss.name : gauss, uni.name : uni}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200., maxVal = 6000.)

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

def simultaneousGaussiansModel(mean1, width1, nEvents1, width2, nEvents2):

    with gl.name_scope('simultaneousGaussiansModel'):

        m1 = gl.Parameter(mean1, name = 'mean', minVal = 5250., maxVal = 5400.)

        with gl.name_scope('gauss1'):

            s1 = gl.Parameter(width1, name = 'sigma', minVal = 0.1, maxVal = width1 * 1.5)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

            gauss1Yield = gl.Parameter(nEvents1, name = 'gauss1Yield', minVal = 0.8 * nEvents1, maxVal = 1.2 * nEvents1)

        with gl.name_scope('gauss2'):

            s2 = gl.Parameter(width2, name = 'sigma', minVal = 0.1, maxVal = width2 * 1.5)

            gauss2 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

            gauss2Yield = gl.Parameter(nEvents2, name = 'gauss2Yield', minVal = 0.8 * nEvents2, maxVal = 1.2 * nEvents2)

    fitYields1 = {gauss1.name : gauss1Yield}
    fitComponents1 = {gauss1.name : gauss1}

    model1 = gl.Model(name = 's1', initialFitYields = fitYields1, initialFitComponents = fitComponents1, minVal = 5000, maxVal = 5600)

    fitYields2 = {gauss2.name : gauss2Yield}
    fitComponents2 = {gauss2.name : gauss2}

    model2 = gl.Model(name = 's2', initialFitYields = fitYields2, initialFitComponents = fitComponents2, minVal = 5000, maxVal = 5600)

    model = gl.SimultaneousModel(name = 's', initialFitComponents = [model1, model2])

    return model, model1, model2

def simultaneousModelLarge(mean1, width1, width2, a, nSignal, nBkg):

    nSignal1, nSignal2, nSignal3, nSignal4 = nSignal
    nBkg1, nBkg2, nBkg3, nBkg4 = nBkg

    with gl.name_scope('simultaneousModelLarge'):

        max = gl.Parameter(5600., name = 'max', fixed = True)
        min = gl.Parameter(5000., name = 'min', fixed = True)

        m1 = gl.Parameter(mean1, name = 'mean', minVal = 5250., maxVal = 5400.)
        s1 = gl.Parameter(width1, name = 'sigma1', minVal = 0.1, maxVal = width1 * 1.5)
        s2 = gl.Parameter(width2, name = 'sigma2', minVal = 0.1, maxVal = width2 * 1.5)

        aExp = gl.Parameter(a, name = 'a', minVal = -0.05, maxVal = -0.0001)

        with gl.name_scope('model1'):

            with gl.name_scope('sig1'):

                gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

                gauss1Yield = gl.Parameter(nSignal1, name = 'gauss1Yield', minVal = 0.8 * nSignal1, maxVal = 1.2 * nSignal1)

            with gl.name_scope('bkg1'):

                exp1 = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

                exp1Yield = gl.Parameter(nBkg1, name = 'exp1Yield', minVal = 0.8 * nBkg1, maxVal = 1.2 * nBkg1)

            model1 = gl.Model(name = 's1', initialFitYields = {gauss1.name : gauss1Yield, exp1.name : exp1Yield}, initialFitComponents = {gauss1.name : gauss1, exp1.name : exp1}, minVal = 5000, maxVal = 5600)

        with gl.name_scope('model2'):

            with gl.name_scope('sig2'):

                gauss2 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

                gauss2Yield = gl.Parameter(nSignal2, name = 'gauss2Yield', minVal = 0.8 * nSignal2, maxVal = 1.2 * nSignal2)

            with gl.name_scope('bkg2'):

                exp2 = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

                exp2Yield = gl.Parameter(nBkg2, name = 'exp2Yield', minVal = 0.8 * nBkg2, maxVal = 1.2 * nBkg2)

            model2 = gl.Model(name = 's2', initialFitYields = {gauss2.name : gauss2Yield, exp2.name : exp2Yield}, initialFitComponents = {gauss2.name : gauss2, exp2.name : exp2}, minVal = 5000, maxVal = 5600)

        with gl.name_scope('model3'):

            with gl.name_scope('sig3'):

                gauss3 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

                gauss3Yield = gl.Parameter(nSignal3, name = 'gauss3Yield', minVal = 0.8 * nSignal3, maxVal = 1.2 * nSignal3)

            with gl.name_scope('bkg3'):

                exp3 = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

                exp3Yield = gl.Parameter(nBkg3, name = 'exp3Yield', minVal = 0.8 * nBkg3, maxVal = 1.2 * nBkg3)

            model3 = gl.Model(name = 's3', initialFitYields = {gauss3.name : gauss3Yield, exp3.name : exp3Yield}, initialFitComponents = {gauss3.name : gauss3, exp3.name : exp3}, minVal = 5000, maxVal = 5600)

        with gl.name_scope('model4'):

            with gl.name_scope('sig4'):

                gauss4 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

                gauss4Yield = gl.Parameter(nSignal4, name = 'gauss4Yield', minVal = 0.8 * nSignal4, maxVal = 1.2 * nSignal4)

            with gl.name_scope('bkg4'):

                exp4 = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

                exp4Yield = gl.Parameter(nBkg4, name = 'exp4Yield', minVal = 0.8 * nBkg4, maxVal = 1.2 * nBkg4)

            model4 = gl.Model(name = 's4', initialFitYields = {gauss4.name : gauss4Yield, exp4.name : exp4Yield}, initialFitComponents = {gauss4.name : gauss4, exp4.name : exp4}, minVal = 5000, maxVal = 5600)

        model = gl.SimultaneousModel(name = 's', initialFitComponents = [model1, model2, model3, model4])

    return model, [model1, model2, model3, model4]

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

    print('testSimpleGaussian')

    # Test generating and fitting back with the same model

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

def testStudentsT():

    model = studentsTModel(5279., 20., 1.5, 1000000.)

    dataGen = model.sample(minVal = 5000., maxVal = 5600.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 400, log = True)
    plt.savefig('studentsTTest.pdf')
    plt.clf()

    generatedParams = {'studentsTTest/mean' : 5279.,
                       'studentsTTest/sigma' : 20,
                       'studentsTTest/nu' : 1.5,
                       'studentsTTest/studentsTYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleGaussianWithExp():

    print('testSimpleGaussianWithExp')

    # Test generating and fitting back with the same model

    model = simpleGaussianWithExpModel(5279., 150., -0.002, 1000000., 2000000.)

    dataGen = model.sample(minVal = 4200., maxVal = 6000.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 300)
    plt.savefig('simpleGaussianWithExpTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianWithExpTest/gauss/mean' : 5279.,
                       'simpleGaussianWithExpTest/gauss/sigma' : 150,
                       'simpleGaussianWithExpTest/exp/a' : -0.002,
                       'simpleGaussianWithExpTest/expYield' : 2000000.,
                       'simpleGaussianWithExpTest/gaussYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleGaussianWithUniform():

    print('testSimpleGaussianWithUniform')

    model = simpleGaussianWithUniformModel(5279., 150., 1000000., 2000000.)

    dataGen = model.sample(minVal = 4200., maxVal = 6000.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 300)
    plt.savefig('simpleGaussianWithUniformTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianWithUniformTest/gauss/mean' : 5279.,
                       'simpleGaussianWithUniformTest/gauss/sigma' : 150,
                       'simpleGaussianWithUniformTest/uniYield' : 2000000.,
                       'simpleGaussianWithUniformTest/gaussYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleGaussianGen():

    print('testSimpleGaussianGen')

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

    print('testSimpleGaussianMCMC')

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

    print('testSimpleCB')

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

def testSimultaneousGaussians():

    print('testSimultaneousGaussians')

    simModel, model1, model2 = simultaneousGaussiansModel(5279., 15., 500000, 30., 300000)

    dataGen = simModel.sample(minVal = 5000., maxVal = 5600.)
    fitter = gl.Fitter(simModel, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter1 = gl.Plotter(model1, dataGen[0])
    plotter1.plotDataModel(nDataBins = 100)
    plt.xlim(5100, 5500)
    plt.savefig('simultaneousGaussiansTest1.pdf')
    plt.clf()

    plotter2 = gl.Plotter(model2, dataGen[1])
    plotter2.plotDataModel(nDataBins = 100)
    plt.xlim(5100, 5500)
    plt.savefig('simultaneousGaussiansTest2.pdf')
    plt.clf()

    generatedParams = {'simultaneousGaussiansModel/mean' : 5279.,
                       'simultaneousGaussiansModel/gauss1/sigma' : 15.,
                       'simultaneousGaussiansModel/gauss2/sigma' : 30.,
                       'simultaneousGaussiansModel/gauss1/gauss1Yield' : 500000.,
                       'simultaneousGaussiansModel/gauss2/gauss2Yield' : 300000.}

    return parameterPullsOkay(generatedParams, simModel.getFloatingParameters())

def testSimultaneousModelLarge():

    print('testSimultaneousModelLarge')

    nSignal = (10000., 10000., 10000., 10000.)
    nBkg = (10000., 10000., 10000., 10000.)

    simModel, (model1, model2, model3, model4) = simultaneousModelLarge(5279., 15., 35., -0.003, nSignal, nBkg)

    dataGen = simModel.sample(minVal = 5000., maxVal = 5600.)
    fitter = gl.Fitter(simModel, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter1 = gl.Plotter(model1, dataGen[0])
    plotter1.plotDataModel(nDataBins = 100)
    plt.xlim(5100, 5500)
    plt.savefig('simultaneousModelLargeTest1.pdf')
    plt.clf()

    plotter2 = gl.Plotter(model2, dataGen[1])
    plotter2.plotDataModel(nDataBins = 100)
    plt.xlim(5100, 5500)
    plt.savefig('simultaneousModelLargeTest2.pdf')
    plt.clf()

    plotter3 = gl.Plotter(model3, dataGen[2])
    plotter3.plotDataModel(nDataBins = 100)
    plt.xlim(5100, 5500)
    plt.savefig('simultaneousModelLargeTest3.pdf')
    plt.clf()

    plotter4 = gl.Plotter(model4, dataGen[3])
    plotter4.plotDataModel(nDataBins = 100)
    plt.xlim(5100, 5500)
    plt.savefig('simultaneousModelLargeTest4.pdf')
    plt.clf()

    generatedParams = {'simultaneousModelLarge/a' : -0.003,
                       'simultaneousModelLarge/mean' : 5279.,
                       'simultaneousModelLarge/model1/bkg1/exp1Yield' : 10000,
                       'simultaneousModelLarge/model1/sig1/gauss1Yield' : 10000,
                       'simultaneousModelLarge/model2/bkg2/exp2Yield' : 10000,
                       'simultaneousModelLarge/model2/sig2/gauss2Yield' : 10000,
                       'simultaneousModelLarge/model3/bkg3/exp3Yield' : 10000,
                       'simultaneousModelLarge/model3/sig3/gauss3Yield' : 10000,
                       'simultaneousModelLarge/model4/bkg4/exp4Yield' : 10000,
                       'simultaneousModelLarge/model4/sig4/gauss4Yield' : 10000,
                       'simultaneousModelLarge/sigma1' : 15.,
                       'simultaneousModelLarge/sigma2' : 35. }

    return parameterPullsOkay(generatedParams, simModel.getFloatingParameters())

def testDoubleGaussianYields():

    print('testDoubleGaussianYields')

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

    print('testDoubleGaussianFrac')

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

    print(testSimultaneousModelLarge())
