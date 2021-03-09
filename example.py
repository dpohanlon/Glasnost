import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

plt.style.use(['fivethirtyeight', 'seaborn-whitegrid', 'seaborn-ticks'])

from matplotlib import rcParams
from matplotlib import gridspec
import matplotlib.ticker as plticker

from matplotlib import cm

from collections import OrderedDict

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

def testPhysicalSimBF():

    with gl.name_scope('massFit'):

        minParam =  gl.Parameter(0.0, name = 'minVal', fixed = True)
        maxParam =  gl.Parameter(10.0, name = 'maxVal', fixed = True)

        m = gl.Parameter(5.0, name = 'mean')
        s = gl.Parameter(1.0, name = 'sigma')

        with gl.name_scope('model1'):

            with gl.name_scope("expBkg"):
                yBkg1 = gl.Parameter(500, name = 'yieldExp', minVal = -1000)

                aExp1 = gl.Parameter(-0.2, name = 'aExp')
                expBkg1 = gl.Exponential({'a' : aExp1, 'min' : minParam, 'max' : maxParam})

            with gl.name_scope("gaussSignal"):
                ySig1 = gl.Parameter(500, name = 'yieldSig', minVal = -1000)

                g1 = gl.Gaussian({'mean' : m, 'sigma' : s})

        with gl.name_scope('model2'):

            with gl.name_scope("expBkg"):
                yBkg2 = gl.Parameter(1500, name = 'yieldExp', minVal = -1000)

                aExp2 = gl.Parameter(-0.1, name = 'aExp')
                expBkg2 = gl.Exponential({'a' : aExp2, 'min' : minParam, 'max' : maxParam})

            with gl.name_scope("gaussSignal"):
                ySig2 = gl.Parameter(1000, name = 'yieldSig', minVal = -1000)

                g2 = gl.Gaussian({'mean' : m, 'sigma' : s})

    initialFitYields1 = {g1.name : ySig1, expBkg1.name : yBkg1}
    initialFitComponents1 = {g1.name : g1, expBkg1.name : expBkg1}

    initialFitYields2= {g2.name : ySig2, expBkg2.name : yBkg2}
    initialFitComponents2 = {g2.name : g2, expBkg2.name : expBkg2}

    model1 = gl.Model(initialFitYields = initialFitYields1, initialFitComponents = initialFitComponents1)
    model2 = gl.Model(initialFitYields = initialFitYields2, initialFitComponents = initialFitComponents2)

    data1 = model1.sample(0, 10)
    data2 = model2.sample(0, 10)

    model = gl.SimultaneousModel(initialFitComponents = [model1, model2])

    # plt.hist(data, bins = 50, histtype = 'stepfilled')
    # plt.savefig('textExp.pdf')

    fitter = gl.Fitter(model, backend = 'emcee')

    res = fitter.fit([data1, data2], verbose = True, nIterations = 2000)

    samples = res.chain[:, 100:, :].reshape((-1, model.getNFloatingParameters()))

    plotter = gl.Plotter(model1, data1)
    plotter.plotDataModel(nDataBins = 50)
    plt.savefig('testPlot1.pdf')
    plt.clf()

    plotter = gl.Plotter(model2, data2)
    plotter.plotDataModel(nDataBins = 50)
    plt.savefig('testPlot2.pdf')
    plt.clf()

    c = corner.corner(samples)#, truths = [-0.2, 500., 5.0, 1.0, 500.])
    c.savefig('corner.pdf')
    plt.clf()

def testPhysicalSim():

    with gl.name_scope('massFit'):

        minParam =  gl.Parameter(0.0, name = 'minVal', fixed = True)
        maxParam =  gl.Parameter(10.0, name = 'maxVal', fixed = True)

        m = gl.Parameter(5.0, name = 'mean')
        s = gl.Parameter(1.0, name = 'sigma')

        with gl.name_scope('model1'):

            with gl.name_scope("expBkg"):
                yBkg1 = gl.Parameter(500, name = 'yieldExp', minVal = -1000)

                aExp1 = gl.Parameter(-0.2, name = 'aExp')
                expBkg1 = gl.Exponential({'a' : aExp1, 'min' : minParam, 'max' : maxParam})

            with gl.name_scope("gaussSignal"):
                ySig1 = gl.Parameter(500, name = 'yieldSig', minVal = -1000)

                g1 = gl.Gaussian({'mean' : m, 'sigma' : s})

        with gl.name_scope('model2'):

            with gl.name_scope("expBkg"):
                yBkg2 = gl.Parameter(1500, name = 'yieldExp', minVal = -1000)

                aExp2 = gl.Parameter(-0.1, name = 'aExp')
                expBkg2 = gl.Exponential({'a' : aExp2, 'min' : minParam, 'max' : maxParam})

            with gl.name_scope("gaussSignal"):
                ySig2 = gl.Parameter(1000, name = 'yieldSig', minVal = -1000)

                g2 = gl.Gaussian({'mean' : m, 'sigma' : s})

    initialFitYields1 = {g1.name : ySig1, expBkg1.name : yBkg1}
    initialFitComponents1 = {g1.name : g1, expBkg1.name : expBkg1}

    initialFitYields2= {g2.name : ySig2, expBkg2.name : yBkg2}
    initialFitComponents2 = {g2.name : g2, expBkg2.name : expBkg2}

    model1 = gl.Model(initialFitYields = initialFitYields1, initialFitComponents = initialFitComponents1, minVal = 0, maxVal = 10)
    model2 = gl.Model(initialFitYields = initialFitYields2, initialFitComponents = initialFitComponents2, minVal = 0, maxVal = 10)

    data1 = model1.sample(0, 10)
    data2 = model2.sample(0, 10)

    model = gl.SimultaneousModel(initialFitComponents = [model1, model2])

    # plt.hist(data, bins = 50, histtype = 'stepfilled')
    # plt.savefig('textExp.pdf')

    fitter = gl.Fitter(model, backend = 'emcee')

    res = fitter.fit([data1, data2], verbose = True, nIterations = 2000)

    samples = res.chain[:, 100:, :].reshape((-1, model.getNFloatingParameters()))

    plotter = gl.Plotter(model1, data1)
    plotter.plotDataModel(nDataBins = 50)
    plt.savefig('testPlot1.pdf')
    plt.clf()

    plotter = gl.Plotter(model2, data2)
    plotter.plotDataModel(nDataBins = 50)
    plt.savefig('testPlot2.pdf')
    plt.clf()

    c = corner.corner(samples)#, truths = [-0.2, 500., 5.0, 1.0, 500.])
    c.savefig('corner.pdf')
    plt.clf()

def testPhysical():

    with gl.name_scope('massFit'):
        with gl.name_scope("expBkg"):
            yBkg = gl.Parameter(500, name = 'yieldExp', minVal = -1000)

            minParam =  gl.Parameter(0.0, name = 'minVal', fixed = True)
            maxParam =  gl.Parameter(10.0, name = 'maxVal', fixed = True)

            aExp = gl.Parameter(-0.2, name = 'aExp')
            # expBkg = gl.Exponential({'a' : aExp, 'min' : gl.Parameter(0.0, fixed = True), 'max' : gl.Parameter(10.0, fixed = True)})
            expBkg = gl.Exponential({'a' : aExp, 'min' : minParam, 'max' : maxParam})

        with gl.name_scope("gaussSignal"):
            ySig = gl.Parameter(500, name = 'yieldSig', minVal = -1000)

            m = gl.Parameter(5.0, name = 'mean')
            s = gl.Parameter(1.0, name = 'sigma')

            g = gl.Gaussian({'mean' : m, 'sigma' : s})

    initialFitYields = {g.name : ySig, expBkg.name : yBkg}
    initialFitComponents = {g.name : g, expBkg.name : expBkg}

    model = gl.Model(initialFitYields = initialFitYields, initialFitComponents = initialFitComponents, minVal = 0., maxVal = 10.)

    data = model.sample(0, 10)

    # plt.hist(data, bins = 50, histtype = 'stepfilled')
    # plt.savefig('textExp.pdf')

    fitter = gl.Fitter(model, backend = 'minuit')

    res = fitter.fit(data, verbose = True)#, nIterations = 5000)

    # samples = res.chain[:, 500:, :].reshape((-1, model.getNFloatingParameters()))

    plotter = gl.Plotter(model, data)
    plotter.plotDataModel(nDataBins = 30)
    plt.savefig('testPlot.pdf')
    plt.clf()

    # c = corner.corner(samples, truths = [-0.2, 500., 5.0, 1.0, 500.], lw = 1.0)
    # c.savefig('corner.pdf')
    # plt.clf()

def testToy():

    # data = np.concatenate((np.random.normal(-1., 4., 1000), np.random.normal(0., 1., 1000), np.random.normal(3., 1., 1000)))

    data1 = np.concatenate((np.random.normal(0., 1., 1000), np.random.normal(0., 3., 1000)))
    data2 = np.concatenate((np.random.normal(0., 1., 500), np.random.normal(0., 3., 500)))

    with gl.name_scope('massFit'):

        m1 = gl.Parameter(0.2, name = 'mean')
        s1 = gl.Parameter(1.0, name = 'sigma1')
        s2 = gl.Parameter(3.0, name = 'sigma2')

        with gl.name_scope("model1"):

            with gl.name_scope("firstGaussian"):

                y1 = gl.Parameter(len(data1) // 2, name = 'yield', minVal = 0)

                g1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

            with gl.name_scope("secondGaussian"):

                y2 = gl.Parameter(len(data1) // 2, name = 'yield', minVal = 0)

                g2 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

            initialFitYields1 = {g1.name : y1, g2.name : y2}
            initialFitComponents1 = {g1.name : g1, g2.name : g2}

            model1 = gl.Model(initialFitYields = initialFitYields1, initialFitComponents = initialFitComponents1, minVal = -10., maxVal = 10.)

        with gl.name_scope("model2"):

            with gl.name_scope("firstGaussian"):

                y3 = gl.Parameter(len(data2) // 2, name = 'yield', minVal = 0)

            with gl.name_scope("secondGaussian"):

                y4 = gl.Parameter(len(data2) // 2, name = 'yield', minVal = 0)

            initialFitYields2 = {g1.name : y3, g2.name : y4}

            model2 = gl.Model(initialFitYields = initialFitYields2, initialFitComponents = initialFitComponents1, minVal = -10., maxVal = 10.)

        models = {model1.name : model1, model2.name : model2}

        model = gl.SimultaneousModel(initialFitComponents = [model1, model2])

    # s1, s2 = model.sample(-10, 10)
    #
    # plt.hist(s1, bins = 100, histtype = 'stepfilled')
    # plt.savefig('samples1.pdf')
    # plt.clf()
    # plt.hist(s2, bins = 100, histtype = 'stepfilled')
    # plt.savefig('samples2.pdf')

    fitter = gl.Fitter(model, backend = 'minuit')

    res = fitter.fit([data1, data2], verbose = True)

    # samples = res.chain[:, 200:, :].reshape((-1, 7))
    #
    # means = np.mean(samples, axis = 0)
    # stds = np.std(samples, axis = 0)
    #
    # for i in range(7):
    #
    #     plt.plot(samples[:,i], lw = 2.0)
    #     plt.savefig('samples' + str(i) + '.png')
    #     plt.clf()
    #
    #     plt.hist(samples[:,i], bins = 50)
    #     plt.savefig('samplesHist' + str(i) + '.png')
    #     plt.clf()
    #
    # c = corner.corner(samples, truths = [0.0, 1000., 1000., 500., 500., 1., 3.])
    # c.savefig('corner.pdf')

    f, axarr = plt.subplots(1, 2, sharey = True)

    plotter = gl.Plotter(model1, data1)
    plotter.plotDataModel(ax = axarr[0])

    axarr[0].set_xlabel('The things')
    axarr[0].set_ylabel('How many things')

    plt.xlim(np.min(data1), np.max(data1))

    # axarr[0].set_yscale("log", nonposy='clip')
    # axarr[0].set_ylim(1)

    plotter = gl.Plotter(model2, data2)
    plotter.plotDataModel(ax = axarr[1])

    axarr[1].set_xlabel('The things')

    plt.xlim(np.min(data2), np.max(data2))

    # axarr[1].set_yscale("log", nonposy='clip')
    # axarr[1].set_ylim(1)

    plt.savefig('testPlot.pdf', bbox_inches = 'tight')

def testWithFracs():

    data = np.concatenate((np.random.normal(-2, 1., 1000), np.random.normal(-2, 3., 1000), np.random.normal(2., 1., 1000), np.random.normal(2., 3., 1000)))
    data = data[data > -10]
    data = data[data < 10]

    with gl.name_scope('fit'):

        with gl.name_scope('signal1'):

            m1_1 = gl.Parameter(-2.2, name = 'mean', minVal = -4, maxVal = 0)
            s1_1 = gl.Parameter(1.0, name = 'sigma1', minVal = 0., maxVal = 3.)
            s2_1 = gl.Parameter(3.0, name = 'sigma2', minVal = 1., maxVal = 5.)

            f1_1 = gl.Parameter(0.75, name = 'frac1', minVal = 0.0, maxVal = 1.0)

            with gl.name_scope("firstGaussian"):

                # y1 = gl.Parameter(len(data) // 2, name = 'yield', minVal = 0)

                g1_1 = gl.Gaussian({'mean' : m1_1, 'sigma' : s1_1})

            with gl.name_scope("secondGaussian"):

                # y2 = gl.Parameter(len(data) // 2, name = 'yield', minVal = 0)

                g2_1 = gl.Gaussian({'mean' : m1_1, 'sigma' : s2_1})

            # initialFitYields1 = {g1.name : y1, g2.name : y2}
            initialFitFracs = {g1_1.name : f1_1}
            initialFitComponents = {g1_1.name : g1_1, g2_1.name : g2_1}

            # model = gl.Model(initialFitYields = initialFitYields1, initialFitComponents = initialFitComponents1, minVal = -10., maxVal = 10.)

            model1 = gl.Model(initialFitFracs = initialFitFracs, initialFitComponents = initialFitComponents, minVal = -10., maxVal = 10.)

        with gl.name_scope('signal2'):

            m1_2 = gl.Parameter(5.0, name = 'mean', minVal = 0, maxVal = 10)
            s1_2 = gl.Parameter(1.0, name = 'sigma1', minVal = 0., maxVal = 3.)
            s2_2 = gl.Parameter(3.0, name = 'sigma2', minVal = 1., maxVal = 5.)

            f1_2 = gl.Parameter(0.75, name = 'frac1', minVal = 0.0, maxVal = 1.0)

            with gl.name_scope("firstGaussian"):

                # y1 = gl.Parameter(len(data) // 2, name = 'yield', minVal = 0)

                g1_2 = gl.Gaussian({'mean' : m1_2, 'sigma' : s1_2})

            with gl.name_scope("secondGaussian"):

                # y2 = gl.Parameter(len(data) // 2, name = 'yield', minVal = 0)

                g2_2 = gl.Gaussian({'mean' : m1_2, 'sigma' : s2_2})

            # initialFitYields1 = {g1.name : y1, g2.name : y2}
            initialFitFracs = {g1_2.name : f1_2}
            initialFitComponents = {g1_2.name : g1_2, g2_2.name : g2_2}

            # model = gl.Model(initialFitYields = initialFitYields1, initialFitComponents = initialFitComponents1, minVal = -10., maxVal = 10.)

            model2 = gl.Model(initialFitFracs = initialFitFracs, initialFitComponents = initialFitComponents, minVal = -10., maxVal = 10.)

        y1 = gl.Parameter(2000, name = 'yield1')
        y2 = gl.Parameter(2000, name = 'yield2')

        initialFitYields = {model1.name : y1, model2.name : y2}
        initialFitComponents = {model1.name : model1, model2.name : model2}

        model = gl.Model(initialFitYields = initialFitYields, initialFitComponents = initialFitComponents, minVal = -10., maxVal = 10.)

    # fitter = gl.Fitter(model, backend = 'minuit')
    #
    # res = fitter.fit(data, verbose = True)
    #
    # # model.setTotalYield(len(data))
    # plotter = gl.Plotter(model, data)
    # plotter.plotDataModel()
    #
    # plt.savefig('plotWithFracs.pdf')

    fitter = gl.Fitter(model, backend = 'emcee')

    res = fitter.fit(data, verbose = True, nIterations = 2000)

    samples = res.chain[:, 1000:, :].reshape((-1, model.getNFloatingParameters()))
    c = corner.corner(samples, lw = 1.0)
    c.savefig('corner.pdf')
    plt.clf()

    # model.setTotalYield(len(data))
    plotter = gl.Plotter(model, data)
    plotter.plotDataModel()

    plt.savefig('plotWithFracs.pdf')

def mcModel():

    with gl.name_scope('mcSignal'):

        with gl.name_scope('g1'):

            m1 = gl.Parameter(5279., name = 'mean', minVal = 4200, maxVal = 5700) # These HAVE to be floats
            s1 = gl.Parameter(30., name = 'sigma', minVal = 5, maxVal = 60) # These HAVE to be floats

            signalGauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope('g2'):

            m2 = gl.Parameter(5279., name = 'mean', minVal = 4200, maxVal = 5700) # These HAVE to be floats
            s2 = gl.Parameter(90., name = 'sigma', minVal = 60, maxVal = 150) # These HAVE to be floats

            signalGauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

        f1 = gl.Parameter(0.5, name = 'frac1', minVal = 0, maxVal = 1.0)

    fitFracs = {signalGauss1.name : f1}
    fitComponents = {signalGauss1.name : signalGauss1, signalGauss2.name : signalGauss2}

    model = gl.Model(initialFitFracs = fitFracs, initialFitComponents = fitComponents, minVal = 4200, maxVal = 5700)

    return model

def crossfeedModel(signalYield, crossfeedYield, cbYield):

    with gl.name_scope('pidToy'):

        minParam =  gl.Parameter(4800., name = 'minVal', fixed = True)
        maxParam =  gl.Parameter(6000., name = 'maxVal', fixed = True)

        with gl.name_scope('signal'):

            with gl.name_scope('g1'):

                m1 = gl.Parameter(5265., name = 'mean', fixed = True)
                s1 = gl.Parameter(67.86, name = 'sigma', fixed = True)

                signalGauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

            with gl.name_scope('g2'):

                m2 = gl.Parameter(5257., name = 'mean', fixed = True)
                s2 = gl.Parameter(185.9, name = 'sigma', fixed = True)

                signalGauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

            y1 = gl.Parameter(signalYield // 2, name = 'g1Yield', minVal = -10., maxVal = signalYield)
            y2 = gl.Parameter(signalYield // 2, name = 'g2Yield', minVal = -10., maxVal = signalYield)
            #
            # # Would be good to have these represented using fractions, i.e., y1/y2
            #
            # fitYields = {signalGauss1.name : y1, signalGauss2.name : y2}
            # fitComponents = {signalGauss1.name : signalGauss1, signalGauss2.name : signalGauss2}
            # #
            # signalModel = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

            f1 = gl.Parameter(0.5665, name = 'frac1', fixed = True)

            fitFracs = {signalGauss1.name : f1}
            fitComponents = {signalGauss1.name : signalGauss1, signalGauss2.name : signalGauss2}

            signalModel = gl.Model(initialFitFracs = fitFracs, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

        with gl.name_scope('crossfeedKpipi'):

            with gl.name_scope('g1'):

                m1_cf = gl.Parameter(5021., name = 'mean', fixed = True)
                s1_cf = gl.Parameter(300., name = 'sigma', fixed = True)

                crossfeedKpipiGauss1 = gl.Gaussian({'mean' : m1_cf, 'sigma' : s1_cf})

            with gl.name_scope('g2'):

                m2_cf = gl.Parameter(5174., name = 'mean', fixed = True)
                s2_cf = gl.Parameter(97.77, name = 'sigma', fixed = True)

                crossfeedKpipiGauss2 = gl.Gaussian({'mean' : m2_cf, 'sigma' : s2_cf})

            y1 = gl.Parameter(crossfeedYield * 0.5, name = 'g1Yield', minVal = -10.)
            y2 = gl.Parameter(crossfeedYield * 0.5, name = 'g2Yield', minVal = -10.)

            # Would be good to have these represented using fractions, i.e., y1/y2

            # fitYields = {crossfeedKpipiGauss1.name : y1, crossfeedKpipiGauss2.name : y2}
            # fitComponents = {crossfeedKpipiGauss1.name : crossfeedKpipiGauss1, crossfeedKpipiGauss2.name : crossfeedKpipiGauss2}
            #
            # cfModel = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

            f1_cf = gl.Parameter(0.3373, name = 'frac1', fixed = True)

            fitFracs = {crossfeedKpipiGauss1.name : f1_cf}
            fitComponents = {crossfeedKpipiGauss1.name : crossfeedKpipiGauss1, crossfeedKpipiGauss2.name : crossfeedKpipiGauss2}

            cfModel = gl.Model(initialFitFracs = fitFracs, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

        with gl.name_scope('combinatorial'):

            a = gl.Parameter(-0.003, name = 'aExp', fixed = True) # ??

            combinatorial = gl.Exponential({'a' : a, 'min' : minParam, 'max' : maxParam})

    # ySig1 = gl.Parameter(signalYield // 2, name = 'sigYield1', minVal = -10.)
    # ySig2 = gl.Parameter(signalYield // 2, name = 'sigYield2', minVal = -10.)
    # yCF1 = gl.Parameter(crossfeedYield // 2, name = 'cfYield1', minVal = -10.)
    # yCF2 = gl.Parameter(crossfeedYield // 2, name = 'cfYield2', minVal = -10.)
    # yBkg = gl.Parameter(cbYield, name = 'cbYield', minVal = -10.)
    #
    # # Have sub-models defined using fractions of components, then report yields for models in addition to yields for individual components?
    # # AND FIX NORMALISATION!
    #
    # fitYields = {signalGauss1.name : ySig1, signalGauss2.name : ySig2, crossfeedKpipiGauss1.name : yCF1, crossfeedKpipiGauss2.name : yCF2, combinatorial.name : yBkg}
    # fitComponents = {signalGauss1.name : signalGauss1, signalGauss2.name : signalGauss2, crossfeedKpipiGauss1.name : crossfeedKpipiGauss1, crossfeedKpipiGauss2.name : crossfeedKpipiGauss2, combinatorial.name : combinatorial}
    #
    # model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

    ySig1 = gl.Parameter(signalYield, name = 'sigYield', minVal = -10.)
    yCF1 = gl.Parameter(crossfeedYield, name = 'cfYield', minVal = -10.)
    yBkg = gl.Parameter(cbYield, name = 'cbYield', minVal = -10.)

    # Have sub-models defined using fractions of components, then report yields for models in addition to yields for individual components?
    # AND FIX NORMALISATION!

    fitYields = {signalModel.name : ySig1, cfModel.name : yCF1, combinatorial.name : yBkg}
    fitComponents = {signalModel.name : signalModel, cfModel.name : cfModel, combinatorial.name : combinatorial}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

    return model

def testModel():

    with gl.name_scope('signal'):

        with gl.name_scope('g1'):

            m1 = gl.Parameter(5265., name = 'mean', fixed = True)
            s1 = gl.Parameter(67.86, name = 'sigma', fixed = True)

            signalGauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope('g2'):

            m2 = gl.Parameter(5257., name = 'mean', fixed = True)
            s2 = gl.Parameter(185.9, name = 'sigma', fixed = True)

            signalGauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

        y1 = gl.Parameter(1000., name = 'g1Yield', minVal = -100., maxVal = 2000.)
        y2 = gl.Parameter(1000., name = 'g2Yield', minVal = -100., maxVal = 2000.)

        fitYields = {signalGauss1.name : y1, signalGauss2.name : y2}

        # f1 = gl.Parameter(0.5, name = 'frac1', minVal = 0, maxVal = 1.0)

        # fitFracs = {signalGauss1.name : f1}
        fitComponents = {signalGauss1.name : signalGauss1, signalGauss2.name : signalGauss2}

    signalModel = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)
    # signalModel = gl.Model(initialFitFracs = fitFracs, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

    return signalModel

if __name__ == '__main__':
    # testPhysicalSimBF()

    # m1 = gl.Parameter(0.0, name = 'mean')
    # s1 = gl.Parameter(1.0, name = 'sigma')
    # g = gl.Gaussian({'mean' : m1, 'sigma' : s1})
    #
    # print(g.integral(-5, 1))

    # minParam =  gl.Parameter(0.0, name = 'minVal', fixed = True)
    # maxParam =  gl.Parameter(10.0, name = 'maxVal', fixed = True)
    #
    # u = gl.Uniform({'min' : minParam, 'max' : maxParam})
    #
    # print(u.integral(7, 30))

    # m1 = gl.Parameter(0.0, name = 'mean')
    # s1 = gl.Parameter(1.0, name = 'sigma')
    # a = gl.Parameter(1.0, name = 'a')
    # n = gl.Parameter(3.0, name = 'n')
    #
    # cb = gl.CrystalBall({'mean' : m1, 'sigma' : s1, 'a' : a, 'n' : n})
    #
    # print(cb.integral(-100, 0))

    # with gl.name_scope('massFit'):
    #     with gl.name_scope("expBkg"):
    #         yBkg = gl.Parameter(500, name = 'yieldExp', minVal = -1000)
    #
    #         minParam =  gl.Parameter(0.0, name = 'minVal', fixed = True)
    #         maxParam =  gl.Parameter(10.0, name = 'maxVal', fixed = True)
    #
    #         aExp = gl.Parameter(-0.2, name = 'aExp')
    #         # expBkg = gl.Exponential({'a' : aExp, 'min' : gl.Parameter(0.0, fixed = True), 'max' : gl.Parameter(10.0, fixed = True)})
    #         expBkg = gl.Exponential({'a' : aExp, 'min' : minParam, 'max' : maxParam})
    #
    #     with gl.name_scope("gaussSignal"):
    #         ySig = gl.Parameter(500, name = 'yieldSig', minVal = -1000)
    #
    #         m = gl.Parameter(5.0, name = 'mean')
    #         s = gl.Parameter(1.0, name = 'sigma')
    #
    #         g = gl.Gaussian({'mean' : m, 'sigma' : s})
    #
    # initialFitYields = {g.name : ySig, expBkg.name : yBkg}
    # initialFitComponents = {g.name : g, expBkg.name : expBkg}
    #
    # model = gl.Model(initialFitYields = initialFitYields, initialFitComponents = initialFitComponents, minVal = 0., maxVal = 10.)

    # testToy()

    # m1 = gl.Parameter(2.0, name = 'mean')
    # n = gl.Parameter(1.0, name = 'nu')
    #
    # st = gl.StudentsT({'mean' : m1, 'nu' : n})
    #
    # x = np.linspace(-8, 8, 1000)
    #
    # plt.plot(x, st.prob(x), lw = 1.0)
    # plt.savefig('st.pdf')

    # testWithFracs()
    #
    # model = mcModel()
    #
    # data = np.concatenate( (np.random.normal(5279., 30, 1000), np.random.normal(5279., 90, 1000)) )
    #
    # fitter = gl.Fitter(model, backend = 'emcee')
    #
    # res = fitter.fit(data, verbose = True, nIterations = 1000)
    #
    # plotter = gl.Plotter(model, data)
    # model.setTotalYield(len(data))
    #
    # plotter.plotDataModel(nDataBins = 35)
    # plt.savefig('mcSignal.pdf')
    # plt.clf()
    #
    # samples = res.chain[:, 100:, :].reshape((-1, model.getNFloatingParameters()))
    # c = corner.corner(samples, lw = 1.0)
    # c.savefig('corner.pdf')
    # plt.clf()

    # model = testModel()
    # model = crossfeedModel(10000., 2000., 10000.)
    # data = model.sample(minVal = 4800., maxVal = 6000.)
    #
    # fitter = gl.Fitter(model, backend = 'minuit')
    # res = fitter.fit(data, verbose = True)
    #
    # plotter = gl.Plotter(model, data)
    # plotter.plotDataModel(nDataBins = 50)
    # plt.savefig('testData.pdf')
    # plt.clf()

    minParam =  gl.Parameter(4800., name = 'minVal', fixed = True)
    maxParam =  gl.Parameter(6000., name = 'maxVal', fixed = True)

    with gl.name_scope('combinatorial'):

        a = gl.Parameter(-0.003, name = 'aExp', fixed = True) # ??

        combinatorial = gl.Exponential({'a' : a, 'min' : minParam, 'max' : maxParam})

        yBkg = gl.Parameter(5000, name = 'cbYield', minVal = -10.)

    with gl.name_scope('signal'):

        with gl.name_scope('g1'):

            m1 = gl.Parameter(5265., name = 'mean', fixed = True)
            s1 = gl.Parameter(67.86, name = 'sigma', fixed = True)

            signalGauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope('g2'):

            m2 = gl.Parameter(5257., name = 'mean', fixed = True)
            s2 = gl.Parameter(185.9, name = 'sigma', fixed = True)

            signalGauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

        f1 = gl.Parameter(0.85, name = 'frac1', fixed = True)

        fitFracs = {signalGauss1.name : f1}
        fitComponents = {signalGauss1.name : signalGauss1, signalGauss2.name : signalGauss2}

        signalModel = gl.Model(initialFitFracs = fitFracs, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

        # y1 = gl.Parameter(8000, name = 'y1', fixed = False, minVal = -100)
        # y2 = gl.Parameter(2000, name = 'y2', fixed = False, minVal = -100)

        # fitYields = {signalGauss1.name : y1, signalGauss2.name : y2}
        # fitComponents = {signalGauss1.name : signalGauss1, signalGauss2.name : signalGauss2}

        # signalModel = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

    ySig1 = gl.Parameter(1000000, name = 'sigYield', minVal = -10.)
    yBkg = gl.Parameter(50000, name = 'cbYield', minVal = -10.)

    # fitYields = {signalGauss1.name : y1, signalGauss2.name : y2, combinatorial.name : yBkg}
    # fitComponents = {signalGauss1.name : signalGauss1, signalGauss2.name : signalGauss2, combinatorial.name : combinatorial}

    fitYields = {signalModel.name : ySig1, combinatorial.name : yBkg}
    fitComponents = {signalModel.name : signalModel, combinatorial.name : combinatorial}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800., maxVal = 6000.)

    data = model.sample(minVal = 4800., maxVal = 6000.)

    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(data, verbose = False, nIterations = 250)
    pprint(model)
    plotter = gl.Plotter(model, data)
    plotter.plotDataModel(nDataBins = 50)
    plt.savefig('testData.pdf')
    plt.clf()
    #
    # fitter = gl.Fitter(model, backend = 'emcee')
    # res = fitter.fit(data, verbose = True, nIterations = 250, nWalkers = 10)
    #
    # plotter = gl.Plotter(model, data)
    # plotter.plotDataModel(nDataBins = 50)
    # plt.savefig('testData.pdf')
    # plt.clf()
    #
    # samples = res.chain[:, 50:, :].reshape((-1, model.getNFloatingParameters()))
    # c = corner.corner(samples, lw = 1.0)
    # c.savefig('corner.pdf')
    # plt.clf()
